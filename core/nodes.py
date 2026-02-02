import json
import logging
import re
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any, Union

from langchain_core.messages import (
    BaseMessage, SystemMessage, RemoveMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.state import AgentState
from core.config import AgentConfig
from core.utils import AgentUtils
from core.tool_validator import validate_tool_execution
from core.tool_sanitizer import ToolSanitizer
from core.safety_guard import SafetyGuard
from core import constants

logger = logging.getLogger("agent")

class IntentClassification(BaseModel):
    intent: Literal["read_only", "write_action"] = Field(
        description="User's intent: 'read_only' (search, query, view) or 'write_action' (create, edit, delete, patch, fix, modify)."
    )
    reasoning: str = Field(description="Short explanation of why this intent was chosen.")

class AgentNodes:
    def __init__(self, config: AgentConfig, llm: BaseChatModel, tools: List[BaseTool], tool_buckets: Dict[str, List[str]], llm_with_tools: Optional[BaseChatModel] = None):
        self.config = config
        self.llm = llm
        self.tools = tools
        self.tool_buckets = tool_buckets
        self.llm_with_tools = llm_with_tools or llm
        self.utils = AgentUtils()

    # --- NODE: SUMMARIZE ---
    
    async def summarize_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")

        if len(messages) <= self.config.summary_threshold:
            return {}

        # Determine cut-off point
        idx = len(messages) - self.config.summary_keep_last
        if idx < 0: idx = 0

        # Try to find a clean break at a HumanMessage
        scan_idx = idx
        while scan_idx < len(messages):
            if isinstance(messages[scan_idx], HumanMessage):
                idx = scan_idx
                break
            scan_idx += 1
        
        to_summarize = messages[:idx]
        if not to_summarize: return {}

        # Format history for LLM
        history_text = self._format_history_for_summary(to_summarize)
        
        prompt = constants.SUMMARY_PROMPT_TEMPLATE.format(
            summary=summary,
            history_text=history_text
        )
        
        try:
            res = await self.llm.ainvoke(prompt)
            delete_msgs = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
            logger.info(f"🧹 Summary: Removed {len(delete_msgs)} messages.")
            return {"summary": res.content, "messages": delete_msgs}
        except Exception as e:
            logger.error(f"Summarization Error: {e}")
            return {}

    def _format_history_for_summary(self, messages: List[BaseMessage]) -> str:
        parts = []
        for m in messages:
            content = str(m.content)
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            parts.append(f"{m.type}: {content}")
        return "\n".join(parts)

    # --- NODE: TOOL FILTER ---

    async def tool_filter_node(self, state: AgentState):
        """
        Determines which tools are allowed for the next step.
        Implements: Token Budget Check, Recovery Mode, Action Phase, and Intent Classification.
        """
        last_msg = state["messages"][-1]

        # 1. Token Budget Check
        if self._is_budget_exhausted(last_msg):
            return self._apply_budget_filter(state)

        # 2. Global Bypass
        if not self.config.enable_tool_filtering:
            return {"allowed_tools": None}

        # 3. Determine Phase
        phase = await self._determine_phase(state)
        
        # 4. Tool Gating
        if phase in ["action", "intent_action"]:
            logger.debug(f"🔓 Filter: {phase.upper()} (All tools allowed)")
            return {"allowed_tools": None}
        else:
            logger.debug(f"🔒 Filter: {phase.upper()} ({len(self.tool_buckets['safe'])} safe tools allowed)")
            return {"allowed_tools": self.tool_buckets["safe"]}

    def _is_budget_exhausted(self, last_msg: BaseMessage) -> bool:
        return isinstance(last_msg, SystemMessage) and "TOKEN BUDGET ALERT" in str(last_msg.content)

    def _apply_budget_filter(self, state: AgentState) -> dict:
        blocked = {"web_search", "fetch_content", "deep_search", "batch_web_search", "crawl_site"}
        current_allowed = state.get("allowed_tools")
        
        if current_allowed is None:
            all_tools = self.tool_buckets["safe"] + self.tool_buckets["write"]
            allowed = [t for t in all_tools if t not in blocked]
        else:
            allowed = [t for t in current_allowed if t not in blocked]
            
        logger.debug("🛡 Filter: Budget Alert -> Blocking retrieval tools")
        return {"allowed_tools": allowed}

    async def _determine_phase(self, state: AgentState) -> str:
        messages = state["messages"]
        last_msg = messages[-1] if messages else None

        # A. Recovery
        if isinstance(last_msg, SystemMessage):
            text = str(last_msg.content).upper()
            if "SYSTEM ALERT" in text or "DO NOT RETRY" in text:
                return "recovery"

        # B. Ongoing Action
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                content = str(m.content)
                if content and not content.startswith("Error"):
                    return "action"
            if isinstance(m, HumanMessage):
                break 

        # C. Intent Classification (LLM)
        if isinstance(last_msg, HumanMessage):
            return await self._classify_intent(last_msg, messages)
        
        return "exploration"

    async def _classify_intent(self, last_msg: HumanMessage, messages: List[BaseMessage]) -> str:
        text = last_msg.content.lower()
        
        # Fast Path: Check for explicit tool names
        write_tool_names = [name.lower() for name in self.tool_buckets["write"]]
        if any(t_name in text for t_name in write_tool_names):
            logger.debug("⚡ Filter: Fast Path (Tool mentioned) -> Write Allowed")
            return "intent_action"

        # LLM Classification
        try:
            classifier = self.llm.with_structured_output(IntentClassification)
            system_prompt = SystemMessage(content=constants.INTENT_CLASSIFICATION_SYSTEM_PROMPT)
            
            # Fix: Use only the last message to avoid invalid message chains (orphaned ToolMessages)
            # context_msgs = [system_prompt] + messages[-3:] 
            context_msgs = [system_prompt, last_msg]
            
            classification = await classifier.ainvoke(context_msgs)
            
            if classification.intent == "write_action":
                logger.info(f"🧠 Intent: WRITE detected. Reason: {classification.reasoning}")
                return "intent_action"
            else:
                logger.info(f"🧠 Intent: READ detected. Reason: {classification.reasoning}")
                return "exploration"
                
        except Exception as e:
            # Downgraded to debug to avoid polluting the UI. Fallback works fine.
            logger.debug(f"⚠️ Intent Classifier Failed: {e}. Falling back to keywords.")
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> str:
        if any(hint in text for hint in constants.INTENT_HINTS):
            return "intent_action"
        return "exploration"

    # --- NODE: TOOLS & VALIDATE ---

    async def tools_and_validate_node(self, state: AgentState):
        messages = state["messages"]
        last_msg = messages[-1]
        
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {}
            
        final_messages = []
        validation_errors = []
        should_force_retry = False
        tool_retries = state.get("tool_retries", {}).copy()

        for tool_call in last_msg.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call["id"]
            
            # Execute
            content = await self._execute_tool(t_name, t_args)
            tool_msg = ToolMessage(content=content, tool_call_id=t_id, name=t_name)

            # Validate
            result = validate_tool_execution(tool_msg, t_args, t_name)
            
            if not result["is_valid"]:
                logger.debug(f"Tool Error ({t_name}): {result['error_message']}")
                
                retry_count = tool_retries.get(t_name, 0)
                if result["retry_needed"]:
                    if retry_count < 3:
                        tool_retries[t_name] = retry_count + 1
                        should_force_retry = True
                        validation_errors.append(f"- Tool '{t_name}' failed (Attempt {retry_count+1}/3): {result['error_message']}")
                    else:
                        tool_msg.content = f"SYSTEM BLOCK: Too many consecutive errors. Output truncated."
                        validation_errors.append(f"- STOP: {t_name} blocked due to repeated failures.")
                        if t_name in tool_retries: del tool_retries[t_name]
                else:
                    validation_errors.append(f"- Tool '{t_name}' returned: {result['error_message']}")
            else:
                if t_name in tool_retries: del tool_retries[t_name]

            final_messages.append(tool_msg)

        if validation_errors:
            sys_msg = self._create_error_system_message(validation_errors, should_force_retry)
            final_messages.append(sys_msg)

        return {
            "messages": final_messages,
            "tool_retries": tool_retries
        }

    async def _execute_tool(self, name: str, args: dict) -> str:
        tool = next((t for t in self.tools if t.name == name), None)
        if not tool:
            return f"Error: Tool '{name}' not found."
        try:
            raw_result = await tool.ainvoke(args)
            content = str(raw_result)
            if not content.strip():
                return "Error: Tool returned empty response."
            return content
        except Exception as e:
            return f"Error: {str(e)}"

    def _create_error_system_message(self, errors: List[str], retry: bool) -> SystemMessage:
        advice = (
            "INSTRUCTION: Arguments invalid or tool failed. Review the error and TRY AGAIN with corrected parameters."
            if retry else
            "INSTRUCTION: Action failed repeatedly or is impossible. DO NOT RETRY the same tool with the same arguments. Try a different approach."
        )
        err_text = "\n".join(errors)
        return SystemMessage(content=f"SYSTEM ALERT:\n{err_text}\n{advice}")

    # --- NODE: REFLECTION ---

    async def reflection_node(self, state: AgentState):
        messages = state["messages"]
        last_msg = messages[-1] if messages else None

        if not isinstance(last_msg, SystemMessage):
            return {}

        if "SYSTEM ALERT" not in str(last_msg.content):
            return {}

        reflection = (
            "REFLECTION:\n"
            "- The previous action failed.\n"
            "- Identify WHY it failed (invalid args, missing data, wrong tool).\n"
            "- DO NOT repeat the same tool with the same arguments.\n"
            "- Change strategy (e.g. use a different tool or gather data first).\n"
            "Reply with a brief plan, then act."
        )
        return {"messages": [SystemMessage(content=reflection)]}
        
    # --- NODE: TOKEN BUDGET ---

    async def token_budget_guard_node(self, state: AgentState):
        used = state.get("token_used", 0)
        budget = state.get("token_budget", 0) or self.config.token_budget

        if budget and used >= budget:
            return {
                "messages": [SystemMessage(
                    content=(
                        "TOKEN BUDGET ALERT:\n"
                        "- Context budget is exhausted.\n"
                        "- Stop gathering new information.\n"
                        "- Use existing data to finish the task.\n"
                        "- DO NOT call search or fetch tools."
                    )
                )],
                "token_budget": budget 
            }
        
        if not state.get("token_budget"):
             return {"token_budget": budget}
             
        return {}
        
    # --- NODE: AGENT ---

    async def agent_node(self, state: AgentState):
        self._log_token_usage(state)
        
        messages = state["messages"]
        
        # Dynamic Binding
        allowed = state.get("allowed_tools")
        if allowed is not None:
            selected_tools = [t for t in self.tools if t.name in allowed]
            current_llm = self.llm.bind_tools(selected_tools)
        else:
            current_llm = self.llm_with_tools

        tools_available = (current_llm != self.llm)
        
        sys_msg = self._build_system_message(state.get("summary", ""), tools_available)
        full_context = [sys_msg] + messages
        
        # Invoke LLM
        response = await self._invoke_llm_with_retry(current_llm, full_context)
        
        last_tool_call = None
        
        # Safety Check
        if SafetyGuard.is_unsafe_write(response, full_context):
            response = SystemMessage(
                content="STOP. You are trying to write a file without valid data from search/fetch. "
                        "Perform a search first to get actual content."
            )
            
        elif isinstance(response, AIMessage) and response.tool_calls:
            ToolSanitizer.sanitize_tool_calls(response.tool_calls)
            last_tool_call = response.tool_calls[0]
            
            # Filename check
            for tc in response.tool_calls:
                if tc['name'] in ['write_file', 'save_file']:
                    if not self._validate_filename(tc['args']):
                        logger.debug("🛡️ Quality Gate: Rejecting garbage filename")
                        response = SystemMessage(content="SYSTEM ERROR: Invalid filename. Please RETRY.")
                        last_tool_call = None
                        break 
        
        # Token Updates
        token_used_update = {}
        if isinstance(response, AIMessage):
            self._patch_token_usage(response, full_context)
            if response.usage_metadata:
                current_used = state.get("token_used", 0)
                added = response.usage_metadata.get("input_tokens", 0)
                token_used_update = {"token_used": current_used + added}

        # Loop Guard for repeated writes
        if isinstance(response, AIMessage) and response.tool_calls:
            if self._is_repeated_write(messages, response):
                response = AIMessage(content="System: File already written. Stop overwriting.")

        return {
            "messages": [response],
            "last_tool_call": last_tool_call,
            **token_used_update
        }

    def _log_token_usage(self, state: AgentState):
        used = state.get("token_used", 0)
        budget = state.get("token_budget", 0) or self.config.token_budget
        if budget > 0:
            percent = (used / budget) * 100
            logger.debug(
                f"💰 Token Usage: [bold cyan]{used}[/] / [bold cyan]{budget}[/] ({percent:.1f}%)", 
                extra={"markup": True}
            )

    def _validate_filename(self, args: dict) -> bool:
        raw_path = str(args.get('path', '')).strip()
        if len(raw_path) < 2 or re.match(r'^[\.,\-_:;\'" ]+$', raw_path):
            return False
        return True

    def _is_repeated_write(self, messages: List[BaseMessage], response: AIMessage) -> bool:
        last_msg = messages[-1] if messages else None
        if isinstance(last_msg, ToolMessage):
            current_tool = response.tool_calls[0]['name']
            last_tool = last_msg.name
            if current_tool == "write_file" and last_tool == "write_file":
                logger.warning("🛑 Loop Guard: Blocked repetitive write_file.")
                return True
        return False

    async def loop_guard_node(self, state: AgentState):
        return {"messages": [AIMessage(content="🛑 **Auto-Stop**: Max steps limit reached.")]}

    # --- HELPERS (LLM & PROMPT) ---

    def _build_system_message(self, summary: str, tools_available: bool = True) -> SystemMessage:
        # 1. Try configured path
        if self.config.prompt_path.exists():
            prompt_file = self.config.prompt_path
        # 2. Try BASE_DIR fallback (if config path is wrong/relative)
        elif (constants.BASE_DIR / "prompt.txt").exists():
            prompt_file = constants.BASE_DIR / "prompt.txt"
        else:
            prompt_file = None

        if prompt_file:
            raw_prompt = prompt_file.read_text("utf-8")
            logger.info(f"✅ System prompt loaded from: {prompt_file} ({len(raw_prompt)} chars)")
        else:
            logger.warning(f"⚠️ System prompt not found at: {self.config.prompt_path}. Using fallback.")
            raw_prompt = (
                "You are an autonomous AI agent.\n"
                "Reason in English, Reply in Russian.\n"
                "Date: {{current_date}}\nCWD: {{cwd}}"
            )
        
        prompt = raw_prompt.replace("{{current_date}}", datetime.now().strftime("%Y-%m-%d"))
        prompt = prompt.replace("{{cwd}}", str(Path.cwd()))
        
        if not tools_available:
            prompt += "\nNOTE: You are in CHAT-ONLY mode. Tools are disabled for this session."
        elif self.config.use_long_term_memory:
             prompt += "\nUse memory tools (recall_facts/remember_fact) when necessary."
             
        if summary:
            prompt += f"\n\n<memory>\n{summary}\n</memory>"
            
        return SystemMessage(content=prompt)

    async def _invoke_llm_with_retry(self, llm, context: List[BaseMessage]) -> AIMessage:
        for attempt in range(3):
            try:
                response = await llm.ainvoke(context)
                if not response.content and not response.tool_calls:
                    raise ValueError("Empty response from LLM")
                return response

            except Exception as e:
                error_str = str(e).lower()
                if any(err in error_str for err in constants.FATAL_ERRORS):
                    logger.error(f"🛑 Fatal LLM Error: {e}")
                    return AIMessage(content=f"System Error: API refused request ({e})")

                if attempt < 2:
                    logger.debug(f"⚠️ LLM Crash (Attempt {attempt+1}): {e}. Retrying...")
                    await asyncio.sleep(1)
                    continue
                
                logger.error(f"💀 All retries failed: {e}")
            
        return AIMessage(content=f"**System Failure**: Multiple API crashes.")
      
    def _patch_token_usage(self, response: AIMessage, context: List[BaseMessage]):
        """Ensures usage_metadata is present in the response."""
        usage = response.usage_metadata or {}
        
        if (isinstance(usage, dict) and usage.get("input_tokens", 0) > 0):
            return

        meta = response.response_metadata or {}
        add_kwargs = response.additional_kwargs or {}

        candidates = [
            meta.get("usage"),
            meta.get("token_usage"),
            add_kwargs.get("usage"),
            add_kwargs.get("token_usage"),
            meta.get("body", {}).get("usage") if isinstance(meta.get("body"), dict) else None,
        ]

        raw_usage = None
        for c in candidates:
            if isinstance(c, dict) and ("prompt_tokens" in c or "completion_tokens" in c):
                raw_usage = c
                break
        
        if raw_usage:
            input_tokens = raw_usage.get("prompt_tokens", 0)
            output_tokens = raw_usage.get("completion_tokens", 0)
            
            if input_tokens is None: input_tokens = 0
            if output_tokens is None: output_tokens = 0
            
            total_tokens = raw_usage.get("total_tokens", input_tokens + output_tokens)

            if input_tokens > 150_000:
                logger.warning(f"⚠️ Suspiciously high token count ({input_tokens}). Possible parsing error or garbage from provider. Raw: {raw_usage}")
                pass 
            elif input_tokens > 0 or output_tokens > 0:
                response.usage_metadata = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "token_source": "Provider"
                }
                return

        input_tokens = self.utils.estimate_payload_tokens(context, self.tools)
        
        output_content = response.content
        if isinstance(output_content, list):
            output_content = " ".join(str(x) for x in output_content)

        output_tokens = self.utils.count_tokens(str(output_content))
        if response.tool_calls:
            output_tokens += self.utils.count_tokens(json.dumps(response.tool_calls, default=str))

        response.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "token_source": "Manual"
        }
