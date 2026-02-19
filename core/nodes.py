import json
import logging
import asyncio
from typing import List, Optional, Dict, Any

from langchain_core.messages import (
    BaseMessage, SystemMessage, RemoveMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from core.state import AgentState
from core.config import AgentConfig
from core import constants
from core.cli_utils import format_exception_friendly
from core.validation import validate_tool_result
from core.utils import truncate_output
from core.errors import format_error, ErrorType

logger = logging.getLogger("agent")

class AgentNodes:
    def __init__(self, config: AgentConfig, llm: BaseChatModel, tools: List[BaseTool], llm_with_tools: Optional[BaseChatModel] = None):
        self.config = config
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = llm_with_tools or llm

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
            err_str = str(e)
            if "content_filter" in err_str or "Moderation Block" in err_str:
                 logger.warning(f"🧹 Summarization skipped due to Content Filter (False Positive). Continuing with full history.")
            else:
                logger.error(f"Summarization Error: {format_exception_friendly(e)}")
            return {}

    def _format_history_for_summary(self, messages: List[BaseMessage]) -> str:
        parts = []
        for m in messages:
            content = str(m.content)
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            parts.append(f"{m.type}: {content}")
        return "\n".join(parts)

    # --- NODE: AGENT ---

    async def agent_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")
        
        # Build System Prompt
        sys_msg = self._build_system_message(summary, tools_available=bool(self.tools))
        
        # Prepare context
        full_context = [sys_msg] + messages
        
        # Invoke LLM
        response = await self._invoke_llm_with_retry(self.llm_with_tools, full_context)
        
        # Token Updates
        token_usage_update = {}
        if isinstance(response, AIMessage):
            if response.usage_metadata:
                 token_usage_update = {"token_usage": response.usage_metadata}

        return {
            "messages": [response],
            **token_usage_update
        }

    # --- NODE: TOOLS ---

    async def tools_node(self, state: AgentState):
        # Invariants Check (Debug Mode)
        self._check_invariants(state)

        messages = state["messages"]
        last_msg = messages[-1]
        
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {}
            
        final_messages = []
        has_error = False

        for tool_call in last_msg.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call["id"]
            
            # Execute
            if self._check_loop(messages, t_name, t_args):
                content = format_error(ErrorType.LOOP_DETECTED, f"Loop detected. You have called '{t_name}' with these exact arguments 3 times in the recent history. Please try a different approach.")
                has_error = True
            else:
                content = await self._execute_tool(t_name, t_args)
            
            # Post-Tool Validation Layer
            validation_error = validate_tool_result(t_name, t_args, content)
            if validation_error:
                content = f"{content}\n\n{validation_error}"
                has_error = True

            # Check for error signature (unified format)
            if "ERROR[" in content:
                has_error = True

            # Truncate output (Safety Layer)
            # Use SafetyPolicy limit from config (defaults to 5000)
            limit = self.config.safety.max_tool_output
            content = truncate_output(content, limit, source=t_name)

            tool_msg = ToolMessage(content=content, tool_call_id=t_id, name=t_name)
            final_messages.append(tool_msg)

        # Reflection Hint (Conditional)
        # If strict_mode is ON, we disable auto-reflection hint.
        if has_error and not self.config.strict_mode:
            # Inject reflection prompt to guide the agent
            reflection_msg = HumanMessage(content=constants.REFLECTION_PROMPT)
            final_messages.append(reflection_msg)

        return {
            "messages": final_messages,
        }

    def _check_invariants(self, state: AgentState):
        """Debug-only validator for graph state."""
        if not self.config.debug:
            return
            
        steps = state.get("steps", 0)
        
        if steps < 0:
            logger.error(f"INVARIANT VIOLATION: steps ({steps}) < 0")
        
    def _check_loop(self, messages: List[BaseMessage], tool_name: str, tool_args: dict) -> bool:
        """
        Detects if the exact same tool call has been made repeatedly (>= 3 times) in recent history.
        """
        count = 0
        # Check last 20 messages to catch loops
        for m in reversed(messages[-20:]):
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.get("name") == tool_name and tc.get("args") == tool_args:
                        count += 1
        
        return count >= 3

    async def _execute_tool(self, name: str, args: dict) -> str:
        tool = next((t for t in self.tools if t.name == name), None)
        if not tool:
            return format_error(ErrorType.NOT_FOUND, f"Tool '{name}' not found.")
        try:
            raw_result = await tool.ainvoke(args)
            content = str(raw_result)
            if not content.strip():
                return format_error(ErrorType.EXECUTION, "Tool returned empty response.")
            return content
        except Exception as e:
            return format_error(ErrorType.EXECUTION, str(e))

    # --- HELPERS ---

    def _build_system_message(self, summary: str, tools_available: bool = True) -> SystemMessage:
        # 1. Load prompt from config path
        if self.config.prompt_path.exists():
            raw_prompt = self.config.prompt_path.read_text("utf-8")
        else:
            raw_prompt = (
                "You are an autonomous AI agent.\n"
                "Reason in English, Reply in Russian.\n"
                "Date: {{current_date}}"
            )
        
        from datetime import datetime
        from pathlib import Path
        
        prompt = raw_prompt.replace("{{current_date}}", datetime.now().strftime("%Y-%m-%d"))
        prompt = prompt.replace("{{cwd}}", str(Path.cwd()))
        
        if self.config.strict_mode:
            prompt += "\nNOTE: STRICT MODE ENABLED. Be precise. No guessing."
        
        if not tools_available:
            prompt += "\nNOTE: You are in CHAT-ONLY mode. Tools are disabled."
             
        if summary:
            prompt += f"\n\n<memory>\n{summary}\n</memory>"
            
        return SystemMessage(content=prompt)

    async def _invoke_llm_with_retry(self, llm, context: List[BaseMessage]) -> AIMessage:
        current_llm = llm
        
        for attempt in range(3):
            try:
                response = await current_llm.ainvoke(context)
                if not response.content and not response.tool_calls:
                    raise ValueError("Empty response from LLM")
                return response
            except Exception as e:
                err_str = str(e)
                # Handle specific tool choice errors from vLLM/OpenAI-compatible servers
                if "auto" in err_str and "tool choice" in err_str and "requires" in err_str:
                    logger.warning("⚠ Server does not support 'auto' tool choice. Falling back to chat-only mode.")
                    # Fallback to base LLM (without tools) for this request
                    current_llm = self.llm 
                    # Add system note about disabled tools
                    context = [m for m in context] # Copy
                    if isinstance(context[0], SystemMessage):
                        context[0] = SystemMessage(content=str(context[0].content) + "\n\nWARNING: Tools are disabled due to server configuration error.")
                    continue

                logger.warning(f"LLM Error (Attempt {attempt+1}): {e}")
                if attempt == 2:
                    friendly_error = format_exception_friendly(e)
                    return AIMessage(content=f"System Error: API request failed after 3 retries. \n{friendly_error}")
                await asyncio.sleep(1)
        return AIMessage(content="System Error: Unknown failure.")
