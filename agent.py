import json
import asyncio
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# --- LANGCHAIN & LANGGRAPH ---
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    BaseMessage, SystemMessage, RemoveMessage, HumanMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# --- CORE MODULES ---
from core.config import AgentConfig
from core.state import AgentState
from core.utils import AgentUtils
from tools.tool_registry import ToolRegistry
from core.tool_validator import validate_tool_execution 

# --- OPTIONAL CORE MODULES ---
from dotenv import load_dotenv

try:
    from core.tool_sanitizer import ToolSanitizer
except ImportError:
    class ToolSanitizer:
        @staticmethod
        def sanitize_tool_calls(tc): pass
        
try:
    from core.safety_guard import SafetyGuard
except ImportError:
    class SafetyGuard:
        ENABLED = False
        @classmethod
        def is_unsafe_write(cls, *args): return False

try:
    from core.logging_config import setup_logging
    logger = setup_logging() 
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("agent")


# ==========================================
# ГРАФ АГЕНТА (WORKFLOW)
# ==========================================

class AgentWorkflow:
    def __init__(self):
        load_dotenv()
        self.config = AgentConfig()
        self.utils = AgentUtils()
        self.tool_registry = ToolRegistry(self.config)
        
        self.llm: Optional[BaseChatModel] = None
        self.llm_with_tools: Optional[BaseChatModel] = None

    async def initialize_resources(self):
        logger.info(f"Initializing agent: [bold cyan]{self.config.provider}[/]", extra={"markup": True})
        
        self.llm = self.config.get_llm()
        await self.tool_registry.load_all()
        
        can_use_tools = self.config.check_tool_support()
        
        if self.tool_registry.tools and can_use_tools:
            try:
                self.llm_with_tools = self.llm.bind_tools(self.tool_registry.tools)
                logger.info("🛠️ Tools bound to LLM successfully.")
            except Exception as e:
                logger.error(f"Failed to bind tools: {e}")
                self.llm_with_tools = self.llm
        else:
            if not can_use_tools:
                logger.debug("⚠️ Tools disabled: Model does not support tool calling.")
            self.llm_with_tools = self.llm

    @property
    def tools(self) -> List[BaseTool]:
        return self.tool_registry.tools

    # --- NODES ---

    async def _summarize_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")

        if len(messages) <= self.config.summary_threshold:
            return {}

        idx = len(messages) - self.config.summary_keep_last
        while idx < len(messages) and idx > 0:
            if isinstance(messages[idx], HumanMessage):
                break
            idx += 1
        
        to_summarize = messages[:idx]
        if not to_summarize: return {}

        history_text = "\n".join([f"{m.type}: {m.content}" for m in to_summarize])
        
        prompt = (
            f"Current memory context:\n<previous_context>\n{summary}\n</previous_context>\n\n"
            f"New events:\n{history_text}\n\n"
            "Update <previous_context>. Keep only key facts, decisions, and results. "
            "Remove chit-chat. Return only the updated context text."
        )
        
        try:
            res = await self.llm.ainvoke(prompt)
            delete_msgs = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
            logger.info(f"🧹 Summary: Removed {len(delete_msgs)} messages.")
            return {"summary": res.content, "messages": delete_msgs}
        except Exception as e:
            logger.error(f"Summarization Error: {e}")
            return {}

    async def _tools_and_validate_node(self, state: AgentState):
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
            
            tool = next((t for t in self.tools if t.name == t_name), None)
            content = ""
            
            # --- 1. EXECUTION ---
            if not tool:
                content = f"Error: Tool '{t_name}' not found."
            else:
                try:
                    raw_result = await tool.ainvoke(t_args)
                    content = str(raw_result)
                except Exception as e:
                    content = f"Error: {str(e)}"

            if not content.strip():
                content = "Error: Tool returned empty response."

            tool_msg = ToolMessage(content=content, tool_call_id=t_id, name=t_name)
            final_messages.append(tool_msg)

            # --- 2. VALIDATION ---
            result = validate_tool_execution(tool_msg, t_args, t_name)
            
            if not result["is_valid"]:
                logger.debug(f"Tool Error ({t_name}): {result['error_message']}")
                validation_errors.append(f"- Tool '{t_name}' failed: {result['error_message']}")
                
                retry_count = tool_retries.get(t_name, 0)
                if result["retry_needed"]:
                    if retry_count < 2:
                        tool_retries[t_name] = retry_count + 1
                        should_force_retry = True
            else:
                if t_name in tool_retries: del tool_retries[t_name]

        # --- 3. RESPONSE ---
        if validation_errors:
            if should_force_retry:
                advice = "INSTRUCTION: Invalid arguments. CALL THE TOOL AGAIN with corrected parameters."
            else:
                advice = (
                    "INSTRUCTION: The action failed due to environment state (e.g., file not found). "
                    "DO NOT RETRY the same action immediately. "
                    "Check available files (list_directory), search for the correct path, or create the file first."
                )
            
            final_messages.append(SystemMessage(content=f"SYSTEM ALERT:\n{validation_errors}\n{advice}"))

        return {
            "messages": final_messages,
            "tool_retries": tool_retries
        }

    async def _agent_node(self, state: AgentState):
        messages = state["messages"]
        tools_available = (self.llm_with_tools != self.llm)
        
        sys_msg = self._build_system_message(state.get("summary", ""), tools_available)
        full_context = [sys_msg] + messages
        
        # 1. Вызов LLM
        response = await self._invoke_llm_with_retry(full_context)
        
        last_tool_call = None
        
        # 2. Quality Gate
        if SafetyGuard.is_unsafe_write(response, full_context):
            response = SystemMessage(
                content="STOP. You are trying to write a file without valid data from search/fetch. "
                        "Perform a search first to get actual content."
            )
            
        # 3. Обработка инструментов
        elif isinstance(response, AIMessage) and response.tool_calls:
            ToolSanitizer.sanitize_tool_calls(response.tool_calls)
            last_tool_call = response.tool_calls[0]
            
            for tc in response.tool_calls:
                if tc['name'] in ['write_file', 'save_file']:
                    raw_path = str(tc['args'].get('path', '')).strip()
                    if len(raw_path) < 2 or re.match(r'^[\.,\-_:;\'" ]+$', raw_path):
                        logger.debug(f"🛡️ Quality Gate: Rejecting garbage filename '{raw_path}'")
                        response = SystemMessage(
                            content=f"SYSTEM ERROR: The filename '{raw_path}' is invalid. "
                                    "Please RETRY with a meaningful filename."
                        )
                        last_tool_call = None
                        break 
                        
        # 4. Патч токенов
        if isinstance(response, AIMessage):
            self._patch_token_usage(response, full_context)
        
        # 5. Loop guard logic
        if isinstance(response, AIMessage) and response.tool_calls:
            last_msg = messages[-1] if messages else None
            if isinstance(last_msg, ToolMessage):
                current_tool = response.tool_calls[0]['name']
                last_tool = last_msg.name
                if current_tool == "write_file" and last_tool == "write_file":
                    logger.warning("🛑 Loop Guard: Blocked repetitive write_file.")
                    response = AIMessage(
                        content="System: File already written. Stop overwriting."
                    )

        return {
            "messages": [response],
            "last_tool_call": last_tool_call
        }
        
    async def _loop_guard_node(self, state: AgentState):
        return {"messages": [AIMessage(content="🛑 **Auto-Stop**: Max steps limit reached.")]}

    # --- HELPERS ---

    def _build_system_message(self, summary: str, tools_available: bool = True) -> SystemMessage:
        if self.config.prompt_path.exists():
            raw_prompt = self.config.prompt_path.read_text("utf-8")
        else:
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

    async def _invoke_llm_with_retry(self, context: List[BaseMessage]) -> AIMessage:
        FATAL_ERRORS = ["401", "unauthorized", "quota", "billing", "context_length_exceeded"]

        for attempt in range(3):
            try:
                response = await self.llm_with_tools.ainvoke(context)
                if not response.content and not response.tool_calls:
                    raise ValueError("Empty response from LLM")
                return response

            except Exception as e:
                error_str = str(e).lower()
                if any(err in error_str for err in FATAL_ERRORS):
                    logger.error(f"🛑 Fatal LLM Error: {e}")
                    return AIMessage(content=f"System Error: API refused request ({e})")

                if attempt < 2:
                    logger.debug(f"⚠️ LLM Crash (Attempt {attempt+1}): {e}. Retrying...")
                    await asyncio.sleep(1)
                    continue
                
                logger.error(f"💀 All retries failed: {e}")
            
        return AIMessage(content=f"**System Failure**: Multiple API crashes.")
      
    # --- ИСПРАВЛЕНО: метод внутри класса и с правильным отступом ---
    def _patch_token_usage(self, response: AIMessage, context: List[BaseMessage]):
        """
        Robust token usage normalization for LangChain / LangGraph
        Works with OpenAI, aggregators (MegaLLM, Pollinations), and fallback.
        """

        # 1. Проверяем, что usage_metadata уже валиден
        usage = response.usage_metadata or {}
        if (
            isinstance(usage, dict)
            and usage.get("input_tokens", 0) > 0
            and usage.get("output_tokens", 0) > 0
        ):
            return

        # 2. Безопасно собираем возможные источники
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
            if not isinstance(c, dict):
                continue
            if "prompt_tokens" in c or "completion_tokens" in c:
                raw_usage = c
                break

        # 3. Применяем usage от провайдера
        if raw_usage:
            input_tokens = raw_usage.get("prompt_tokens", 0)
            output_tokens = raw_usage.get("completion_tokens", 0)
            total_tokens = raw_usage.get(
                "total_tokens", input_tokens + output_tokens
            )

            if input_tokens > 0 or output_tokens > 0:
                response.usage_metadata = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "token_source": "Provider"
                }
                return

        # 4. Fallback: ручной подсчёт
        input_tokens = self.utils.estimate_payload_tokens(context, self.tools)

        output_content = response.content
        if isinstance(output_content, list):
            output_content = " ".join(str(x) for x in output_content)

        output_tokens = self.utils.count_tokens(str(output_content))

        if response.tool_calls:
            output_tokens += self.utils.count_tokens(
                json.dumps(response.tool_calls, default=str)
            )

        response.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "token_source": "Manual"
        }
    
    # --- GRAPH BUILDER ---

    def build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("loop_guard", self._loop_guard_node)
        workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})
        
        workflow.add_node("tools", self._tools_and_validate_node)
        
        tools_enabled = bool(self.tools) and self.config.check_tool_support()

        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "update_step") 
        workflow.add_edge("update_step", "agent")

        def should_continue(state: AgentState):
            steps = state.get("steps", 0)
            messages = state.get("messages", [])

            if steps >= self.config.max_loops:
                logger.debug(f"🛑 Loop Guard: {steps} steps.")
                return "loop_guard"

            if not messages: return "agent"
            last_msg = messages[-1]

            if isinstance(last_msg, SystemMessage): return "agent"
            if tools_enabled and isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                return "tools"
            if isinstance(last_msg, ToolMessage): return "agent"
            return END

        destinations = ["tools", "loop_guard", "agent", END] if tools_enabled else ["loop_guard", END]
        workflow.add_conditional_edges("agent", should_continue, destinations)

        if tools_enabled:
            workflow.add_edge("tools", "agent")

        workflow.add_edge("loop_guard", END)

        return workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    async def main():
        wf = AgentWorkflow()
        await wf.initialize_resources()
        print(f"✅ Agent Ready. Tools: {len(wf.tools)}")

    asyncio.run(main())