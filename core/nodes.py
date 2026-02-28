import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    __slots__ = ('config', 'llm', 'tools', 'llm_with_tools', 'tools_map', '_cached_base_prompt')

    def __init__(self, config: AgentConfig, llm: BaseChatModel, tools: List[BaseTool], llm_with_tools: Optional[BaseChatModel] = None):
        self.config = config
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = llm_with_tools or llm
        
        # Оптимизация: O(1) доступ к инструментам вместо O(N) перебора списка
        self.tools_map = {t.name: t for t in tools}
        
        # Оптимизация: кэширование базового промпта (чтобы не читать с диска на каждый шаг)
        self._cached_base_prompt: Optional[str] = None

    # --- NODE: SUMMARIZE ---
    
    def _estimate_tokens(self, messages: List[BaseMessage]) -> int:
        """Грубая оценка токенов входящего контекста: сумма символов / 3.
        Учитывает как текстовый контент, так и аргументы вызовов инструментов."""
        total_chars = 0
        for m in messages:
            # 1. Текстовый контент (строка или мультимодальный список)
            content = m.content
            if isinstance(content, list):
                content = " ".join(str(part) for part in content)
            total_chars += len(str(content))
            
            # 2. Вызовы инструментов (JSON аргументы от LLM могут быть огромными)
            if hasattr(m, "tool_calls") and m.tool_calls:
                total_chars += sum(len(str(tc)) for tc in m.tool_calls)
                
        return total_chars // 3

    async def summarize_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")

        estimated_tokens = self._estimate_tokens(messages)

        if estimated_tokens <= self.config.summary_threshold:
            return {}

        logger.debug(f"📊 Context size: ~{estimated_tokens} tokens. Summarizing...")

        # Determine cut-off point
        idx = max(0, len(messages) - self.config.summary_keep_last)

        # Try to find a clean break at a HumanMessage
        for scan_idx in range(idx, len(messages)):
            if isinstance(messages[scan_idx], HumanMessage):
                idx = scan_idx
                break
        
        to_summarize = messages[:idx]
        
        # ЗАЩИТА: Если последние N сообщений сами по себе весят больше лимита,
        # мы не можем ничего сжать без потери недавнего контекста.
        if not to_summarize: 
            logger.warning(
                f"⚠ Context (~{estimated_tokens} tokens) exceeds threshold, "
                "but cannot summarize further without deleting the most recent active messages. "
                "Expanding context dynamically for this turn."
            )
            return {}

        history_text = self._format_history_for_summary(to_summarize)
        
        prompt = constants.SUMMARY_PROMPT_TEMPLATE.format(
            summary=summary,
            history_text=history_text
        )
        
        try:
            res = await self.llm.ainvoke(prompt)
            
            delete_msgs =[RemoveMessage(id=m.id) for m in to_summarize if m.id]
            logger.info(f"🧹 Summary: Removed {len(delete_msgs)} messages. Generated new summary.")
            
            # --- USER REQUESTED NOTIFICATION ---
            print(f"\n\033[93m[SYSTEM] 🧹 Сработала автоматическая суммаризация (контекст > {self.config.summary_threshold}). Старые сообщения сжаты.\033[0m\n")
            # -----------------------------------

            return {"summary": res.content, "messages": delete_msgs}
        except Exception as e:
            err_str = str(e)
            if "content_filter" in err_str or "Moderation Block" in err_str:
                 logger.warning("🧹 Summarization skipped due to Content Filter (False Positive). Continuing with full history.")
            else:
                logger.error(f"Summarization Error: {format_exception_friendly(e)}")
            return {}
            
    def _format_history_for_summary(self, messages: List[BaseMessage]) -> str:
        return "\n".join(
            f"{m.type}: {str(m.content)[:500]}{'... [truncated]' if len(str(m.content)) > 500 else ''}"
            for m in messages
        )

    # --- NODE: AGENT ---

    async def agent_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")
        
        sys_msg = self._build_system_message(summary, tools_available=bool(self.tools))
        full_context = [sys_msg] + messages
        
        response = await self._invoke_llm_with_retry(self.llm_with_tools, full_context)
        
        token_usage_update = {}
        if isinstance(response, AIMessage) and response.usage_metadata:
            token_usage_update = {"token_usage": response.usage_metadata}

        return {
            "messages": [response],
            **token_usage_update
        }

    # --- NODE: TOOLS ---

    async def tools_node(self, state: AgentState):
        self._check_invariants(state)

        messages = state["messages"]
        last_msg = messages[-1]
        
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {}
            
        final_messages =[]
        has_error = False
        
        # Оптимизация: собираем историю вызовов один раз, а не для каждого инструмента
        recent_calls =[]
        for m in reversed(messages[-20:]):
            if isinstance(m, AIMessage) and m.tool_calls:
                recent_calls.extend(m.tool_calls)

        for tool_call in last_msg.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call["id"]
            
            # Проверка на зацикливание (теперь работает моментально)
            loop_count = sum(1 for tc in recent_calls if tc.get("name") == t_name and tc.get("args") == t_args)
            
            if loop_count >= 3:
                content = format_error(ErrorType.LOOP_DETECTED, f"Loop detected. You have called '{t_name}' with these exact arguments 3 times in the recent history. Please try a different approach.")
                has_error = True
            else:
                content = await self._execute_tool(t_name, t_args)
            
            # Post-Tool Validation Layer
            validation_error = validate_tool_result(t_name, t_args, content)
            if validation_error:
                content = f"{content}\n\n{validation_error}"
                has_error = True

            if "ERROR[" in content:
                has_error = True

            limit = self.config.safety.max_tool_output
            content = truncate_output(content, limit, source=t_name)

            tool_msg = ToolMessage(content=content, tool_call_id=t_id, name=t_name)
            final_messages.append(tool_msg)

        if has_error and not self.config.strict_mode:
            reflection_msg = HumanMessage(content=constants.REFLECTION_PROMPT)
            final_messages.append(reflection_msg)

        return {"messages": final_messages}

    def _check_invariants(self, state: AgentState):
        if not self.config.debug: return
        steps = state.get("steps", 0)
        if steps < 0:
            logger.error(f"INVARIANT VIOLATION: steps ({steps}) < 0")

    async def _execute_tool(self, name: str, args: dict) -> str:
        # Быстрый поиск за O(1)
        tool = self.tools_map.get(name)
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

    def _get_base_prompt(self) -> str:
        """Ленивая загрузка и кэширование промпта для устранения дискового I/O"""
        if self._cached_base_prompt is None:
            if self.config.prompt_path.exists():
                self._cached_base_prompt = self.config.prompt_path.read_text("utf-8")
            else:
                self._cached_base_prompt = (
                    "You are an autonomous AI agent.\n"
                    "Reason in English, Reply in Russian.\n"
                    "Date: {{current_date}}"
                )
        return self._cached_base_prompt

    def _build_system_message(self, summary: str, tools_available: bool = True) -> SystemMessage:
        raw_prompt = self._get_base_prompt()
        
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
                if "auto" in err_str and "tool choice" in err_str and "requires" in err_str:
                    logger.warning("⚠ Server does not support 'auto' tool choice. Falling back to chat-only mode.")
                    current_llm = self.llm 
                    # Безопасное копирование контекста
                    context = list(context)
                    if isinstance(context[0], SystemMessage):
                        context[0] = SystemMessage(content=str(context[0].content) + "\n\nWARNING: Tools are disabled due to server configuration error.")
                    continue

                logger.warning(f"LLM Error (Attempt {attempt+1}): {e}")
                if attempt == 2:
                    friendly_error = format_exception_friendly(e)
                    return AIMessage(content=f"System Error: API request failed after 3 retries. \n{friendly_error}")
                await asyncio.sleep(1)
        return AIMessage(content="System Error: Unknown failure.")