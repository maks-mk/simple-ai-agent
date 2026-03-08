import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import print as rprint
from rich.panel import Panel

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
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
    __slots__ = ("config", "llm", "tools", "llm_with_tools", "tools_map", "_cached_base_prompt")

    # Only these tools are allowed to run in parallel in a single tool-call batch.
    # Any unknown or mutating tool keeps sequential execution for safety.
    PARALLEL_SAFE_TOOL_NAMES = frozenset(
        {
            "read_file",
            "list_directory",
            "search_in_file",
            "search_in_directory",
            "tail_file",
            "find_file",
            "web_search",
            "fetch_content",
            "batch_web_search",
            "get_public_ip",
            "lookup_ip_info",
            "get_system_info",
            "get_local_network_info",
            "find_process_by_port",
        }
    )
    # Read-only tools can be called repeatedly while an agent verifies edits/results.
    READ_ONLY_LOOP_TOLERANT_TOOL_NAMES = frozenset(
        {
            "read_file",
            "search_in_file",
            "search_in_directory",
            "tail_file",
            "find_file",
            "list_directory",
            "web_search",
            "fetch_content",
            "batch_web_search",
        }
    )

    def __init__(
        self,
        config: AgentConfig,
        llm: BaseChatModel,
        tools: List[BaseTool],
        llm_with_tools: Optional[BaseChatModel] = None,
    ):
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

        prompt = constants.SUMMARY_PROMPT_TEMPLATE.format(summary=summary, history_text=history_text)

        try:
            res = await self.llm.ainvoke(prompt)

            delete_msgs = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
            logger.info(f"🧹 Summary: Removed {len(delete_msgs)} messages. Generated new summary.")

            # --- КРАСИВОЕ УВЕДОМЛЕНИЕ ЧЕРЕЗ RICH ---
            rprint(
                Panel(
                    f"[dim]Контекст превысил порог в {self.config.summary_threshold} токенов.\n"
                    f"Старые сообщения ({len(delete_msgs)} шт.) успешно сжаты в память.[/dim]",
                    title="[bold yellow]🧹 Авто-суммаризация памяти[/]",
                    border_style="yellow",
                    padding=(0, 2),
                    expand=False,
                )
            )
            # --------------------------------------

            return {"summary": res.content, "messages": delete_msgs}
        except Exception as e:
            err_str = str(e)
            if "content_filter" in err_str or "Moderation Block" in err_str:
                logger.warning(
                    "🧹 Summarization skipped due to Content Filter (False Positive). Continuing with full history."
                )
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
        critic_feedback = (state.get("critic_feedback") or "").strip()
        current_task = state.get("current_task") or self._derive_current_task(messages)

        tools_available = bool(self.tools) and self.config.model_supports_tools
        sys_msg = self._build_system_message(summary, tools_available=tools_available)
        full_context: List[BaseMessage] = [sys_msg]

        if critic_feedback:
            full_context.append(
                SystemMessage(
                    content=(
                        "INTERNAL CRITIC FEEDBACK:\n"
                        f"{critic_feedback}\n"
                        "Use this only as an internal hint. Do not mention the critic."
                    )
                )
            )

        full_context.extend(messages)

        response = await self._invoke_llm_with_retry(self.llm_with_tools, full_context)

        token_usage_update = {}
        if isinstance(response, AIMessage) and response.usage_metadata:
            token_usage_update = {"token_usage": response.usage_metadata}

        has_tool_calls = bool(
            tools_available and isinstance(response, AIMessage) and getattr(response, "tool_calls", None)
        )

        return {
            "messages": [response],
            "current_task": current_task,
            "critic_feedback": "",
            "critic_status": "",
            "critic_source": "" if has_tool_calls else "agent",
            **token_usage_update,
        }

    # --- NODE: CRITIC ---

    async def critic_node(self, state: AgentState):
        messages = state.get("messages", [])
        current_task = (state.get("current_task") or self._derive_current_task(messages)).strip()
        summary = state.get("summary", "")
        critic_source = (state.get("critic_source") or "agent").strip().lower()
        trace = self._format_trace_for_critic(messages)

        prompt = constants.CRITIC_PROMPT_TEMPLATE.format(
            current_task=current_task or "No explicit task provided.",
            summary=summary or "No prior summary.",
            source=critic_source or "agent",
            trace=trace,
        )

        response = await self._invoke_llm_with_retry(self.llm, [SystemMessage(content=prompt)])
        parsed = self._parse_critic_response(str(response.content))

        if not parsed:
            status, reason, next_step = self._infer_critic_fallback(messages, critic_source)
            logger.info(
                f"Critic returned malformed verdict. Falling back to heuristic {status}: {reason}"
            )
        else:
            status, reason, next_step = parsed

        if status == "FINISHED":
            if critic_source == "tools":
                feedback = (
                    "Task appears complete based on the tool results. "
                    "Provide a concise final answer to the user in Russian without calling more tools "
                    "unless a final verification is still genuinely required."
                )
            else:
                feedback = ""
        else:
            next_step_line = next_step if next_step and next_step != "NONE" else "perform an explicit verification step"
            feedback = f"Task incomplete. Reason: {reason}. Next step: {next_step_line}."

        return {
            "critic_status": status,
            "critic_feedback": feedback,
            "current_task": current_task,
        }

    # --- NODE: TOOLS ---

    async def tools_node(self, state: AgentState):
        self._check_invariants(state)

        messages = state["messages"]
        last_msg = messages[-1]

        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {}

        final_messages: List[ToolMessage] = []
        has_error = False

        # Оптимизация: собираем историю вызовов один раз, а не для каждого инструмента.
        # ВАЖНО: исключаем последний AI message, чтобы текущий вызов не считался "повтором".
        recent_calls = []
        history_window = self.config.effective_tool_loop_window
        history_slice = messages[-(history_window + 1):-1] if history_window > 0 else messages[:-1]
        for m in reversed(history_slice):
            if isinstance(m, AIMessage) and m.tool_calls:
                recent_calls.extend(m.tool_calls)

        tool_calls = list(last_msg.tool_calls)

        if self._can_parallelize_tool_calls(tool_calls):
            processed = await asyncio.gather(
                *(self._process_tool_call(tool_call, recent_calls) for tool_call in tool_calls)
            )
            for tool_msg, had_error in processed:
                final_messages.append(tool_msg)
                has_error = has_error or had_error
        else:
            for tool_call in tool_calls:
                tool_msg, had_error = await self._process_tool_call(tool_call, recent_calls)
                final_messages.append(tool_msg)
                has_error = has_error or had_error

        if has_error and not self.config.strict_mode:
            reflection_msg = HumanMessage(content=constants.REFLECTION_PROMPT)
            final_messages.append(reflection_msg)

        return {"messages": final_messages, "critic_source": "tools"}

    def _can_parallelize_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> bool:
        if len(tool_calls) < 2:
            return False

        # Unknown tools are treated as potentially state-changing.
        return all(tc.get("name") in self.PARALLEL_SAFE_TOOL_NAMES for tc in tool_calls)

    async def _process_tool_call(
        self, tool_call: Dict[str, Any], recent_calls: List[Dict[str, Any]]
    ) -> Tuple[ToolMessage, bool]:
        t_name = tool_call["name"]
        t_args = tool_call["args"]
        t_id = tool_call["id"]

        had_error = False

        # Проверка на зацикливание
        loop_count = sum(
            1 for tc in recent_calls if tc.get("name") == t_name and tc.get("args") == t_args
        )

        loop_limit = (
            self.config.effective_tool_loop_limit_readonly
            if t_name in self.READ_ONLY_LOOP_TOLERANT_TOOL_NAMES
            else self.config.effective_tool_loop_limit_mutating
        )

        if loop_count >= loop_limit:
            content = format_error(
                ErrorType.LOOP_DETECTED,
                f"Loop detected. You have called '{t_name}' with these exact arguments {loop_limit} times in the recent history. Please try a different approach.",
            )
            had_error = True
        else:
            content = await self._execute_tool(t_name, t_args)

        # Post-Tool Validation Layer
        validation_error = validate_tool_result(t_name, t_args, content)
        if validation_error:
            content = f"{content}\n\n{validation_error}"
            had_error = True

        if "ERROR[" in content:
            had_error = True

        limit = self.config.safety.max_tool_output
        content = truncate_output(content, limit, source=t_name)

        return ToolMessage(content=content, tool_call_id=t_id, name=t_name), had_error

    def _check_invariants(self, state: AgentState):
        if not self.config.debug:
            return
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

    def _derive_current_task(self, messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                content = self._stringify_content(message.content).strip()
                if content and content != constants.REFLECTION_PROMPT:
                    return content
        return ""

    def _format_trace_for_critic(self, messages: List[BaseMessage], max_messages: int = 12) -> str:
        if not messages:
            return "No recent messages."

        trace_messages = messages[-max_messages:]
        formatted = [self._format_message_for_critic(message) for message in trace_messages]
        trace = "\n".join(part for part in formatted if part).strip()
        if not trace:
            return "No recent messages."
        return trace[:12000]

    def _format_message_for_critic(self, message: BaseMessage) -> str:
        content = self._stringify_content(message.content)
        content = self._compact_text(content, 1200)

        if isinstance(message, ToolMessage):
            label = f"tool[{message.name or 'unknown'}]"
            return f"{label}: {content or '[empty tool output]'}"

        if isinstance(message, AIMessage):
            parts: List[str] = []
            if content:
                parts.append(f"assistant: {content}")
            for tool_call in getattr(message, "tool_calls", []) or []:
                name = tool_call.get("name", "unknown")
                args = self._compact_text(str(tool_call.get("args", {})), 300)
                parts.append(f"assistant_tool_call[{name}]: {args}")
            return "\n".join(parts) if parts else "assistant: [empty response]"

        if isinstance(message, HumanMessage):
            role = "system_hint" if content == constants.REFLECTION_PROMPT else "user"
            return f"{role}: {content or '[empty message]'}"

        return f"{message.type}: {content or '[empty message]'}"

    def _stringify_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _compact_text(self, text: str, limit: int) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 15] + "... [truncated]"

    def _parse_critic_response(self, text: str) -> Optional[Tuple[str, str, str]]:
        parsed: Dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().upper()
            if key in {"STATUS", "REASON", "NEXT_STEP"} and key not in parsed:
                parsed[key] = value.strip()

        status = parsed.get("STATUS", "").upper().rstrip(".")
        upper_text = text.upper()
        if status not in {"FINISHED", "INCOMPLETE"}:
            if "INCOMPLETE" in upper_text:
                status = "INCOMPLETE"
            elif "FINISHED" in upper_text or "COMPLETE" in upper_text or "DONE" in upper_text:
                status = "FINISHED"
            else:
                return None

        reason = parsed.get("REASON", "").strip()
        if not reason:
            reason = self._extract_reason_from_freeform_text(text, status)

        next_step = parsed.get("NEXT_STEP", "").strip()
        if not next_step:
            next_step = "NONE" if status == "FINISHED" else "continue with the next obvious missing step"

        return status, reason, next_step.upper() if next_step.upper() == "NONE" else next_step

    def _extract_reason_from_freeform_text(self, text: str, status: str) -> str:
        cleaned = " ".join(text.split()).strip(" -:")
        if cleaned:
            return self._compact_text(cleaned, 180)
        if status == "FINISHED":
            return "Task appears completed."
        return "Task appears incomplete."

    def _infer_critic_fallback(
        self, messages: List[BaseMessage], critic_source: str
    ) -> Tuple[str, str, str]:
        last_ai = self._get_last_ai_message(messages)
        recent_tools = self._get_recent_tool_messages(messages)

        if any(self._is_tool_error_message(msg) for msg in recent_tools):
            return "INCOMPLETE", "Recent tool execution reported an explicit error.", "fix the failed step"

        if last_ai and self._message_indicates_incomplete(self._stringify_content(last_ai.content)):
            return "INCOMPLETE", "The latest assistant message suggests work is still pending.", "continue with the remaining step"

        if critic_source == "tools" and recent_tools:
            return "FINISHED", "Recent tool results indicate the requested action completed successfully.", "NONE"

        if last_ai and not getattr(last_ai, "tool_calls", None):
            content = self._stringify_content(last_ai.content)
            if content.strip():
                return "FINISHED", "The latest assistant response appears to deliver the requested result.", "NONE"

        return "INCOMPLETE", "Completion is still unclear from the latest trace.", "continue only if something obvious is still missing"

    def _get_last_ai_message(self, messages: List[BaseMessage]) -> Optional[AIMessage]:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def _get_recent_tool_messages(self, messages: List[BaseMessage], limit: int = 6) -> List[ToolMessage]:
        recent: List[ToolMessage] = []
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                recent.append(message)
                if len(recent) >= limit:
                    break
            elif recent:
                break
        recent.reverse()
        return recent

    def _is_tool_error_message(self, message: ToolMessage) -> bool:
        content = self._stringify_content(message.content)
        normalized = content.strip().lower()
        return (
            getattr(message, "status", "") == "error"
            or normalized.startswith(("error", "ошибка", "error["))
            or "error[" in normalized
            or "traceback" in normalized
        )

    def _message_indicates_incomplete(self, text: str) -> bool:
        normalized = " ".join(text.lower().split())
        if not normalized:
            return False

        incomplete_markers = (
            "task incomplete",
            "incomplete",
            "next step",
            "need to",
            "still need",
            "remaining",
            "not finished",
            "failed",
            "error",
            "не заверш",
            "ещё нужно",
            "еще нужно",
            "осталось",
            "следующий шаг",
            "нужно ",
            "требуется",
            "не удалось",
            "ошибка",
            "не готов",
            "продолжу",
            "теперь найду",
            "теперь создам",
        )
        uncertainty_markers = (
            "похоже",
            "кажется",
            "вероятно",
            "probably",
            "maybe",
            "perhaps",
            "likely",
            "seems",
        )

        if any(marker in normalized for marker in incomplete_markers):
            return True
        if any(marker in normalized for marker in uncertainty_markers):
            return True
        return False

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
        max_attempts = max(1, self.config.max_retries)
        retry_delay = max(0, self.config.retry_delay)

        for attempt in range(max_attempts):
            try:
                response = await current_llm.ainvoke(context)
                if not response.content and not response.tool_calls:
                    raise ValueError("Empty response from LLM")
                return response
            except Exception as e:
                err_str = str(e)
                if "auto" in err_str and "tool choice" in err_str and "requires" in err_str:
                    logger.warning(
                        "⚠ Server does not support 'auto' tool choice. Falling back to chat-only mode."
                    )
                    current_llm = self.llm
                    # Безопасное копирование контекста
                    context = list(context)
                    if isinstance(context[0], SystemMessage):
                        context[0] = SystemMessage(
                            content=str(context[0].content)
                            + "\n\nWARNING: Tools are disabled due to server configuration error."
                        )
                    continue

                is_fatal = self._is_fatal_llm_error(e)
                logger.warning(f"LLM Error (Attempt {attempt+1}/{max_attempts}): {e}")

                if is_fatal:
                    logger.error(f"Fatal LLM error detected. Aborting request: {e}")
                    raise

                if attempt == max_attempts - 1:
                    raise

                await asyncio.sleep(retry_delay)

        raise RuntimeError("LLM retry loop exited unexpectedly without a response.")

    def _is_fatal_llm_error(self, error: Exception) -> bool:
        err_str = " ".join(str(error).lower().split())
        fatal_markers = (
            "insufficient_balance",
            "insufficient account balance",
            "invalid_api_key",
            "incorrect api key",
            "authentication failed",
            "unauthorized",
            "forbidden",
            "permission denied",
            "billing",
            "payment required",
            "error code: 401",
            "error code: 402",
            "error code: 403",
        )
        return any(marker in err_str for marker in fatal_markers)


