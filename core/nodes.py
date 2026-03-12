import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from langgraph.types import interrupt

from core.state import AgentState
from core.config import AgentConfig
from core import constants
from core.run_logger import JsonlRunLogger
from core.tool_policy import ToolMetadata, default_tool_metadata
from core.tool_results import parse_tool_execution_result
from core.validation import validate_tool_result
from core.utils import truncate_output
from core.errors import format_error, ErrorType
from core.message_utils import compact_text, is_error_text, is_tool_message_error, stringify_content
from core.text_utils import format_exception_friendly

logger = logging.getLogger("agent")


class AgentNodes:
    __slots__ = (
        "config",
        "llm",
        "tools",
        "llm_with_tools",
        "tools_map",
        "tool_metadata",
        "run_logger",
        "_cached_base_prompt",
    )

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
        tool_metadata: Optional[Dict[str, ToolMetadata]] = None,
        run_logger: Optional[JsonlRunLogger] = None,
    ):
        self.config = config
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = llm_with_tools or llm

        # Оптимизация: O(1) доступ к инструментам вместо O(N) перебора списка
        self.tools_map = {t.name: t for t in tools}
        self.tool_metadata = tool_metadata or {}
        self.run_logger = run_logger

        # Оптимизация: кэширование базового промпта (чтобы не читать с диска на каждый шаг)
        self._cached_base_prompt: Optional[str] = None

    def _log_run_event(self, state: AgentState | None, event_type: str, **payload: Any) -> None:
        if not self.run_logger:
            return
        session_id = None if state is None else state.get("session_id")
        self.run_logger.log_event(session_id, event_type, **payload)

    def _metadata_for_tool(self, tool_name: str) -> ToolMetadata:
        return self.tool_metadata.get(tool_name, default_tool_metadata(tool_name))

    def _tool_is_read_only(self, tool_name: str) -> bool:
        metadata = self._metadata_for_tool(tool_name)
        return metadata.read_only and not metadata.mutating and not metadata.destructive

    def _tool_requires_approval(self, tool_name: str) -> bool:
        if not self.config.enable_approvals:
            return False
        metadata = self._metadata_for_tool(tool_name)
        return metadata.requires_approval or metadata.destructive or metadata.mutating

    def tool_calls_require_approval(self, tool_calls: List[Dict[str, Any]]) -> bool:
        return any(self._tool_requires_approval((tool_call.get("name") or "unknown_tool")) for tool_call in tool_calls)

    # --- NODE: SUMMARIZE ---

    def _estimate_tokens(self, messages: List[BaseMessage]) -> int:
        """Грубая оценка токенов входящего контекста: сумма символов / 2.
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

        return total_chars // 2

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
            self._log_run_event(
                state,
                "summary_compacted",
                estimated_tokens=estimated_tokens,
                removed_messages=len(delete_msgs),
                summarized_messages=len(to_summarize),
            )

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

    def _build_agent_context(
        self,
        messages: List[BaseMessage],
        summary: str,
        critic_feedback: str,
        tools_available: bool,
        unresolved_tool_error: str,
    ) -> List[BaseMessage]:
        sanitized_messages = self._sanitize_messages_for_model(messages)
        full_context: List[BaseMessage] = [
            self._build_system_message(summary, tools_available=tools_available)
        ]
        safety_overlay = self._build_safety_overlay(tools_available=tools_available)
        if safety_overlay:
            full_context.append(SystemMessage(content=safety_overlay))
        if unresolved_tool_error:
            full_context.append(
                SystemMessage(
                    content=constants.UNRESOLVED_TOOL_ERROR_PROMPT_TEMPLATE.format(
                        error_summary=unresolved_tool_error
                    )
                )
            )
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
        full_context.extend(sanitized_messages)
        if critic_feedback and sanitized_messages and isinstance(sanitized_messages[-1], AIMessage):
            full_context.append(
                HumanMessage(
                    content=(
                        "Continue the task from the latest incomplete assistant attempt. "
                        "Use the critic feedback above, do not repeat the same incomplete answer, "
                        "and only stop when the requested work is actually finished."
                    )
                )
            )
        return full_context

    def _sanitize_messages_for_model(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        sanitized: List[BaseMessage] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                content = stringify_content(message.content).strip()
                if content == constants.REFLECTION_PROMPT:
                    continue
            sanitized.append(message)
        return sanitized

    def _build_safety_overlay(self, tools_available: bool) -> str:
        if not tools_available:
            return ""
        overlay_lines: List[str] = []
        if self.config.enable_approvals:
            overlay_lines.append(
                "SAFETY POLICY: Mutating or destructive tools may require explicit user approval before execution."
            )
        if self.config.enable_shell_tool:
            overlay_lines.append(
                "SAFETY POLICY: Shell execution is high risk. Prefer safer project-local tools whenever possible."
            )
        return "\n".join(overlay_lines).strip()

    def _build_agent_result(
        self,
        response: AIMessage,
        current_task: str,
        tools_available: bool,
    ) -> Dict[str, Any]:
        token_usage_update = {}
        if getattr(response, "usage_metadata", None):
            token_usage_update = {"token_usage": response.usage_metadata}

        has_tool_calls = False
        protocol_error = ""

        if isinstance(response, AIMessage):
            t_calls = list(getattr(response, "tool_calls", []))
            invalid_calls = list(getattr(response, "invalid_tool_calls", []))

            missing_fields = [
                tc for tc in t_calls
                if not tc.get("id") or not tc.get("name")
            ]
            if missing_fields or invalid_calls:
                protocol_error = self._build_tool_protocol_error(missing_fields, invalid_calls)
                response = AIMessage(
                    content=self._merge_protocol_error_into_content(response.content, protocol_error),
                    additional_kwargs=response.additional_kwargs,
                    response_metadata=response.response_metadata,
                    usage_metadata=response.usage_metadata,
                    id=response.id,
                )
                t_calls = []

            has_tool_calls = bool(tools_available and t_calls)

        return {
            "messages": [response],
            "current_task": current_task,
            "critic_feedback": "",
            "critic_status": "",
            "critic_source": "" if has_tool_calls else "agent",
            "pending_approval": None,
            "last_tool_error": "",
            "last_tool_result": "",
            **token_usage_update,
        }

    def _merge_protocol_error_into_content(self, content: Any, protocol_error: str) -> str:
        base_text = stringify_content(content).strip()
        if not protocol_error:
            return base_text
        if not base_text:
            return protocol_error
        return f"{base_text}\n\n{protocol_error}"

    def _build_tool_protocol_error(
        self,
        missing_fields: List[Dict[str, Any]],
        invalid_calls: List[Dict[str, Any]],
    ) -> str:
        details = []
        if missing_fields:
            details.append(
                f"{len(missing_fields)} tool call(s) were missing required 'name' or 'id' fields"
            )
        if invalid_calls:
            details.append(
                f"{len(invalid_calls)} tool call(s) had invalid arguments and could not be parsed"
            )
        joined = "; ".join(details) if details else "Malformed tool call payload received."
        return (
            "INTERNAL TOOL PROTOCOL ERROR: "
            f"{joined}. Do not invent tool names or IDs. "
            "If tools are still needed, issue a fresh valid tool call."
        )
    
    def _build_critic_feedback(
        self,
        status: str,
        reason: str,
        next_step: str,
        critic_source: str,
    ) -> str:
        if status == "FINISHED":
            if critic_source == "tools":
                return (
                    "Task appears complete based on the tool results. "
                    "Provide a concise final answer to the user in Russian without calling more tools "
                    "unless a final verification is still genuinely required."
                )
            return ""

        next_step_line = next_step if next_step and next_step != "NONE" else "perform an explicit verification step"
        return f"Task incomplete. Reason: {reason}. Next step: {next_step_line}."

    # --- NODE: AGENT ---

    async def agent_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")
        critic_feedback = (state.get("critic_feedback") or "").strip()
        current_task = state.get("current_task") or self._derive_current_task(messages)
        unresolved_tool_error = self._get_unresolved_tool_error(messages)
        self._log_run_event(
            state,
            "agent_node_start",
            run_id=state.get("run_id", ""),
            step=state.get("steps", 0),
            current_task=current_task,
        )

        tools_available = bool(self.tools) and self.config.model_supports_tools
        full_context = self._build_agent_context(
            messages,
            summary,
            critic_feedback,
            tools_available,
            unresolved_tool_error,
        )
        response = await self._invoke_llm_with_retry(
            self.llm_with_tools,
            full_context,
            state=state,
            node_name="agent",
        )
        result = self._build_agent_result(response, current_task, tools_available)
        self._log_run_event(
            state,
            "agent_node_end",
            run_id=state.get("run_id", ""),
            tool_calls=len(getattr(response, "tool_calls", []) or []),
            content_preview=compact_text(stringify_content(response.content), 240),
        )
        return result

    # --- NODE: CRITIC ---

    async def critic_node(self, state: AgentState):
        messages = state.get("messages", [])
        current_task = (state.get("current_task") or self._derive_current_task(messages)).strip()
        summary = state.get("summary", "")
        critic_source = (state.get("critic_source") or "agent").strip().lower()
        trace = self._format_trace_for_critic(messages)
        unresolved_tool_error = self._get_unresolved_tool_error(messages) or "None."

        prompt = constants.CRITIC_PROMPT_TEMPLATE.format(
            current_task=current_task or "No explicit task provided.",
            summary=summary or "No prior summary.",
            unresolved_tool_error=unresolved_tool_error,
            source=critic_source or "agent",
            trace=trace,
        )

        response = await self._invoke_llm_with_retry(
            self.llm,
            [SystemMessage(content=prompt)],
            state=state,
            node_name="critic",
        )
        parsed = self._parse_critic_response(str(response.content))

        if not parsed:
            status, reason, next_step = self._infer_critic_fallback(messages, critic_source)
            logger.info(
                f"Critic returned malformed verdict. Falling back to heuristic {status}: {reason}"
            )
        else:
            status, reason, next_step = parsed

        status, reason, next_step = self._apply_unresolved_tool_error_guard(
            status,
            reason,
            next_step,
            messages,
        )
        self._log_run_event(
            state,
            "critic_verdict",
            run_id=state.get("run_id", ""),
            status=status,
            reason=reason,
            next_step=next_step,
            source=critic_source,
        )

        return {
            "critic_status": status,
            "critic_feedback": self._build_critic_feedback(status, reason, next_step, critic_source),
            "current_task": current_task,
        }

    async def approval_node(self, state: AgentState):
        messages = state.get("messages", [])
        if not messages:
            return {"pending_approval": None}

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {"pending_approval": None}

        protected_calls = []
        for tool_call in last_msg.tool_calls:
            tool_name = tool_call.get("name") or "unknown_tool"
            if not self._tool_requires_approval(tool_name):
                continue
            metadata = self._metadata_for_tool(tool_name)
            protected_calls.append(
                {
                    "id": tool_call.get("id") or "",
                    "name": tool_name,
                    "args": tool_call.get("args") or {},
                    "policy": metadata.to_dict(),
                }
            )

        if not protected_calls:
            return {"pending_approval": None}

        payload = {
            "kind": "tool_approval",
            "message": "Approve protected tool execution?",
            "tools": protected_calls,
            "run_id": state.get("run_id", ""),
            "session_id": state.get("session_id", ""),
        }
        self._log_run_event(
            state,
            "approval_requested",
            run_id=state.get("run_id", ""),
            tool_names=[tool["name"] for tool in protected_calls],
        )
        decision = interrupt(payload)
        approved = self._approval_decision_is_approved(decision)
        approval_state = {
            "approved": approved,
            "decision": decision,
            "tool_call_ids": [tool["id"] for tool in protected_calls if tool["id"]],
            "tool_names": [tool["name"] for tool in protected_calls],
        }
        self._log_run_event(
            state,
            "approval_resolved",
            run_id=state.get("run_id", ""),
            approved=approved,
            tool_names=approval_state["tool_names"],
        )
        return {"pending_approval": approval_state}

    def _approval_decision_is_approved(self, decision: Any) -> bool:
        if isinstance(decision, bool):
            return decision
        if isinstance(decision, dict):
            if "approved" in decision:
                return bool(decision.get("approved"))
            action = str(decision.get("action", "")).strip().lower()
            return action in {"approve", "approved", "yes", "y"}
        return bool(decision)

    # --- NODE: TOOLS ---

    async def tools_node(self, state: AgentState):
        self._check_invariants(state)

        messages = state["messages"]
        last_msg = messages[-1]

        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {}

        final_messages: List[ToolMessage] = []
        has_error = False
        last_error = ""
        last_result = ""
        approval_state = state.get("pending_approval") or {}

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
                *(self._process_tool_call(tool_call, recent_calls, state, approval_state) for tool_call in tool_calls)
            )
            for tool_msg, had_error in processed:
                final_messages.append(tool_msg)
                has_error = has_error or had_error
                parsed = parse_tool_execution_result(tool_msg.content)
                if parsed.ok:
                    last_result = parsed.message
                else:
                    last_error = parsed.message
        else:
            for tool_call in tool_calls:
                tool_msg, had_error = await self._process_tool_call(tool_call, recent_calls, state, approval_state)
                final_messages.append(tool_msg)
                has_error = has_error or had_error
                parsed = parse_tool_execution_result(tool_msg.content)
                if parsed.ok:
                    last_result = parsed.message
                else:
                    last_error = parsed.message

        return {
            "messages": final_messages,
            "critic_source": "tools",
            "critic_feedback": "",
            "pending_approval": None,
            "last_tool_error": last_error,
            "last_tool_result": last_result,
        }

    def _can_parallelize_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> bool:
        if len(tool_calls) < 2:
            return False

        return all(self._tool_is_read_only(tc.get("name") or "unknown_tool") for tc in tool_calls)

    async def _process_tool_call(
        self,
        tool_call: Dict[str, Any],
        recent_calls: List[Dict[str, Any]],
        state: AgentState,
        approval_state: Dict[str, Any],
    ) -> Tuple[ToolMessage, bool]:
        # Безопасное извлечение с фоллбеками
        t_name = tool_call.get("name") or "unknown_tool"
        t_args = tool_call.get("args") or {}
    
        # Генерируем фейковый ID, если LLM забыла его указать, чтобы Pydantic не упал
        t_id = tool_call.get("id")
        if not t_id:
            t_id = f"call_missing_{uuid.uuid4().hex[:8]}"

        had_error = False
        metadata = self._metadata_for_tool(t_name)

        if self._tool_requires_approval(t_name) and not self._tool_call_is_approved(t_id, approval_state):
            content = format_error(
                ErrorType.ACCESS_DENIED,
                f"Execution of '{t_name}' was cancelled by approval policy.",
            )
            self._log_run_event(
                state,
                "tool_call_denied",
                run_id=state.get("run_id", ""),
                tool_name=t_name,
                tool_args=t_args,
                policy=metadata.to_dict(),
            )
            limit = self.config.safety.max_tool_output
            return ToolMessage(content=truncate_output(content, limit, source=t_name), tool_call_id=t_id, name=t_name), True

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
            self._log_run_event(
                state,
                "tool_call_start",
                run_id=state.get("run_id", ""),
                tool_name=t_name,
                tool_args=t_args,
                policy=metadata.to_dict(),
            )
            content = await self._execute_tool(t_name, t_args)

        # Post-Tool Validation Layer
        validation_error = validate_tool_result(t_name, t_args, content)
        if validation_error:
            content = f"{content}\n\n{validation_error}"
            had_error = True

        if is_error_text(content):
            had_error = True

        limit = self.config.safety.max_tool_output
        content = truncate_output(content, limit, source=t_name)
        parsed_result = parse_tool_execution_result(content)
        self._log_run_event(
            state,
            "tool_call_end",
            run_id=state.get("run_id", ""),
            tool_name=t_name,
            tool_args=t_args,
            result=parsed_result.to_event_payload(),
        )

        return ToolMessage(content=content, tool_call_id=t_id, name=t_name), had_error

    def _tool_call_is_approved(self, tool_call_id: str, approval_state: Dict[str, Any]) -> bool:
        if not self.config.enable_approvals:
            return True
        if not approval_state:
            return False
        if not approval_state.get("approved"):
            return False
        approved_ids = set(approval_state.get("tool_call_ids") or [])
        return not approved_ids or tool_call_id in approved_ids

    def _check_invariants(self, state: AgentState):
        if not self.config.debug:
            return
        steps = state.get("steps", 0)
        if steps < 0:
            logger.error(f"INVARIANT VIOLATION: steps ({steps}) < 0")

    def _get_unresolved_tool_error(self, messages: List[BaseMessage]) -> str:
        last_error_index = -1
        for index, message in enumerate(messages):
            if isinstance(message, ToolMessage) and is_tool_message_error(message):
                last_error_index = index

        if last_error_index == -1:
            return ""

        for message in messages[last_error_index + 1 :]:
            if isinstance(message, ToolMessage) and not is_tool_message_error(message):
                return ""

        block_start = last_error_index
        while block_start > 0 and isinstance(messages[block_start - 1], ToolMessage):
            block_start -= 1

        error_lines: List[str] = []
        for message in messages[block_start:]:
            if not isinstance(message, ToolMessage):
                break
            if not is_tool_message_error(message):
                continue
            label = f"tool[{message.name or 'unknown'}]"
            error_lines.append(f"{label}: {compact_text(stringify_content(message.content), 320)}")

        return "\n".join(error_lines[:3]).strip()

    def _assistant_acknowledges_unresolved_tool_error(self, text: str) -> bool:
        normalized = " ".join(text.lower().split())
        if not normalized:
            return False

        failure_markers = (
            "не удалось",
            "не смог",
            "не могу",
            "не получилось",
            "ошибка",
            "сбой",
            "не завершил",
            "не завершена",
            "не выполнен",
            "не выполнена",
            "не удалось проверить",
            "не удалось подтвердить",
            "cannot",
            "can't",
            "could not",
            "unable",
            "failed",
            "error",
            "blocker",
        )
        return any(marker in normalized for marker in failure_markers)

    def _apply_unresolved_tool_error_guard(
        self,
        status: str,
        reason: str,
        next_step: str,
        messages: List[BaseMessage],
    ) -> Tuple[str, str, str]:
        unresolved_tool_error = self._get_unresolved_tool_error(messages)
        if not unresolved_tool_error:
            return status, reason, next_step

        last_ai = self._get_last_ai_message(messages)
        if not last_ai:
            return (
                "INCOMPLETE",
                "An unresolved tool failure remains without any assistant follow-up.",
                "address the failed tool step or explain the blocker honestly",
            )

        last_text = stringify_content(last_ai.content)
        if self._assistant_acknowledges_unresolved_tool_error(last_text):
            return (
                "FINISHED",
                "The assistant explicitly reported the unresolved tool failure to the user.",
                "NONE",
            )

        return (
            "INCOMPLETE",
            "An unresolved tool failure was not addressed in the latest assistant response.",
            "fix the failed tool step or clearly report the blocker to the user",
        )

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
                content = stringify_content(message.content).strip()
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
        content = stringify_content(message.content)
        content = compact_text(content, 1200)

        if isinstance(message, ToolMessage):
            label = f"tool[{message.name or 'unknown'}]"
            return f"{label}: {content or '[empty tool output]'}"

        if isinstance(message, AIMessage):
            parts: List[str] = []
            if content:
                parts.append(f"assistant: {content}")
            for tool_call in getattr(message, "tool_calls", []) or []:
                name = tool_call.get("name", "unknown")
                args = compact_text(str(tool_call.get("args", {})), 300)
                parts.append(f"assistant_tool_call[{name}]: {args}")
            return "\n".join(parts) if parts else "assistant: [empty response]"

        if isinstance(message, HumanMessage):
            role = "system_hint" if content == constants.REFLECTION_PROMPT else "user"
            return f"{role}: {content or '[empty message]'}"

        return f"{message.type}: {content or '[empty message]'}"


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
            return compact_text(cleaned, 180)
        if status == "FINISHED":
            return "Task appears completed."
        return "Task appears incomplete."

    def _infer_critic_fallback(
        self, messages: List[BaseMessage], critic_source: str
    ) -> Tuple[str, str, str]:
        last_ai = self._get_last_ai_message(messages)
        recent_tools = self._get_recent_tool_messages(messages)

        if any(is_tool_message_error(msg) for msg in recent_tools):
            return "INCOMPLETE", "Recent tool execution reported an explicit error.", "fix the failed step"

        if last_ai and self._message_indicates_incomplete(stringify_content(last_ai.content)):
            return "INCOMPLETE", "The latest assistant message suggests work is still pending.", "continue with the remaining step"

        if critic_source == "tools" and recent_tools:
            return "FINISHED", "Recent tool results indicate the requested action completed successfully.", "NONE"

        if last_ai and not getattr(last_ai, "tool_calls", None):
            content = stringify_content(last_ai.content)
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
            prompt_path = self.config.prompt_path.absolute()
            if self.config.prompt_path.exists():
                try:
                    self._cached_base_prompt = self.config.prompt_path.read_text("utf-8")
                    logger.info("Loaded prompt from file: %s", prompt_path)
                except Exception as e:
                    logger.warning(
                        "Failed to read prompt file %s: %s. Using built-in prompt.",
                        prompt_path,
                        e,
                    )
                    self._cached_base_prompt = (
                        "You are an autonomous AI agent.\n"
                        "Reason in English, Reply in Russian.\n"
                        "Date: {{current_date}}"
                    )
            else:
                logger.info("Prompt file not found at %s. Using built-in prompt.", prompt_path)
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

    async def _invoke_llm_with_retry(
        self,
        llm,
        context: List[BaseMessage],
        state: AgentState | None = None,
        node_name: str = "",
    ) -> AIMessage:
        current_llm = llm
        max_attempts = max(1, self.config.max_retries)
        retry_delay = max(0, self.config.retry_delay)

        for attempt in range(max_attempts):
            try:
                response = await current_llm.ainvoke(context)
                invalid_calls = getattr(response, "invalid_tool_calls", None)
                if not response.content and not response.tool_calls and not invalid_calls:
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
                self._log_run_event(
                    state,
                    "llm_retry",
                    node=node_name,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    fatal=is_fatal,
                    error=str(e),
                )

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





