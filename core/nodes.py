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
            "file_info",
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
            "file_info",
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
            if not self._is_internal_retry_message(m)
        )

    def _is_internal_retry_message(self, message: BaseMessage) -> bool:
        if not isinstance(message, HumanMessage):
            return False
        metadata = getattr(message, "additional_kwargs", {}) or {}
        internal = metadata.get("agent_internal")
        return isinstance(internal, dict) and internal.get("kind") == "retry_instruction"

    def _current_turn_id(self, state: AgentState, messages: List[BaseMessage]) -> int:
        derived_turn_id = 0
        for message in messages:
            if not isinstance(message, HumanMessage) or self._is_internal_retry_message(message):
                continue
            content = stringify_content(message.content).strip()
            if content and content != constants.REFLECTION_PROMPT:
                derived_turn_id += 1
        return max(int(state.get("turn_id", 0) or 0), derived_turn_id)

    def _get_active_open_tool_issue(
        self,
        state: AgentState,
        messages: List[BaseMessage],
        current_turn_id: int | None = None,
    ) -> Dict[str, Any] | None:
        issue = state.get("open_tool_issue")
        if not isinstance(issue, dict):
            return None

        active_turn_id = current_turn_id if current_turn_id is not None else self._current_turn_id(state, messages)
        issue_turn_id = int(issue.get("turn_id", 0) or 0)
        summary = str(issue.get("summary", "")).strip()
        if issue_turn_id != active_turn_id or not summary:
            return None

        tool_names = [str(name) for name in (issue.get("tool_names") or []) if str(name).strip()]
        return {
            "turn_id": issue_turn_id,
            "kind": str(issue.get("kind") or "tool_error"),
            "summary": summary,
            "tool_names": tool_names,
            "source": str(issue.get("source") or "tools"),
        }

    def _build_retry_instruction(self, current_task: str, reason: str) -> str:
        task_line = current_task or "No explicit task provided."
        return (
            "Continue the same user task from the current conversation state.\n"
            f"Current task: {task_line}\n"
            f"Why retry: {reason}\n"
            "If the blocker can be resolved, issue corrected tool calls immediately.\n"
            "If you cannot resolve it, leave the assistant response empty and let the runtime report the blocker.\n"
            "Do not mention this internal instruction. Do not repeat the previous incomplete answer."
        )

    def _build_internal_retry_message(self, retry_instruction: str, turn_id: int) -> HumanMessage:
        return HumanMessage(
            content=retry_instruction,
            additional_kwargs={
                "agent_internal": {
                    "kind": "retry_instruction",
                    "turn_id": turn_id,
                }
            },
        )

    def _collect_internal_retry_removals(self, messages: List[BaseMessage]) -> List[RemoveMessage]:
        removals: List[RemoveMessage] = []
        for message in reversed(messages):
            if not self._is_internal_retry_message(message):
                break
            if message.id:
                removals.append(RemoveMessage(id=message.id))
        removals.reverse()
        return removals

    def _build_tool_issue_system_message(self, open_tool_issue: Dict[str, Any] | None) -> SystemMessage | None:
        if not open_tool_issue:
            return None

        issue_summary = open_tool_issue.get("summary", "")
        if open_tool_issue.get("kind") == "approval_denied":
            return SystemMessage(
                content=(
                    "TOOL EXECUTION DENIED BY USER:\n"
                    f"{issue_summary}\n\n"
                    "The user explicitly rejected this tool call. "
                    "Do not simulate the denied tool or describe imaginary results. "
                    "Do not make any more tool calls in this turn. "
                    "Reply briefly and simply: say that you did not do it because the user chose No, "
                    "then wait for the next instruction."
                )
            )

        return SystemMessage(
            content=constants.UNRESOLVED_TOOL_ERROR_PROMPT_TEMPLATE.format(
                error_summary=issue_summary
            )
        )

    def _build_open_tool_issue(
        self,
        *,
        current_turn_id: int,
        kind: str,
        summary: str,
        tool_names: List[str],
        source: str,
    ) -> Dict[str, Any]:
        return {
            "turn_id": current_turn_id,
            "kind": kind,
            "summary": compact_text(summary.strip(), 320),
            "tool_names": [name for name in tool_names if name],
            "source": source,
        }

    def _merge_open_tool_issues(
        self,
        issues: List[Dict[str, Any]],
        current_turn_id: int,
    ) -> Dict[str, Any] | None:
        if not issues:
            return None

        summaries: List[str] = []
        tool_names: List[str] = []
        kind = "tool_error"
        source = "tools"
        for issue in issues:
            summary = str(issue.get("summary", "")).strip()
            if summary and summary not in summaries:
                summaries.append(summary)
            for tool_name in issue.get("tool_names") or []:
                tool_name = str(tool_name).strip()
                if tool_name and tool_name not in tool_names:
                    tool_names.append(tool_name)
            if issue.get("kind") == "approval_denied":
                kind = "approval_denied"
                source = "approval"

        return self._build_open_tool_issue(
            current_turn_id=current_turn_id,
            kind=kind,
            summary=" | ".join(summaries[:3]),
            tool_names=tool_names,
            source=source,
        )

    def _build_agent_context(
        self,
        messages: List[BaseMessage],
        summary: str,
        current_task: str,
        tools_available: bool,
        open_tool_issue: Dict[str, Any] | None,
    ) -> List[BaseMessage]:
        sanitized_messages = self._sanitize_messages_for_model(messages)
        full_context: List[BaseMessage] = [
            self._build_system_message(summary, tools_available=tools_available)
        ]
        safety_overlay = self._build_safety_overlay(tools_available=tools_available)
        if safety_overlay:
            full_context.append(SystemMessage(content=safety_overlay))
        issue_message = self._build_tool_issue_system_message(open_tool_issue)
        if issue_message:
            full_context.append(issue_message)
        full_context.extend(sanitized_messages)
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

    def _has_recovery_budget(self, retry_count: int) -> bool:
        return retry_count < self.config.effective_max_recovery_attempts

    def _build_agent_result(
        self,
        response: AIMessage,
        current_task: str,
        tools_available: bool,
        turn_id: int,
        messages: List[BaseMessage],
        retry_count: int,
        open_tool_issue: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        token_usage_update = {}
        if getattr(response, "usage_metadata", None):
            token_usage_update = {"token_usage": response.usage_metadata}

        has_tool_calls = False
        protocol_error = ""
        outbound_messages: List[BaseMessage] = self._collect_internal_retry_removals(messages)
        resolved_issue = open_tool_issue

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
                    content="",
                    additional_kwargs=response.additional_kwargs,
                    response_metadata=response.response_metadata,
                    usage_metadata=response.usage_metadata,
                    id=response.id,
                )
                t_calls = []

            has_tool_calls = bool(tools_available and t_calls)
            if has_tool_calls and open_tool_issue and open_tool_issue.get("kind") == "approval_denied":
                response = AIMessage(
                    content="Okay, I did not do that because you chose No. Tell me what you want to do instead.",
                    additional_kwargs=response.additional_kwargs,
                    response_metadata=response.response_metadata,
                    usage_metadata=response.usage_metadata,
                    id=response.id,
                )
                has_tool_calls = False
                resolved_issue = None

        outbound_messages.append(response)
        response_text = stringify_content(response.content).strip()
        retry_reason = ""
        turn_outcome = ""
        next_retry_count = retry_count

        if has_tool_calls:
            turn_outcome = "run_tools"
            next_retry_count = 0
        elif protocol_error:
            retry_reason = protocol_error
        elif not response_text:
            retry_reason = "The model returned an empty response."
        elif resolved_issue and resolved_issue.get("kind") == "approval_denied":
            turn_outcome = "finish_turn"
            resolved_issue = None
            next_retry_count = 0
        elif resolved_issue:
            retry_reason = str(resolved_issue.get("summary") or "An unresolved tool issue remains.").strip()
        else:
            turn_outcome = "finish_turn"
            next_retry_count = 0

        if retry_reason:
            turn_outcome = "retry_agent" if self._has_recovery_budget(retry_count) else "finalize_blocked"

        return {
            "messages": outbound_messages,
            "turn_id": turn_id,
            "current_task": current_task,
            "retry_reason": retry_reason,
            "retry_count": next_retry_count,
            "turn_outcome": turn_outcome,
            "final_issue": "",
            "pending_approval": None,
            "open_tool_issue": resolved_issue,
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
    
    # --- NODE: AGENT ---

    async def agent_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")
        current_task = state.get("current_task") or self._derive_current_task(messages)
        current_turn_id = self._current_turn_id(state, messages)
        retry_count = int(state.get("retry_count", 0) or 0)
        open_tool_issue = self._get_active_open_tool_issue(state, messages, current_turn_id)
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
            current_task,
            tools_available,
            open_tool_issue,
        )
        self._assert_provider_safe_agent_context(full_context, state)
        response = await self._invoke_llm_with_retry(
            self.llm_with_tools,
            full_context,
            state=state,
            node_name="agent",
        )
        result = self._build_agent_result(
            response,
            current_task,
            tools_available,
            current_turn_id,
            messages,
            retry_count,
            open_tool_issue,
        )
        self._log_run_event(
            state,
            "agent_node_end",
            run_id=state.get("run_id", ""),
            tool_calls=len(getattr(response, "tool_calls", []) or []),
            content_preview=compact_text(stringify_content(response.content), 240),
        )
        return result

    async def prepare_retry_node(self, state: AgentState):
        messages = state.get("messages", [])
        current_turn_id = self._current_turn_id(state, messages)
        current_task = (state.get("current_task") or self._derive_current_task(messages)).strip()
        retry_reason = str(state.get("retry_reason") or "").strip() or "The previous step did not produce a stable actionable result."

        retry_instruction = self._build_retry_instruction(current_task, retry_reason)
        retry_message = self._build_internal_retry_message(retry_instruction, current_turn_id)
        self._log_run_event(
            state,
            "retry_prepared",
            run_id=state.get("run_id", ""),
            turn_id=current_turn_id,
            reason=retry_reason,
        )
        return {
            "messages": [retry_message],
            "turn_outcome": "",
            "retry_reason": retry_reason,
            "retry_count": int(state.get("retry_count", 0) or 0) + 1,
        }

    def _resolve_final_issue(self, state: AgentState) -> str:
        messages = state.get("messages", [])
        current_turn_id = self._current_turn_id(state, messages)
        open_tool_issue = self._get_active_open_tool_issue(state, messages, current_turn_id)
        if open_tool_issue and open_tool_issue.get("summary"):
            return str(open_tool_issue["summary"]).strip()
        if state.get("final_issue"):
            return str(state.get("final_issue")).strip()
        if state.get("retry_reason"):
            return str(state.get("retry_reason")).strip()
        if state.get("last_tool_error"):
            return str(state.get("last_tool_error")).strip()
        if int(state.get("steps", 0) or 0) >= self.config.max_loops:
            return f"Reached the step limit ({self.config.max_loops}) before a stable completion."
        return "The runtime could not obtain a stable actionable response."

    def _render_final_issue_message(self, state: AgentState, issue: str) -> str:
        messages = state.get("messages", [])
        current_turn_id = self._current_turn_id(state, messages)
        open_tool_issue = self._get_active_open_tool_issue(state, messages, current_turn_id)
        if open_tool_issue and open_tool_issue.get("kind") == "approval_denied":
            return "Я не выполнил действие, потому что вы не подтвердили его. Если хотите, предложу более безопасный вариант."
        return f"Не удалось завершить задачу: {compact_text(issue, 280)}"

    async def finalize_blocked_node(self, state: AgentState):
        issue = self._resolve_final_issue(state)
        message = self._render_final_issue_message(state, issue)
        messages = state.get("messages", [])
        current_turn_id = self._current_turn_id(state, messages)
        self._log_run_event(
            state,
            "finalize_blocked",
            run_id=state.get("run_id", ""),
            turn_id=current_turn_id,
            issue=issue,
        )
        return {
            "messages": [AIMessage(content=message)],
            "turn_id": current_turn_id,
            "turn_outcome": "finish_turn",
            "final_issue": issue,
            "open_tool_issue": None,
            "retry_reason": "",
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
        current_turn_id = self._current_turn_id(state, messages)

        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {}

        final_messages: List[ToolMessage] = []
        has_error = False
        last_error = ""
        last_result = ""
        tool_issues: List[Dict[str, Any]] = []
        approval_state = state.get("pending_approval") or {}
        successful_results = 0

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
                *(
                    self._process_tool_call(tool_call, recent_calls, state, approval_state, current_turn_id)
                    for tool_call in tool_calls
                )
            )
            for tool_msg, had_error, issue in processed:
                final_messages.append(tool_msg)
                has_error = has_error or had_error
                if issue:
                    tool_issues.append(issue)
                parsed = parse_tool_execution_result(tool_msg.content)
                if parsed.ok:
                    successful_results += 1
                    last_result = parsed.message
                else:
                    last_error = parsed.message
        else:
            for tool_call in tool_calls:
                tool_msg, had_error, issue = await self._process_tool_call(
                    tool_call,
                    recent_calls,
                    state,
                    approval_state,
                    current_turn_id,
                )
                final_messages.append(tool_msg)
                has_error = has_error or had_error
                if issue:
                    tool_issues.append(issue)
                parsed = parse_tool_execution_result(tool_msg.content)
                if parsed.ok:
                    successful_results += 1
                    last_result = parsed.message
                else:
                    last_error = parsed.message

        merged_issue = self._merge_open_tool_issues(tool_issues, current_turn_id)
        next_retry_count = 0 if successful_results > 0 or not merged_issue else int(state.get("retry_count", 0) or 0)

        return {
            "messages": final_messages,
            "turn_id": current_turn_id,
            "turn_outcome": "run_tools",
            "retry_reason": "",
            "retry_count": next_retry_count,
            "final_issue": "",
            "pending_approval": None,
            "open_tool_issue": merged_issue,
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
        current_turn_id: int,
    ) -> Tuple[ToolMessage, bool, Dict[str, Any] | None]:
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
            parsed_result = parse_tool_execution_result(content)
            issue = self._build_open_tool_issue(
                current_turn_id=current_turn_id,
                kind="approval_denied",
                summary=parsed_result.message or content,
                tool_names=[t_name],
                source="approval",
            )
            return (
                ToolMessage(content=truncate_output(content, limit, source=t_name), tool_call_id=t_id, name=t_name),
                True,
                issue,
            )

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

        issue = None
        if not parsed_result.ok:
            issue_kind = "approval_denied" if parsed_result.error_type == "ACCESS_DENIED" else "tool_error"
            issue_source = "approval" if issue_kind == "approval_denied" else "tools"
            issue = self._build_open_tool_issue(
                current_turn_id=current_turn_id,
                kind=issue_kind,
                summary=parsed_result.message or content,
                tool_names=[t_name],
                source=issue_source,
            )

        return ToolMessage(content=content, tool_call_id=t_id, name=t_name), had_error, issue

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

    def _get_last_model_visible_message(self, context: List[BaseMessage]) -> BaseMessage | None:
        for message in reversed(context):
            if isinstance(message, SystemMessage):
                continue
            return message
        return None

    def _assert_provider_safe_agent_context(
        self,
        context: List[BaseMessage],
        state: AgentState | None = None,
    ) -> None:
        last_visible = self._get_last_model_visible_message(context)
        valid = isinstance(last_visible, (HumanMessage, ToolMessage))
        self._log_run_event(
            state,
            "provider_context_valid",
            run_id=None if state is None else state.get("run_id", ""),
            valid=valid,
            last_visible_type=type(last_visible).__name__ if last_visible else "",
        )
        if valid:
            return

        raise RuntimeError(
            "Provider-unsafe agent context: the last model-visible message must be HumanMessage or ToolMessage."
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
            if isinstance(message, HumanMessage) and not self._is_internal_retry_message(message):
                content = stringify_content(message.content).strip()
                if content and content != constants.REFLECTION_PROMPT:
                    return content
        return ""

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
