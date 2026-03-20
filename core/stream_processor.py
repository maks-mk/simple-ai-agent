import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from langchain_core.messages import AIMessage, AIMessageChunk, RemoveMessage, ToolMessage
from rich.console import Console, Group as RichGroup
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.spinner import Spinner
from rich.syntax import Syntax
from core.cli_utils import (
    TokenTracker,
    format_tool_display,
    format_tool_output,
    parse_thought,
    prepare_markdown_for_render,
)
from core.message_utils import is_tool_message_error, stringify_content

DIFF_REGEX = re.compile(r"```diff\r?\n(.*?)```", re.DOTALL)
logger = logging.getLogger("agent")


@dataclass
class StreamProcessResult:
    stats: Optional[str]
    interrupt: Optional[Dict[str, Any]] = None


class StreamProcessor:
    __slots__ = (
        'console',
        'tracker',
        'full_text',
        'clean_full',
        'has_thought',
        'printed_len',
        'printed_tool_ids',
        'tool_buffer',
        'tool_start_times',
        'start_time',
        'pending_interrupt',
        'active_node',
    )

    def __init__(self, console: Console):
        self.console = console
        self.tracker = TokenTracker()
        self.full_text = ""
        self.clean_full = ""
        self.has_thought = False
        self.printed_len = 0
        self.printed_tool_ids: Set[str] = set()
        self.tool_buffer: Dict[str, Dict[str, Any]] = {}
        self.tool_start_times: Dict[str, float] = {}
        self.start_time = time.time()
        self.pending_interrupt: Optional[Dict[str, Any]] = None
        self.active_node: str = "agent"

    async def process_stream(self, stream):
        """Consumes the agent event stream and updates the UI."""
        try:
            with Live(
                self._spinner_row(),
                refresh_per_second=15,
                console=self.console,
                transient=True,
            ) as live:
                async for mode, payload in stream:
                    self._handle_stream_event(mode, payload)
                    self._update_live_display(live)
                    if self.pending_interrupt is not None:
                        break
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.console.print("\n[bold red]🛑 Stopped by user[/]")
            return StreamProcessResult(stats=None)

        self._commit_printed_text()
        if self.pending_interrupt is not None:
            interrupt_payload = self.pending_interrupt
            self.pending_interrupt = None
            return StreamProcessResult(stats=None, interrupt=interrupt_payload)
        duration = time.time() - self.start_time
        return StreamProcessResult(stats=self.tracker.render(duration))

    def _handle_stream_event(self, mode: str, payload: Any) -> None:
        if mode == "updates":
            self._handle_updates(payload)
        elif mode == "messages":
            self._handle_messages(payload)

    def _append_text(self, chunk: str) -> None:
        if not chunk:
            return

        self.full_text += chunk
        if '<th' in self.full_text:
            _, self.clean_full, self.has_thought = parse_thought(self.full_text)
        else:
            self.clean_full = self.full_text
            self.has_thought = False

    def _handle_updates(self, payload: Dict) -> None:
        if "__interrupt__" in payload:
            self.active_node = "approval"
            interrupt_entries = payload.get("__interrupt__") or []
            if interrupt_entries:
                interrupt_value = getattr(interrupt_entries[0], "value", interrupt_entries[0])
                if isinstance(interrupt_value, dict):
                    self.pending_interrupt = interrupt_value
                else:
                    self.pending_interrupt = {"value": interrupt_value}
            return
        self.tracker.update_from_node_update(payload)
        self._commit_printed_text()

        summarize_payload = payload.get("summarize") or {}
        if summarize_payload:
            self._handle_summarize_update(summarize_payload)

        agent_payload = payload.get("agent") or {}
        messages = agent_payload.get("messages", [])
        if not isinstance(messages, list):
            messages = [messages]

        last_msg = messages[-1] if messages else None
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            for tool_call in last_msg.tool_calls:
                self._remember_tool_call(tool_call)
                self._render_tool_call(tool_call)

    def _handle_summarize_update(self, payload: Dict[str, Any]) -> None:
        summary_text = payload.get("summary")
        removed_messages = payload.get("messages") or []
        remove_count = sum(1 for item in removed_messages if isinstance(item, RemoveMessage))
        if not summary_text:
            return
        rendered_count = remove_count if remove_count > 0 else len(removed_messages)
        self.console.print(
            Padding(
                f"[approval.summary]note[/] [dim]Context compressed automatically ({rendered_count} message(s) summarized).[/]",
                (0, 0, 0, 4),
            )
        )

    def _handle_messages(self, payload: tuple) -> None:
        msg, metadata = payload
        node = metadata.get("langgraph_node")
        self.tracker.update_from_message(msg)

        if node:
            self.active_node = node

        if node == "agent" and isinstance(msg, (AIMessage, AIMessageChunk)):
            self._handle_agent_message(msg)
        elif node == "tools" and isinstance(msg, ToolMessage):
            self._handle_tool_result(msg)

    def _handle_agent_message(self, msg: AIMessage | AIMessageChunk) -> None:
        if msg.tool_calls:
            self._commit_printed_text()

        chunk = self._extract_text_content(msg.content)
        if chunk:
            self._append_text(chunk)
            self._try_commit()

    def _extract_text_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(x.get("text", "") for x in content if isinstance(x, dict))
        return ""

    def _remember_tool_call(self, tool_call: Dict[str, Any]) -> None:
        tool_id = tool_call.get("id")
        if not tool_id:
            return
        self.tool_buffer[tool_id] = {
            "name": tool_call.get("name", "unknown_tool"),
            "args": tool_call.get("args", {}),
        }

    def _render_tool_call(self, tool_call: Dict[str, Any]) -> None:
        tool_id = tool_call["id"]
        if tool_id in self.printed_tool_ids:
            return

        # Record start time for duration tracking
        self.tool_start_times[tool_id] = time.time()

        display_str = format_tool_display(tool_call["name"], tool_call["args"])
        if "(" in display_str:
            name_part, args_part = display_str.split("(", 1)
            args_part = "(" + args_part
            display_styled = f"[tool.name]{name_part}[/][tool.args]{args_part}[/]"
        else:
            display_styled = f"[tool.name]{display_str}[/]"

        self.console.print(Padding(f"[tool.badge]tool[/] [tool.badge]▶[/] {display_styled}", (0, 0, 0, 4)))
        self.printed_tool_ids.add(tool_id)
        self.active_node = "tools"

    def _handle_tool_result(self, msg: ToolMessage) -> None:
        tool_id = msg.tool_call_id
        if tool_id in self.tool_buffer and tool_id not in self.printed_tool_ids:
            self._render_tool_call({"id": tool_id, **self.tool_buffer[tool_id]})

        content_str = stringify_content(msg.content)
        is_error = is_tool_message_error(msg)
        summary = format_tool_output(msg.name, content_str, is_error)
        style = "tool.error" if is_error else "tool.result"
        icon = "[tool.error]✖ [/]" if is_error else "[tool.result]✔ [/]"

        # Compute and show tool execution duration
        elapsed_str = ""
        start_t = self.tool_start_times.pop(tool_id, None)
        if start_t is not None:
            elapsed = time.time() - start_t
            elapsed_str = f" [tool.timing]{elapsed:.1f}s[/]"

        self.console.print(Padding(f"[tool.badge]tool[/] {icon}[{style}]{summary}[/]{elapsed_str}", (0, 0, 0, 4)))

        if not is_error:
            self._render_diff_preview(content_str)

        self.active_node = "agent"

    def _render_diff_preview(self, content: str) -> None:
        diff_match = DIFF_REGEX.search(content)
        if not diff_match:
            return

        diff_code = diff_match.group(1).strip()
        syntax = Syntax(diff_code, "diff", theme="monokai", line_numbers=True, word_wrap=True)
        self.console.print(Padding(syntax, (0, 0, 0, 8)))

    def _commit_printed_text(self, end_index: Optional[int] = None) -> None:
        limit = end_index if end_index is not None else len(self.clean_full)
        if limit <= self.printed_len:
            return

        new_text = prepare_markdown_for_render(self.clean_full[self.printed_len:limit])
        self.console.print(Padding(Markdown(new_text, code_theme="dracula", hyperlinks=False), (0, 0, 0, 4)))
        self.printed_len = limit

    def _try_commit(self) -> None:
        if len(self.clean_full) <= self.printed_len:
            return

        pending = self.clean_full[self.printed_len:]
        last_newline = pending.rfind('\n\n')
        if last_newline == -1:
            return

        commit_len = last_newline + 2
        absolute_end = self.printed_len + commit_len
        if self.clean_full[:absolute_end].count("```") % 2 != 0:
            return

        self._commit_printed_text(end_index=absolute_end)

    def _current_pending_markdown(self) -> str:
        pending = self.clean_full[self.printed_len:]
        if not pending.strip():
            return ""

        render_text = pending
        if self.clean_full.count("```") % 2 != 0:
            render_text += "\n```"
        return prepare_markdown_for_render(render_text)

    def _elapsed(self) -> str:
        """Returns formatted elapsed time since stream start."""
        return f"{time.time() - self.start_time:.0f}s"

    def _status_label(self) -> str:
        node_labels = {
            "agent": "Thinking",
            "critic": "Reviewing",
            "tools": "Running tools",
            "summarize": "Compressing context",
            "approval": "Waiting for approval",
        }
        return node_labels.get(self.active_node, "Thinking")

    def _spinner_row(self):
        base = self._status_label()
        if self.has_thought:
            label = f"[agent.thought]{base}[/] [tool.timing]{self._elapsed()}[/]"
        else:
            label = f"{base} [tool.timing]{self._elapsed()}[/]"
        return Spinner("dots", text=label, style="status.spinner")

    def _update_live_display(self, live: Live) -> None:
        try:
            renderables = []

            # Always show any pending partial text ABOVE the spinner
            pending_markdown = self._current_pending_markdown()
            if pending_markdown:
                renderables.append(
                    Padding(
                        Markdown(pending_markdown, code_theme="dracula", hyperlinks=False),
                        (0, 0, 0, 4),
                    )
                )

            # Spinner is always visible so the user sees the agent is still working.
            renderables.append(self._spinner_row())

            live.update(RichGroup(*renderables))
        except Exception as e:
            logger.debug("Live display update failed: %s", e)



