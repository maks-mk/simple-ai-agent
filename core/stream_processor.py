import asyncio
import re
import time
from typing import Any, Dict, Optional, Set

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table

from core.cli_utils import (
    TokenTracker,
    format_tool_display,
    format_tool_output,
    parse_thought,
    prepare_markdown_for_render,
)
from core.message_utils import is_tool_message_error, stringify_content

DIFF_REGEX = re.compile(r"```diff\r?\n(.*?)```", re.DOTALL)


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
        'status_text',
        'start_time',
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
        self.status_text = "Thinking..."
        self.start_time = time.time()

    async def process_stream(self, stream):
        """Consumes the agent event stream and updates the UI."""
        try:
            with Live(
                Spinner("dots", text=self.status_text, style="cyan"),
                refresh_per_second=15,
                console=self.console,
                transient=True,
            ) as live:
                async for mode, payload in stream:
                    self._handle_stream_event(mode, payload)
                    self._update_live_display(live)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.console.print("\n[bold red]🛑 Stopped by user[/]")
            return

        self._commit_printed_text()
        duration = time.time() - self.start_time
        return self.tracker.render(duration)

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
        self.tracker.update_from_node_update(payload)
        self._commit_printed_text()

        agent_payload = payload.get("agent") or {}
        messages = agent_payload.get("messages", [])
        if not isinstance(messages, list):
            messages = [messages]

        last_msg = messages[-1] if messages else None
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            for tool_call in last_msg.tool_calls:
                self._remember_tool_call(tool_call)
                self._render_tool_call(tool_call)

    def _handle_messages(self, payload: tuple) -> None:
        msg, metadata = payload
        node = metadata.get("langgraph_node")
        self.tracker.update_from_message(msg)

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
        self.tool_buffer[tool_call["id"]] = {
            "name": tool_call["name"],
            "args": tool_call["args"],
        }

    def _render_tool_call(self, tool_call: Dict[str, Any]) -> None:
        tool_id = tool_call["id"]
        if tool_id in self.printed_tool_ids:
            return

        display_str = format_tool_display(tool_call["name"], tool_call["args"])
        if "(" in display_str:
            name_part, args_part = display_str.split("(", 1)
            args_part = "(" + args_part
            display_styled = f"[tool.name]{name_part}[/][tool.args]{args_part}[/]"
        else:
            display_styled = f"[tool.name]{display_str}[/]"

        self.console.print(Padding(f"›  {display_styled}", (0, 0, 0, 2)))
        self.printed_tool_ids.add(tool_id)
        self.status_text = f"Running {tool_call['name']}..."

    def _handle_tool_result(self, msg: ToolMessage) -> None:
        tool_id = msg.tool_call_id
        if tool_id in self.tool_buffer and tool_id not in self.printed_tool_ids:
            self._render_tool_call({"id": tool_id, **self.tool_buffer[tool_id]})

        content_str = stringify_content(msg.content)
        is_error = is_tool_message_error(msg)
        summary = format_tool_output(msg.name, content_str, is_error)
        style = "tool.error" if is_error else "tool.result"
        icon = "[tool.error]✖ [/]" if is_error else "[tool.result]✔ [/]"
        self.console.print(Padding(f"  {icon}[{style}]{summary}[/]", (0, 0, 0, 4)))

        if not is_error:
            self._render_diff_preview(content_str)

        self.status_text = "Thinking..."

    def _render_diff_preview(self, content: str) -> None:
        diff_match = DIFF_REGEX.search(content)
        if not diff_match:
            return

        diff_code = diff_match.group(1).strip()
        syntax = Syntax(diff_code, "diff", theme="monokai", line_numbers=True, word_wrap=True)
        self.console.print(Padding(syntax, (0, 0, 0, 6)))

    def _commit_printed_text(self, end_index: Optional[int] = None) -> None:
        limit = end_index if end_index is not None else len(self.clean_full)
        if limit <= self.printed_len:
            return

        new_text = prepare_markdown_for_render(self.clean_full[self.printed_len:limit])
        self.console.print(Padding(Markdown(new_text, code_theme="dracula", hyperlinks=False), (0, 0, 0, 2)))
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

    def _spinner_row(self):
        return Spinner("dots", text=self.status_text, style="status.spinner")

    def _update_live_display(self, live: Live) -> None:
        try:
            if self.has_thought:
                self.status_text = "[agent.thought]Thinking...[/]"
            elif self.status_text == "[agent.thought]Thinking...[/]":
                self.status_text = "Thinking..."

            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(justify="left", ratio=1)

            pending_markdown = self._current_pending_markdown()
            if pending_markdown:
                grid.add_row(Padding(Markdown(pending_markdown, code_theme="dracula", hyperlinks=False), (0, 0, 0, 2)))
            else:
                grid.add_row(self._spinner_row())

            live.update(grid)
        except Exception:
            pass







