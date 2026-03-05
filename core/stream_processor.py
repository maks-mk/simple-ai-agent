import time
import asyncio
import re
from typing import Dict, Any, Optional, Set
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.padding import Padding
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from core.cli_utils import (
    TokenTracker,
    clean_markdown_text,
    parse_thought,
    format_tool_output,
    format_tool_display
)

# Компилируем Regex один раз при старте модуля (ускорение поиска diff-ов)
DIFF_REGEX = re.compile(r"```diff\r?\n(.*?)```", re.DOTALL)

class StreamProcessor:
    __slots__ = ('console', 'tracker', 'full_text', 'clean_full', 'has_thought',
                 'printed_len', 'printed_tool_ids', 'tool_buffer', 'status_text', 'start_time')

    def __init__(self, console: Console):
        self.console = console
        self.tracker = TokenTracker()
        
        # State
        self.full_text = ""          
        self.clean_full = ""          # Кэшированный чистый текст без мыслей
        self.has_thought = False      # Флаг состояния размышлений
        self.printed_len = 0         
        
        self.printed_tool_ids: Set[str] = set()
        self.tool_buffer: Dict[str, Dict[str, Any]] = {}        
        self.status_text = "Thinking..."
        self.start_time = time.time()

    async def process_stream(self, stream):
        """
        Consumes the agent event stream and updates the UI.
        """
        try:
            with Live(Spinner("dots", text=self.status_text, style="cyan"), 
                      refresh_per_second=15, 
                      console=self.console, 
                      transient=True) as live:
                
                async for mode, payload in stream:
                    if mode == "updates":
                        self._handle_updates(payload)
                    elif mode == "messages":
                        self._handle_messages(payload)
                            
                    self._update_live_display(live)

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.console.print("\n[bold red]🛑 Stopped by user[/]")
            return 
        
        # Manually print any remaining text that wasn't streamed
        self._commit_printed_text(end_index=None)
        
        duration = time.time() - self.start_time
        return self.tracker.render(duration)

    def _append_text(self, chunk: str):
        """Adds text and updates the parsed caches efficiently."""
        if chunk:
            self.full_text += chunk
            # Быстрая проверка: если нет тегов мыслей, пропускаем дорогой parse_thought
            if '<th' in self.full_text:
                _, self.clean_full, self.has_thought = parse_thought(self.full_text)
            else:
                self.clean_full = self.full_text
                self.has_thought = False

    def _handle_updates(self, payload: Dict):
        self.tracker.update_from_node_update(payload)
        self._commit_printed_text()

        if "agent" in payload:
            messages = payload["agent"].get("messages", [])
            if not isinstance(messages, list): 
                messages = [messages]
            
            last_msg = messages[-1] if messages else None
            
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    self._handle_tool_call(tc)

    def _handle_messages(self, payload: tuple):
        msg, metadata = payload
        node = metadata.get("langgraph_node")
        self.tracker.update_from_message(msg)
            
        if node == "agent" and isinstance(msg, (AIMessage, AIMessageChunk)):
            if msg.tool_calls:
                self._commit_printed_text()
            
            if msg.content:
                chunk = self._extract_text_content(msg.content)
                self._append_text(chunk)
                self._try_commit()

        elif node == "tools" and isinstance(msg, ToolMessage):
            self._handle_tool_result(msg)

    def _commit_printed_text(self, end_index: Optional[int] = None):
        """Transfers text from dynamic Live to static console log."""
        limit = end_index if end_index is not None else len(self.clean_full)
        if limit > self.printed_len:
            new_text = self.clean_full[self.printed_len:limit]
            
            # Use standard rich Markdown renderer
            formatted_content = Markdown(new_text, code_theme="dracula", hyperlinks=False)
            
            # self.console.print безопасно прерывает Live, печатает и возвращает Live обратно
            self.console.print(Padding(formatted_content, (0, 0, 0, 2)))
            self.printed_len = limit

    def _try_commit(self):
        """Attempts to commit text if safe (paragraph boundaries)."""
        if len(self.clean_full) <= self.printed_len: 
            return
        
        pending = self.clean_full[self.printed_len:]
        last_newline = pending.rfind('\n\n')
        
        if last_newline != -1:
            commit_len = last_newline + 2 
            absolute_end = self.printed_len + commit_len
            
            # Глобальная проверка: открыты ли сейчас блоки кода
            # Если нечетное кол-во ``` от начала всего текста - мы внутри кода!
            if self.clean_full[:absolute_end].count("```") % 2 != 0:
                return

            self._commit_printed_text(end_index=absolute_end)

    def _extract_text_content(self, content: Any) -> str:
        if isinstance(content, str): 
            return content
        if isinstance(content, list):
            return "".join(x.get("text", "") for x in content if isinstance(x, dict))
        return ""

    def _handle_tool_call(self, tc: Dict[str, Any]):
        t_id = tc["id"]
        self.tool_buffer[t_id] = {"name": tc["name"], "args": tc["args"]}
        
        if t_id not in self.printed_tool_ids:
            display_str = format_tool_display(tc["name"], tc["args"])
            
            if "(" in display_str:
                name_part, args_part = display_str.split("(", 1)
                args_part = "(" + args_part 
                display_styled = f"[tool.name]{name_part}[/][tool.args]{args_part}[/]"
            else:
                display_styled = f"[tool.name]{display_str}[/]"
            
            self.console.print(Padding(f"›  {display_styled}", (0, 0, 0, 2)))
            
            self.printed_tool_ids.add(t_id)
            self.status_text = f"Running {tc['name']}..."

    def _handle_tool_result(self, msg: ToolMessage):
        t_id = msg.tool_call_id
        content_str = str(msg.content)
        
        is_error = (
            getattr(msg, "status", "") == "error" 
            or content_str.startswith(("Error", "Ошибка", "ERROR["))
            or "ERROR[" in content_str
        )

        if t_id in self.tool_buffer and t_id not in self.printed_tool_ids:
            info = self.tool_buffer[t_id]
            display_str = format_tool_display(info["name"], info["args"])
            self.console.print(Padding(f"›  [tool.name]{display_str}[/]", (0, 0, 0, 2)))
            self.printed_tool_ids.add(t_id)
            
        summary = format_tool_output(msg.name, content_str, is_error)
        
        if is_error:
            icon = "[tool.error]✖ [/]" 
            style = "tool.error"
        else:
            icon = "[tool.result]✔ [/]"
            style = "tool.result"
            
        self.console.print(Padding(f"  {icon} [{style}]{summary}[/]", (0, 0, 0, 4)))
        
        if not is_error:
            # Используем пре-скомпилированный regex
            diff_match = DIFF_REGEX.search(content_str)
            if diff_match:
                diff_code = diff_match.group(1).strip()
                syntax = Syntax(diff_code, "diff", theme="monokai", line_numbers=True, word_wrap=True)
                self.console.print(Padding(syntax, (0, 0, 0, 6)))

        self.status_text = "Thinking..."

    def _update_live_display(self, live: Live):
        try:
            # 1. Сброс зависшего статуса после того, как мысль завершилась
            if self.has_thought:
                self.status_text = "[agent.thought]Thinking...[/]"
            elif self.status_text == "[agent.thought]Thinking...[/]":
                self.status_text = "Thinking..."
            
            pending = self.clean_full[self.printed_len:]
            
            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(justify="left", ratio=1)
            
            # 2. Прячем спиннер, если начал генерироваться текст
            if pending.strip():
                 render_text = pending
                 if self.clean_full.count("```") % 2 != 0:
                     render_text += "\n```"
                 # Добавляем только Markdown, без спиннера сверху!
                 grid.add_row(Padding(Markdown(clean_markdown_text(render_text), code_theme="ansi_dark", hyperlinks=False), (0, 0, 0, 2)))
            else:
                 # Спиннер рисуется только пока текста нет
                 spinner = Spinner("dots", text=self.status_text, style="status.spinner")
                 grid.add_row(spinner)
                 
            live.update(grid)
        except Exception:
            pass

