import time
import asyncio
import re
from typing import Dict, Any, Optional, Set
from rich.console import Console, Group
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
    format_tool_output
)

class StreamProcessor:
    def __init__(self, console: Console):
        self.console = console
        self.tracker = TokenTracker()
        self.full_text = ""          
        self.printed_len = 0         
        self.printed_tool_ids: Set[str] = set()
        self.tool_buffer: Dict[str, Dict[str, Any]] = {}        
        self.status_text = "Thinking..."
        self.start_time = time.time()

    async def run(self, agent_app, user_input: str, thread_id: str, max_loops: int):
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_loops * 4}
        initial_state = {
            "messages": [("user", user_input)], 
            "steps": 0,
            "token_used": 0 
        }
        
        try:
            with Live(Spinner("dots", text=self.status_text, style="cyan"), 
                      refresh_per_second=12, 
                      console=self.console, 
                      transient=True) as live:
                
                async for mode, payload in agent_app.astream(
                    initial_state,
                    config=config,
                    stream_mode=["messages", "updates"]
                ):
                    await asyncio.sleep(0.01)

                    if mode == "updates":
                        self._handle_updates(payload, live)
                    elif mode == "messages":
                        self._handle_messages(payload, live)
                            
                    self._update_live_display(live)

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.console.print("\n[bold red]🛑 Stopped by user[/]")
            return 
        
        # Manually print any remaining text that wasn't streamed
        self._commit_printed_text(None)
        
        duration = time.time() - self.start_time
        return self.tracker.render(duration)

    def _handle_updates(self, payload: Dict, live: Live):
        self.tracker.update_from_node_update(payload)
        self._commit_printed_text(live)

        if "agent" in payload:
            messages = payload["agent"].get("messages", [])
            if not isinstance(messages, list): messages = [messages]
            last_msg = messages[-1] if messages else None
            
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    self.tool_buffer[tc["id"]] = {"name": tc["name"], "args": tc["args"]}
                    self._handle_tool_call(tc, live)

    def _handle_messages(self, payload: tuple, live: Live):
        msg, metadata = payload
        node = metadata.get("langgraph_node")
        self.tracker.update_from_message(msg)
            
        if node == "agent" and isinstance(msg, (AIMessage, AIMessageChunk)):
            if msg.tool_calls:
                self._commit_printed_text(live)
            
            if msg.content:
                chunk = self._extract_text_content(msg.content)
                self.full_text += chunk
                self._try_commit(live)

        elif node == "tools" and isinstance(msg, ToolMessage):
            self._handle_tool_result(msg, live)

    def _commit_printed_text(self, live: Optional[Live], end_index: int = None):
        """Переносит текст из динамического Live в статический лог консоли."""
        _, clean_full, _ = parse_thought(self.full_text)
        
        limit = end_index if end_index is not None else len(clean_full)
        if limit > self.printed_len:
            new_text = clean_full[self.printed_len:limit]
            # Do not strip/clean to preserve code block structure exactly as generated
            formatted_content = self._extract_and_format_code(new_text)
            
            target = live.console if live else self.console
            target.print(Padding(formatted_content, (0, 0, 0, 2)))
            self.printed_len = limit

    def _try_commit(self, live: Live):
        """Пытается зафиксировать текст, если это безопасно (построчно и не внутри кода)."""
        _, clean_full, _ = parse_thought(self.full_text)
        if len(clean_full) <= self.printed_len: return
        
        # 1. Если мы внутри открытого блока кода, рендерим его принудительно, 
        # чтобы пользователь видел прогресс (Fix Streaming Lag)
        if self._is_open_code_block(clean_full):
            # Внимание: здесь мы не сдвигаем printed_len, а просто обновляем Live Display,
            # но _commit_printed_text переносит в лог только завершенные куски.
            # Для фикса лага нам нужно, чтобы _update_live_display показывал "pending" часть,
            # включая незакрытый блок. Это уже работает в _update_live_display.
            # Проблема была в том, что мы не коммитим часть ДО блока кода.
            pass

        # 2. Only commit complete blocks (paragraphs) to ensure correct Markdown rendering
        # We wait for double newline (\n\n) to preserve block formatting (tables, lists, etc.)
        pending = clean_full[self.printed_len:]
        last_newline = pending.rfind('\n\n')
        
        if last_newline != -1:
            # Commit up to the newline (inclusive of the first \n, but keep spacing clean)
            commit_len = last_newline + 2 # Include both \n\n

            candidate_text = pending[:commit_len]
            
            # 3. Verify the candidate slice doesn't split a code block
            # If the slice has an odd number of backticks, it means we are splitting inside a block
            if self._is_open_code_block(candidate_text):
                return
                
            self._commit_printed_text(live, end_index=self.printed_len + commit_len)

    def _is_open_code_block(self, text: str) -> bool:
        """Проверяет, находится ли текст внутри открытого блока кода (нечетное число ```)."""
        return text.count("```") % 2 != 0

    def _extract_text_content(self, content: Any) -> str:
        if isinstance(content, str): return content
        if isinstance(content, list):
            return "".join(x.get("text", "") for x in content if isinstance(x, dict))
        return ""

    def _handle_tool_call(self, tc: Dict[str, Any], live: Live):
        self.tool_buffer[tc["id"]] = {"name": tc["name"], "args": tc["args"]}
        
        t_id = tc["id"]
        if t_id not in self.printed_tool_ids:
            arg_str = self._format_tool_args(tc["args"])
            target = live.console if live else self.console
            
            # Use a cleaner, single-line format with clear iconography
            name_styled = f"[tool.name]{tc['name']}[/]"
            args_styled = f"[tool.args]{arg_str}[/]" if arg_str else ""
            
            target.print(Padding(f"›  {name_styled} {args_styled}", (0, 0, 0, 2)))
            
            self.printed_tool_ids.add(t_id)
            self.status_text = f"Running {tc['name']}..."

    def _handle_tool_result(self, msg: ToolMessage, live: Live = None):
        t_id = msg.tool_call_id
        content_str = str(msg.content)
        is_error = getattr(msg, "status", "") == "error" or content_str.startswith(("Error", "Ошибка"))
        
        target = live.console if live else self.console

        # If we missed the call (rare), print it now
        if t_id in self.tool_buffer and t_id not in self.printed_tool_ids:
            info = self.tool_buffer[t_id]
            arg_str = self._format_tool_args(info["args"])
            target.print(Padding(f"›  [tool.name]{info['name']}[/] [tool.args]{arg_str}[/]", (0, 0, 0, 2)))
            self.printed_tool_ids.add(t_id)
            
        summary = format_tool_output(msg.name, content_str, is_error)
        
        # Modern arrow connector for result
        if is_error:
            icon = "[tool.error]✖ [/]" 
            style = "tool.error"
        else:
            icon = "[tool.result]✔ [/]"
            style = "tool.result"
            
        target.print(Padding(f"  {icon} [{style}]{summary}[/]", (0, 0, 0, 4)))
        self.status_text = "Thinking..."

    def _update_live_display(self, live: Live):
        try:
            thought_content, clean_full, has_thought = parse_thought(self.full_text)
            if has_thought and thought_content:
                self.status_text = "[agent.thought]Thinking...[/]"
            
            pending = clean_full[self.printed_len:]
            
            # Create a structured grid layout for the dashboard
            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(justify="left", ratio=1)
            
            # 1. Status Row
            spinner = Spinner("dots", text=self.status_text, style="status.spinner")
            grid.add_row(spinner)
            
            # 2. Content Row (Thought process)
            if pending.strip():
                 # Fix Streaming Lag: If code block is open, close it artificially for rendering
                 render_text = pending
                 if self._is_open_code_block(render_text):
                     render_text += "\n```"
                     
                 grid.add_row(Padding(Markdown(clean_markdown_text(render_text), code_theme="ansi_dark"), (0, 0, 0, 2)))
                 
            live.update(grid)
        except Exception:
            # Atomic rendering fallback
            pass

    def _format_tool_args(self, args: Any) -> str:
        if isinstance(args, dict):
            arg_str = str(next(iter(args.values()), ""))
        else: arg_str = str(args)
        return (arg_str[:47] + "...") if len(arg_str) > 50 else arg_str

    def _extract_and_format_code(self, text: str) -> Group:
        # Simplified regex to capture everything between backticks
        # We parse the language manually from the content
        pattern = r'```(.*?)```'
        parts = []
        last_end = 0
        
        for match in re.finditer(pattern, text, re.DOTALL):
            if match.start() > last_end:
                pre_text = text[last_end:match.start()]
                if pre_text.strip(): parts.append(Markdown(pre_text, code_theme="dracula"))
            
            content = match.group(1)
            lang = "text"
            code = content
            
            # Try to extract language from the first line or word
            if content:
                first_match = re.match(r'^[ \t]*(\w+)(?:\s|$)(.*)', content, re.DOTALL)
                if first_match:
                    possible_lang = first_match.group(1)
                    rest = first_match.group(2)
                    
                    # Verify it's a valid-looking lang (not just code)
                    if len(possible_lang) < 15: 
                        lang = possible_lang
                        code = rest
            
            # Auto-detection fallback if lang is still text
            if lang == "text" and code.strip():
                first_line = code.strip().split('\n')[0]
                if any(kw in first_line for kw in ['def ', 'class ', 'import ', 'print(', 'if __name__']): lang = "python"
                elif any(kw in first_line for kw in ['function', 'const ', 'let ', 'var ', '=>', 'console.log']): lang = "javascript"
                elif any(kw in first_line for kw in ['package ', 'func ', 'import (', 'go mod']): lang = "go"
                elif any(kw in first_line for kw in ['fn ', 'pub ', 'impl ', 'use std::']): lang = "rust"
                elif any(kw in first_line for kw in ['#include', 'int main', 'std::']): lang = "cpp"
                elif '<' in first_line and '>' in first_line: lang = "html"
                elif '{' in first_line and '}' in first_line: lang = "json"
            
            # Strip leading newline from code if present (common after lang tag)
            if code.startswith('\n'): code = code[1:]
            elif code.startswith('\r\n'): code = code[2:]
            
            syntax = Syntax(code, lang, theme="dracula", line_numbers=True, word_wrap=True, padding=(1, 2))
            parts.append(syntax)
            last_end = match.end()
            
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining.strip(): parts.append(Markdown(remaining, code_theme="dracula"))
            
        return Group(*parts) if parts else Group(Markdown(text, code_theme="dracula"))
