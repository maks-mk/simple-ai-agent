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

    async def process_stream(self, stream):
        """
        Consumes the agent event stream and updates the UI.
        """
        try:
            with Live(Spinner("dots", text=self.status_text, style="cyan"), 
                      refresh_per_second=12, 
                      console=self.console, 
                      transient=True) as live:
                
                async for mode, payload in stream:
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
        """Transfers text from dynamic Live to static console log."""
        _, clean_full, _ = parse_thought(self.full_text)
        
        limit = end_index if end_index is not None else len(clean_full)
        if limit > self.printed_len:
            new_text = clean_full[self.printed_len:limit]
            
            # Use standard rich Markdown renderer
            formatted_content = Markdown(new_text, code_theme="dracula")
            
            target = live.console if live else self.console
            target.print(Padding(formatted_content, (0, 0, 0, 2)))
            self.printed_len = limit

    def _try_commit(self, live: Live):
        """Attempts to commit text if safe (paragraph boundaries)."""
        _, clean_full, _ = parse_thought(self.full_text)
        if len(clean_full) <= self.printed_len: return
        
        # Only commit complete blocks (paragraphs) to ensure correct Markdown rendering
        # We wait for double newline (\n\n) to preserve block formatting
        pending = clean_full[self.printed_len:]
        last_newline = pending.rfind('\n\n')
        
        if last_newline != -1:
            # Commit up to the newline
            commit_len = last_newline + 2 
            
            # Verify we are not inside a code block (simple check)
            candidate_text = pending[:commit_len]
            if candidate_text.count("```") % 2 != 0:
                return

            self._commit_printed_text(live, end_index=self.printed_len + commit_len)

    def _extract_text_content(self, content: Any) -> str:
        if isinstance(content, str): return content
        if isinstance(content, list):
            return "".join(x.get("text", "") for x in content if isinstance(x, dict))
        return ""

    def _handle_tool_call(self, tc: Dict[str, Any], live: Live):
        self.tool_buffer[tc["id"]] = {"name": tc["name"], "args": tc["args"]}
        
        t_id = tc["id"]
        if t_id not in self.printed_tool_ids:
            # Use smart formatting from deepagents-cli style
            display_str = format_tool_display(tc["name"], tc["args"])
            target = live.console if live else self.console
            
            # Extract name and args from the formatted string for styling
            # format_tool_display returns "name(args)"
            # We want to style "name" and "args" separately if possible, or just print the whole thing
            
            if "(" in display_str:
                name_part, args_part = display_str.split("(", 1)
                args_part = "(" + args_part # Add back the parenthesis
                
                name_styled = f"[tool.name]{name_part}[/]"
                args_styled = f"[tool.args]{args_part}[/]"
                display_styled = f"{name_styled}{args_styled}"
            else:
                display_styled = f"[tool.name]{display_str}[/]"
            
            target.print(Padding(f"›  {display_styled}", (0, 0, 0, 2)))
            
            self.printed_tool_ids.add(t_id)
            self.status_text = f"Running {tc['name']}..."

    def _handle_tool_result(self, msg: ToolMessage, live: Live = None):
        t_id = msg.tool_call_id
        content_str = str(msg.content)
        is_error = getattr(msg, "status", "") == "error" or content_str.startswith(("Error", "Ошибка"))
        
        target = live.console if live else self.console

        if t_id in self.tool_buffer and t_id not in self.printed_tool_ids:
            info = self.tool_buffer[t_id]
            # Use smart formatting here too if we missed the call
            display_str = format_tool_display(info["name"], info["args"])
            target.print(Padding(f"›  [tool.name]{display_str}[/]", (0, 0, 0, 2)))
            self.printed_tool_ids.add(t_id)
            
        summary = format_tool_output(msg.name, content_str, is_error)
        
        if is_error:
            icon = "[tool.error]✖ [/]" 
            style = "tool.error"
        else:
            icon = "[tool.result]✔ [/]"
            style = "tool.result"
            
        target.print(Padding(f"  {icon} [{style}]{summary}[/]", (0, 0, 0, 4)))
        
        if not is_error:
            diff_match = re.search(r"```diff\n(.*?)```", content_str, re.DOTALL)
            if diff_match:
                diff_code = diff_match.group(1).strip()
                syntax = Syntax(diff_code, "diff", theme="monokai", line_numbers=True, word_wrap=True)
                target.print(Padding(syntax, (0, 0, 0, 6)))

        self.status_text = "Thinking..."

    def _update_live_display(self, live: Live):
        try:
            thought_content, clean_full, has_thought = parse_thought(self.full_text)
            if has_thought and thought_content:
                self.status_text = "[agent.thought]Thinking...[/]"
            
            pending = clean_full[self.printed_len:]
            
            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(justify="left", ratio=1)
            
            spinner = Spinner("dots", text=self.status_text, style="status.spinner")
            grid.add_row(spinner)
            
            if pending.strip():
                 # Append closing block if open to help rendering
                 render_text = pending
                 if render_text.count("```") % 2 != 0:
                     render_text += "\n```"
                     
                 grid.add_row(Padding(Markdown(clean_markdown_text(render_text), code_theme="ansi_dark"), (0, 0, 0, 2)))
                 
            live.update(grid)
        except Exception:
            pass

