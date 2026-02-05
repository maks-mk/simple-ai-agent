import sys
import os
import asyncio
import warnings
import time
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dotenv import load_dotenv

# --- PROJECT IMPORTS ---
from core.constants import BASE_DIR
from core.ui_theme import AGENT_THEME

# Ensure project modules take precedence
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# --- UI IMPORTS ---
from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.padding import Padding
from rich.table import Table
from rich import box
from rich.syntax import Syntax

# --- PROMPT IMPORTS ---
from prompt_toolkit import PromptSession
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import WordCompleter, PathCompleter, Completer
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

# --- LANGCHAIN IMPORTS ---
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk, SystemMessage

# --- LOCAL IMPORTS ---
try:
    from agent import AgentWorkflow, logger
except ImportError as e:
    if str(Path.cwd()) != str(BASE_DIR):
        sys.path.append(".")
    try:
        from agent import AgentWorkflow, logger
    except ImportError:
        raise ImportError(f"Could not import 'agent' module. sys.path: {sys.path}. Error: {e}")

from core.cli_utils import (
    TokenTracker, 
    clean_markdown_text, 
    parse_thought, 
    format_tool_output, 
    get_key_bindings
)
from core.config import AgentConfig
from core.logging_config import setup_logging

# --- CONFIG ---
warnings.filterwarnings("ignore")
console = Console(theme=AGENT_THEME)
logging.getLogger("httpx").setLevel(logging.WARNING)

class MergeCompleter(Completer):
    def __init__(self, completers):
        self.completers = completers
    def get_completions(self, document, complete_event):
        for completer in self.completers:
            yield from completer.get_completions(document, complete_event)

# ======================================================
# STREAM PROCESSOR (STABLE RENDER)
# ======================================================

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
            "messages": [HumanMessage(content=user_input)], 
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
        
        # 1. Do not commit if we are inside a code block (wait for it to close)
        if self._is_open_code_block(clean_full): return
        
        # 2. Only commit complete lines to avoid breaking Markdown paragraphs or words
        pending = clean_full[self.printed_len:]
        last_newline = pending.rfind('\n')
        
        if last_newline != -1:
            # Commit up to the last newline (inclusive)
            commit_len = last_newline + 1
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
            suffix = f" [tool.args]· {arg_str}[/]" if arg_str else ""
            target.print(Padding(f"🔧 [tool.name]{tc['name']}[/]{suffix}", (0, 0, 0, 2)))
            self.printed_tool_ids.add(t_id)
            self.status_text = f"Running {tc['name']}..."

    def _handle_tool_result(self, msg: ToolMessage, live: Live = None):
        t_id = msg.tool_call_id
        content_str = str(msg.content)
        is_error = getattr(msg, "status", "") == "error" or content_str.startswith(("Error", "Ошибка"))
        
        target = live.console if live else self.console

        if t_id in self.tool_buffer and t_id not in self.printed_tool_ids:
            info = self.tool_buffer[t_id]
            arg_str = self._format_tool_args(info["args"])
            target.print(Padding(f"🔧 [tool.name]{info['name']}[/] [tool.args]· {arg_str}[/]", (0, 0, 0, 2)))
            self.printed_tool_ids.add(t_id)
            
        summary = format_tool_output(msg.name, content_str, is_error)
        icon = "[tool.error]❌[/]" if is_error else "[tool.result]└─[/]"
        target.print(Padding(f"{icon} {summary}", (0, 0, 0, 4)))
        self.status_text = "Analyzing..."

    def _update_live_display(self, live: Live):
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
             grid.add_row(Padding(Markdown(clean_markdown_text(pending), code_theme="ansi_dark"), (0, 0, 0, 2)))
             
        live.update(grid)

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
            # Handle cases like "python\n", "python code", or just "\n"
            if content:
                # Split only on the first newline/space to separate lang tag
                # We need to be careful: "python\ncode" vs "code"
                # Heuristic: if starts with newline, no lang.
                # If starts with word chars then space/newline, that's lang.
                
                first_match = re.match(r'^[ \t]*(\w+)(?:\s|$)(.*)', content, re.DOTALL)
                if first_match:
                    possible_lang = first_match.group(1)
                    rest = first_match.group(2)
                    
                    # Verify it's a valid-looking lang (not just code)
                    # Common langs or short identifiers
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

# ======================================================
# UI HELPERS
# ======================================================

def get_prompt_message():
    cwd = Path.cwd()
    home = Path.home()
    try:
        parts = ("~",) + cwd.relative_to(home).parts
    except ValueError:
        parts = cwd.parts
    if len(parts) > 4:
        display_parts = [parts[0], "\u2026"] + list(parts[-2:]) 
    else:
        display_parts = parts
    path_str = "/".join(display_parts).replace("\\", "/")
    return HTML(f'<style bg="#0077c2" fg="white"> Agent </style><style fg="#0077c2"></style><style fg="#ansigreen"> {path_str} </style><style fg="#ansigreen" bold="true">❯</style> ')

def get_bottom_toolbar():
    return HTML(' <b>ALT+ENTER</b> Multiline | <b>/tools</b> List | <b>/help</b> Help | <b>exit</b> Quit ')

def show_help(workflow: AgentWorkflow):
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Tool")
    table.add_column("Description")
    for t in workflow.tools:
        table.add_row(t.name, (t.description[:60] + "...") if t.description else "No description")
    console.print(Panel(table, title="[bold blue]Available Tools[/]"))

# ======================================================
# MAIN
# ======================================================

async def main():
    os.system("cls" if os.name == "nt" else "clear")
    load_dotenv(BASE_DIR / '.env')

    # 1. Load Config
    try:
        temp_cfg = AgentConfig()
        log_level = logging.DEBUG if temp_cfg.debug else logging.WARNING
        setup_logging(level=log_level)
        
    except Exception as e:
        console.print(f"[bold red]Config Error:[/] {e}")
        return

    # 2. Initialize Workflow
    try:
        with console.status("[bold green]Initializing system...[/]"):
            workflow = AgentWorkflow()
            await workflow.initialize_resources()
            agent_app = workflow.build_graph()
    except Exception as e:
        console.print(f"[bold red]Init Error:[/] {e}")
        return

    # Clear and Print Header with Info
    os.system("cls" if os.name == "nt" else "clear")
    
    model_name = temp_cfg.gemini_model if temp_cfg.provider == "gemini" else temp_cfg.openai_model
    header_info = f"[bold blue]AI Agent CLI[/]\n[dim]Model: {model_name} | Tools: {len(workflow.tools)}[/]"
    console.print(Panel(header_info, subtitle="v7.3b"))

    if temp_cfg.debug:
        console.print("[yellow]🐛 Debug mode enabled[/]")

    # 3. Session Setup
    session = PromptSession(
        history=FileHistory(".history"),
        completer=MergeCompleter([WordCompleter(['/help', '/tools', 'exit', 'clear']), PathCompleter()]),
        key_bindings=get_key_bindings(),
        lexer=PygmentsLexer(MarkdownLexer),
        auto_suggest=AutoSuggestFromHistory()
    )

    thread_id = "main_session"
    last_stats = None

    while True:
        try:
            if last_stats:
                console.print(last_stats, justify="right")
                last_stats = None

            user_input = await session.prompt_async(get_prompt_message(), bottom_toolbar=get_bottom_toolbar)
            user_input = user_input.strip()
            
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break
            if user_input.lower() in ["clear", "reset"]:
                thread_id = f"session_{int(time.time())}"
                os.system("cls" if os.name == "nt" else "clear")
                continue
            
            if user_input.lower() in ["/help", "/tools"]:
                show_help(workflow)
                continue

            # Auto-fix for interrupted tool calls
            try:
                config = {"configurable": {"thread_id": thread_id}}
                current_state = agent_app.get_state(config)
                
                if current_state and current_state.values:
                    messages = current_state.values.get("messages", [])
                    if messages:
                        # Find the last AIMessage with tool calls
                        last_ai_msg = None
                        last_ai_idx = -1
                        for i in range(len(messages) - 1, -1, -1):
                            m = messages[i]
                            if isinstance(m, (AIMessage, AIMessageChunk)) and m.tool_calls:
                                last_ai_msg = m
                                last_ai_idx = i
                                break
                        
                        if last_ai_msg:
                            # Gather existing ToolMessages after this AIMessage
                            existing_tool_outputs = set()
                            for j in range(last_ai_idx + 1, len(messages)):
                                m = messages[j]
                                if isinstance(m, ToolMessage):
                                    existing_tool_outputs.add(m.tool_call_id)
                            
                            # Identify missing responses
                            missing_tool_calls = []
                            for tc in last_ai_msg.tool_calls:
                                if tc["id"] not in existing_tool_outputs:
                                    missing_tool_calls.append(tc)
                            
                            if missing_tool_calls:
                                console.print(f"[dim]⚠ Detected {len(missing_tool_calls)} interrupted tool execution(s). Filling gaps...[/]")
                                tool_msgs = []
                                for tc in missing_tool_calls:
                                    tool_msgs.append(ToolMessage(
                                        tool_call_id=tc["id"],
                                        content="Error: Execution interrupted by user.",
                                        name=tc["name"]
                                    ))
                                # update_state returns a dict, not awaitable in some versions
                                agent_app.update_state(config, {"messages": tool_msgs}, as_node="tools")
                                console.print("[dim]✔ History repaired (filled gaps). Ready for new input.[/]")
            except Exception as e:
                # Silently fail or log debug if state repair fails, to not block the user
                pass

            processor = StreamProcessor(console)
            last_stats = await processor.run(agent_app, user_input, thread_id, temp_cfg.max_loops)

        except (KeyboardInterrupt, asyncio.CancelledError):
            continue
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")

    # Final Cleanup
    if hasattr(workflow, 'tool_registry') and workflow.tool_registry:
        await workflow.tool_registry.cleanup()
    try:
        from tools.system_tools import _net_client
        if _net_client: await _net_client.close()
    except ImportError: pass

if __name__ == "__main__":
    asyncio.run(main())
