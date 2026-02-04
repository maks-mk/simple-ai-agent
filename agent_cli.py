import sys
import os
import shutil
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
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import WordCompleter, PathCompleter, Completer
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.shortcuts import CompleteStyle

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

    async def run(self, agent_app, user_input: str, thread_id: str, max_loops: int, token_budget: int = 0):
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_loops * 4}
        initial_state = {
            "messages": [HumanMessage(content=user_input)], 
            "steps": 0,
            "token_budget": token_budget, 
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
        
        if isinstance(msg, SystemMessage):
            self._handle_system_message(msg, node, live)
            
        if node == "agent" and isinstance(msg, (AIMessage, AIMessageChunk)):
            if msg.tool_calls:
                self._commit_printed_text(live)
                # Removed eager printing to prevent empty/partial tool calls
            
            if msg.content:
                chunk = self._extract_text_content(msg.content)
                self.full_text += chunk
                # Removed eager commit on newline to allow code blocks to complete

        elif node == "tools" and isinstance(msg, ToolMessage):
            self._handle_tool_result(msg, live)

    def _handle_system_message(self, msg: SystemMessage, node: str, live: Live):
        content = str(msg.content)
        target = live.console if live else self.console
        
        if "SYSTEM ALERT" in content:
            # Extract the error part for cleaner display
            clean_err = content.replace("SYSTEM ALERT:", "").strip().split("\n")[0]
            target.print(Padding(f"[bold red]⚠ System Alert:[/][red] {clean_err}[/]", (0, 0, 0, 4)))
            self.status_text = "Self-correcting..."
            
        elif "REFLECTION" in content:
            target.print(Padding(f"[bold yellow]↺ Reflection:[/][yellow italic] Adjusting strategy...[/]", (0, 0, 0, 4)))
            self.status_text = "Reflecting..."

    def _commit_printed_text(self, live: Optional[Live]):
        """Переносит текст из динамического Live в статический лог консоли."""
        _, clean_full, _ = parse_thought(self.full_text)
        if len(clean_full) > self.printed_len:
            new_text = clean_full[self.printed_len:]
            # Do not strip to preserve markdown structure
            cleaned_chunk = clean_markdown_text(new_text)
            formatted_content = self._extract_and_format_code(cleaned_chunk)
            
            target = live.console if live else self.console
            target.print(Padding(formatted_content, (0, 0, 0, 2)))
            self.printed_len = len(clean_full)

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
        # Improved regex to handle optional spaces, various newlines, and loose language identifiers
        pattern = r'```[ \t]*(\w*)?[ \t]*(?:\r?\n|\r)(.*?)```'
        parts = []
        last_end = 0
        
        for match in re.finditer(pattern, text, re.DOTALL):
            if match.start() > last_end:
                pre_text = text[last_end:match.start()]
                if pre_text.strip(): parts.append(Markdown(pre_text, code_theme="ansi_dark"))
            
            lang = match.group(1) or "text"
            code = match.group(2)
            
            # Simple heuristic to detect language if not specified
            if lang == "text" and code.strip():
                first_line = code.strip().split('\n')[0]
                if any(kw in first_line for kw in ['def ', 'class ', 'import ', 'print(', 'if __name__']): lang = "python"
                elif any(kw in first_line for kw in ['function', 'const ', 'let ', 'var ', '=>', 'console.log']): lang = "javascript"
                elif any(kw in first_line for kw in ['package ', 'func ', 'import (', 'go mod']): lang = "go"
                elif any(kw in first_line for kw in ['fn ', 'pub ', 'impl ', 'use std::']): lang = "rust"
                elif any(kw in first_line for kw in ['#include', 'int main', 'std::']): lang = "cpp"
                elif '<' in first_line and '>' in first_line: lang = "html"
                elif '{' in first_line and '}' in first_line: lang = "json"
            
            # Add padding to code blocks for better readability
            # Use 'ansi_dark' which usually looks better on dark terminals than monokai
            syntax = Syntax(code, lang, theme="ansi_dark", line_numbers=True, word_wrap=True, padding=(1, 2))
            parts.append(syntax)
            last_end = match.end()
            
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining.strip(): parts.append(Markdown(remaining, code_theme="ansi_dark"))
            
        return Group(*parts) if parts else Group(Markdown(text, code_theme="ansi_dark"))

# ======================================================
# UI HELPERS (RESTORED)
# ======================================================

def render_chat_history(console: Console, messages: List[Any]):
    if not messages: return
    for msg in messages:
        if isinstance(msg, HumanMessage):
            console.print(Padding(Panel(Markdown(str(msg.content).strip()), title="[user.say]You[/]", title_align="right", border_style="green", padding=(0, 1), subtitle_align="right"), (1, 0, 1, 4)))
            
        elif isinstance(msg, AIMessage):
            # Parse thought to hide it or show it differently
            thought, content, _ = parse_thought(str(msg.content))
            

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "tool")
                    console.print(Padding(f"🔧 [tool.name]{name}[/] [dim]...[/]", (0, 0, 0, 8)))

            # 2. Content
            if content.strip():
                console.print(Padding(Panel(
                    Markdown(content.strip()),
                    title="[agent.say]Agent[/]",
                    title_align="left",
                    border_style="blue",
                    padding=(0, 1)
                ), (0, 4, 1, 0)))
                
        elif isinstance(msg, ToolMessage):
            # Show tool result compactly
            name = msg.name or "tool"
            content = str(msg.content)
            is_error = getattr(msg, "status", "") == "error" or content.startswith(("Error", "Ошибка"))
            summary = format_tool_output(name, content, is_error)
            
            style = "tool.error" if is_error else "tool.result"
            icon = "❌" if is_error else "└─"
            console.print(Padding(f"[{style}]{icon} {summary}[/]", (0, 0, 0, 8)))

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
# MAIN (RESTORED LOGIC)
# ======================================================

async def main():
    os.system("cls" if os.name == "nt" else "clear")
    load_dotenv(BASE_DIR / '.env')

    # 1. Load Config
    try:
        temp_cfg = AgentConfig()
        
        # Re-initialize logging with config values
        # If debug is False, show only WARNING+ in console to keep UI clean.
        # File logs will still capture everything down to DEBUG because file_handler is configured independently.
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
    console.print(Panel(header_info, subtitle="v7.2b"))

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
                # get_state returns StateSnapshot which is not awaitable in some versions or context
                # But here it seems to be an async method in the compiled graph? 
                # Actually, compiled graph .get_state() is async.
                current_state = await agent_app.get_state(config)
                
                # StateSnapshot object behaves like a named tuple/object
                if current_state and hasattr(current_state, "values") and current_state.values:
                    messages = current_state.values.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                            console.print("[yellow]⚠ Detected interrupted tool execution. Auto-fixing history...[/]")
                            tool_msgs = []
                            for tc in last_msg.tool_calls:
                                tool_msgs.append(ToolMessage(
                                    tool_call_id=tc["id"],
                                    content="Error: Execution interrupted by user.",
                                    name=tc["name"]
                                ))
                            await agent_app.update_state(config, {"messages": tool_msgs}, as_node="tools")
            except Exception as e:
                # logger.warning(f"State repair failed: {e}")
                pass

            processor = StreamProcessor(console)
            last_stats = await processor.run(agent_app, user_input, thread_id, temp_cfg.max_loops, token_budget=temp_cfg.token_budget)

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