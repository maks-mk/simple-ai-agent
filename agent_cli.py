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
# New imports for enhanced UI
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

# --- CONFIG ---
warnings.filterwarnings("ignore")
console = Console()
logging.getLogger("httpx").setLevel(logging.WARNING)

class MergeCompleter(Completer):
    """
    Объединяет несколько автодополнений (например, команды + пути к файлам).
    Добавлено вручную для совместимости со старыми версиями prompt_toolkit.
    """
    def __init__(self, completers):
        self.completers = completers

    def get_completions(self, document, complete_event):
        for completer in self.completers:
            yield from completer.get_completions(document, complete_event)

# ======================================================
# STREAM PROCESSOR
# ======================================================

class StreamProcessor:
    """
    Handles real-time streaming of agent execution events and messages.
    """
    
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
                      refresh_per_second=12, # Increased for smoother animation
                      console=self.console, 
                      transient=True) as live:
                
                async for mode, payload in agent_app.astream(
                    initial_state,
                    config=config,
                    stream_mode=["messages", "updates"]
                ):
                    await asyncio.sleep(0.01) # Optimized: 0.005 -> 0.01 (100Hz is enough, saves CPU)

                    if mode == "updates":
                        self._handle_updates(payload, live)
                    elif mode == "messages":
                        self._handle_messages(payload, live)
                            
                    self._update_live_display(live)

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.console.print("\n[bold red]🛑 Stopped by user[/]")
            return 

        # self._commit_printed_text(None) # Disable final commit to prevent duplicate block printing
        
        # Manually print any remaining text that wasn't streamed
        _, clean_full, _ = parse_thought(self.full_text)
        if len(clean_full) > self.printed_len:
             new_text = clean_full[self.printed_len:]
             cleaned_chunk = clean_markdown_text(new_text)
             formatted_content = self._extract_and_format_code(cleaned_chunk)
             self.console.print(Padding(formatted_content, (0, 0, 0, 2)))

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

    def _handle_messages(self, payload: tuple, live: Live):
        msg, metadata = payload
        node = metadata.get("langgraph_node")
        self.tracker.update_from_message(msg)
        
        if isinstance(msg, SystemMessage):
            self._handle_system_message(msg, node, live)
            
        if node == "agent" and isinstance(msg, (AIMessage, AIMessageChunk)):
            if msg.tool_calls:
                self._commit_printed_text(live)
                for tc in msg.tool_calls:
                    self._handle_tool_call(tc, live)
            
            if msg.content:
                chunk = self._extract_text_content(msg.content)
                self.full_text += chunk

        elif node == "tools" and isinstance(msg, ToolMessage):
            self._handle_tool_result(msg, live)

    def _handle_system_message(self, msg: SystemMessage, node: str, live: Live):
        content = str(msg.content)
        if node == "tools":
            error_preview = content.split('\n')[0]
            live.console.print(Padding(f"🔧 [bold magenta]Self-Correction:[/bold magenta] {error_preview}", (0, 0, 0, 4)))
            self.status_text = "Correcting strategy..."
        elif node == "token_budget_guard":
             live.console.print(Padding(f"💰 [bold red]Budget Alert:[/bold red] Context limit reached. Switching to wrap-up mode.", (0, 0, 0, 4)))
             self.status_text = "Budget exhausted..."
        elif node == "agent":
            warning_preview = content
            if len(warning_preview) > 100: warning_preview = warning_preview[:97] + "..."
            live.console.print(Padding(f"🛡️ [bold orange3]Quality Gate:[/bold orange3] {warning_preview}", (0, 0, 0, 4)))
            self.status_text = "Safety protocol triggered..."

    def _extract_text_content(self, content: Any) -> str:
        if isinstance(content, str): 
            return content
        if isinstance(content, list):
            return "".join(x.get("text", "") for x in content if isinstance(x, dict))
        return ""

    def _handle_tool_call(self, tc: Dict, live: Live):
        t_id, t_name = tc.get("id"), tc.get("name")
        args = tc.get("args", {})
        self.tool_buffer[t_id] = {"name": t_name, "args": args}
        arg_str = self._format_tool_args(args)
        arg_display = f" [dim]{arg_str}[/]" if arg_str else ""
        self.status_text = f"[bold cyan]Thinking:[/] {t_name}{arg_display}"
            
    def _handle_tool_result(self, msg: ToolMessage, live: Live):
        t_id = msg.tool_call_id
        content_str = str(msg.content)
        is_error = getattr(msg, "status", "") == "error" or content_str.startswith(("Error", "Ошибка"))
    
        tool_meta = {
            "default": {"icon": "🔧", "color": "cyan"},
            "search": {"icon": "🔍", "color": "magenta"},
            "file": {"icon": "📄", "color": "blue"},
            "write": {"icon": "✏️", "color": "yellow"},
            "web": {"icon": "🌐", "color": "green"},
            "exec": {"icon": "⚡", "color": "red"}
        }
    
        category = "default"
        if msg.name:
            name_lower = msg.name.lower()
            if any(k in name_lower for k in ["search", "query", "find"]): category = "search"
            elif any(k in name_lower for k in ["read", "view", "list", "dir", "cat"]): category = "file"
            elif any(k in name_lower for k in ["write", "save", "edit", "create"]): category = "write"
            elif any(k in name_lower for k in ["web", "crawl", "fetch", "http"]): category = "web"
            elif any(k in name_lower for k in ["exec", "run", "bash", "shell", "cmd"]): category = "exec"
    
        style = tool_meta[category]
    
        if t_id in self.tool_buffer and t_id not in self.printed_tool_ids:
            info = self.tool_buffer[t_id]
            t_name = info["name"]
            args = info["args"]
            arg_str = self._format_tool_args(args)
            
            header = f"{style['icon']} [{style['color']}]{t_name}[/]"
            if arg_str: header += f" [dim]· {arg_str}[/]"
            live.console.print(Padding(header, (0, 0, 0, 2)))
            self.printed_tool_ids.add(t_id)
    
        summary = format_tool_output(msg.name, content_str, is_error)
    
        if is_error:
            error_panel = Panel(
                f"[bold]❌ {msg.name} failed[/]\n{summary}",
                border_style="red",
                padding=(0, 2),
                width=min(100, self.console.width - 6)
            )
            live.console.print(Padding(error_panel, (0, 0, 0, 4)))
        else:
            connector = f"[{style['color']}]└─[/]"
            live.console.print(Padding(f"{connector} {summary}", (0, 0, 0, 4)))
    
        self.status_text = "Analyzing..."
    
    def _format_tool_args(self, args: Any) -> str:
        arg_str = ""
        if isinstance(args, dict):
            priority_keys = ["query", "queries", "path", "file_path", "url", "urls", "filename"]
            for key in priority_keys:
                if key in args:
                    val = args[key]
                    arg_str = str(val) if isinstance(val, list) else str(val)
                    break
            if not arg_str and args:
                arg_str = str(list(args.values())[0])
        elif isinstance(args, str):
            arg_str = args
            
        clean_arg = str(arg_str).strip().replace("\n", " ")
        if len(clean_arg) > 50: 
            return clean_arg[:47] + "..."
        return clean_arg

    def _extract_and_format_code(self, text: str) -> Group:
        pattern = r'```(\w+)?\n(.*?)```'
        parts = []
        last_end = 0
        
        for match in re.finditer(pattern, text, re.DOTALL):
            if match.start() > last_end:
                pre_text = text[last_end:match.start()]
                if pre_text.strip(): parts.append(Markdown(pre_text))
            
            lang = match.group(1) or "text"
            code = match.group(2)
            
            if lang == "text" and code.strip():
                first_line = code.strip().split('\n')[0]
                if any(kw in first_line for kw in ['def ', 'class ', 'import ', 'print(']): lang = "python"
                elif any(kw in first_line for kw in ['function', 'const ', 'let ', 'var ', '=>']): lang = "javascript"
                elif '<' in first_line and '>' in first_line: lang = "html"
                elif '{' in first_line and '}' in first_line: lang = "json"
            
            syntax = Syntax(code, lang, theme="monokai", line_numbers=True, word_wrap=True, padding=(1, 2))
            parts.append(syntax)
            last_end = match.end()
        
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining.strip(): parts.append(Markdown(remaining))
        
        return Group(*parts) if parts else Group(Markdown(text))

    def _commit_printed_text(self, live: Optional[Live]):
        _, clean_full, _ = parse_thought(self.full_text)
        if len(clean_full) > self.printed_len:
            new_text = clean_full[self.printed_len:]
            cleaned_chunk = clean_markdown_text(new_text)
            target = live.console if live else self.console
            formatted_content = self._extract_and_format_code(cleaned_chunk)
            target.print(Padding(formatted_content, (0, 0, 0, 2)))
            self.printed_len = len(clean_full)

    def _update_live_display(self, live: Live):
        thought_content, clean_full, has_thought = parse_thought(self.full_text)
        if has_thought and thought_content:
            self.status_text = "[yellow italic]Thinking...[/]"
        
        pending = clean_full[self.printed_len:]
        renderable = Spinner("dots", text=self.status_text, style="cyan")
        if pending.strip():
             renderable = Group(Padding(Markdown(clean_markdown_text(pending)), (0, 0, 0, 2)), renderable)
        live.update(renderable)

# ======================================================
# UI HELPERS
# ======================================================

def render_chat_history(console: Console, messages: List[Any]):
    """Renders the entire chat history to the console."""
    if not messages:
        return

    for msg in messages:
        if isinstance(msg, HumanMessage):
            console.print(Padding(Panel(
                Markdown(str(msg.content).strip()),
                title="[bold green]You[/]",
                title_align="right",
                border_style="green",
                padding=(0, 1),
                subtitle_align="right"
            ), (1, 0, 1, 4))) # Add some spacing
            
        elif isinstance(msg, AIMessage):
            # Parse thought to hide it or show it differently
            thought, content, _ = parse_thought(str(msg.content))
            
            # 1. Tool Calls (Compact)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "tool")
                    console.print(Padding(f"🔧 [cyan]{name}[/] [dim]...[/]", (0, 0, 0, 8)))

            # 2. Content
            if content.strip():
                console.print(Padding(Panel(
                    Markdown(content.strip()),
                    title="[bold blue]Agent[/]",
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
            
            style = "red" if is_error else "dim cyan"
            icon = "❌" if is_error else "└─"
            console.print(Padding(f"[{style}]{icon} {summary}[/]", (0, 0, 0, 8)))
            
    # Extra space at bottom before prompt
    # console.print()

def get_prompt_message():
    """
    Generates a stylish prompt with a smart-shortened path.
    Example: ~/Projects/.../src/utils
    """
    cwd = Path.cwd()
    home = Path.home()
    
    # 1. Попытка заменить путь к пользователю на ~
    try:
        # Получаем части пути относительно home
        parts = ("~",) + cwd.relative_to(home).parts
    except ValueError:
        # Если мы не в home (например на другом диске), берем полный путь
        parts = cwd.parts

    # 2. Умное сокращение: если вложенность больше 4 папок
    if len(parts) > 4:
        # Оставляем: начало (~ или C:) + "..." + пред-последнюю + текущую папку
        # Пример: ~/Projects/.../backend/src
        display_parts = [parts[0], "\u2026"] + list(parts[-2:]) 
    else:
        display_parts = parts

    # Собираем строку через / (красивее для промпта, чем \)
    path_str = "/".join(display_parts).replace("\\", "/")

    # 3. HTML разметка
    return HTML(
        f'<style bg="#0077c2" fg="white"> Agent </style>'
        f'<style fg="#0077c2"></style>'
        f'<style fg="#ansigreen"> {path_str} </style>'
        f'<style fg="#ansigreen" bold="true">❯</style> '
    )
    
def get_bottom_toolbar():
    """Returns the bottom status toolbar."""
    return HTML(
        ' <b>ALT+ENTER</b> Multiline '
        '<style fg="gray">|</style> <b>/tools</b> List '
        '<style fg="gray">|</style> <b>exit</b> Quit '
        '<style fg="gray">|</style> <style bg="ansiyellow" fg="black"> IDLE </style>'
    )

def show_help(workflow: AgentWorkflow):
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Tool", style="green")
    table.add_column("Description")
    
    for t in workflow.tools:
        desc = t.description.split("\n")[0] if t.description else "No description"
        if len(desc) > 60: desc = desc[:57] + "..."
        table.add_row(t.name, desc)

    console.print(Panel(
        Group(
            Markdown("### 🎮 Commands"),
            Markdown("- `/help` - Show this menu"),
            Markdown("- `/tools` - Show available tools"),
            Markdown("- `/refresh` - Redraw chat history"),
            Markdown("- `exit` / `quit` - Close application"),
            Markdown("- `clear` / `reset` - Start new session"),
            Markdown("- `Alt+Enter` - Multi-line input"),
            Markdown("---"),
            Markdown("### 🛠 Available Tools"),
            table
        ),
        title="[bold blue]Help Menu[/]",
        border_style="blue"
    ))

# ======================================================
# MAIN ENTRY POINT
# ======================================================

async def main():
    os.system("cls" if os.name == "nt" else "clear")
    load_dotenv(BASE_DIR / '.env')
    
    console.print(Panel("[bold blue]AI Agent CLI[/]", subtitle="v7.0 Enhanced"))

    # 1. Load Config
    try:
        temp_cfg = AgentConfig()
        token_budget = temp_cfg.token_budget
        if temp_cfg.debug:
            logger.setLevel(logging.DEBUG)
            console.print("[yellow]🐛 Debug mode enabled[/]")
        else:
            logger.setLevel(logging.WARNING)
    except Exception as e:
        console.print(f"[bold red]Config Error:[/] {e}")
        return

    # 2. Initialize Workflow
    try:
        with console.status("[bold green]Initializing system...[/]", spinner="dots"):
            workflow = AgentWorkflow()
            await workflow.initialize_resources()
            agent_app = workflow.build_graph()
        console.print("[bold green]System Ready![/]")
    except Exception as e:
        console.print(f"[bold red]Init Error:[/] {e}")
        if temp_cfg.debug:
            import traceback
            traceback.print_exc()
        return

    # 3. Show Status Bar
    cfg = workflow.config
    console.print(
        f"[dim]Model:[/] [bold cyan]{cfg.gemini_model if cfg.provider == 'gemini' else cfg.openai_model}[/] "
        f"[dim]Temp:[/] [bold cyan]{cfg.temperature}[/] "
        f"[dim]Budget:[/] [bold cyan]{token_budget}[/]\n"
    )

    # 4. Input Configuration
    # Auto-completion for commands and local files
    command_completer = WordCompleter(['/help', '/tools', 'exit', 'quit', 'clear', 'reset'], ignore_case=True)
    combined_completer = MergeCompleter([command_completer, PathCompleter()])

    # Enhanced Session
    session = PromptSession(
        history=FileHistory(".history"),
        style=Style.from_dict({
            "completion-menu.completion": "bg:#008888 #ffffff",
            "completion-menu.completion.current": "bg:#00aaaa #000000",
            "scrollbar.background": "bg:#88aaaa",
            "scrollbar.button": "bg:#222222",
        }),
        key_bindings=get_key_bindings(),
        lexer=PygmentsLexer(MarkdownLexer),
        completer=combined_completer,
        complete_style=CompleteStyle.MULTI_COLUMN,
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True
    )

    thread_id = "main_session"
    last_stats = None

    # Initial Header
    os.system("cls" if os.name == "nt" else "clear")
    
    # Header Info
    model_name = cfg.gemini_model if cfg.provider == 'gemini' else cfg.openai_model
    tool_count = len(workflow.tools)
    header_text = f"[bold blue]AI Agent CLI[/]\n[dim]Model: {model_name} | Tools: {tool_count}[/]"
    
    console.print(Panel(header_text, subtitle="v7.0 Enhanced", border_style="dim"))

    while True:
        try:
            # --- CHAT UI REFRESH ---
            # Removed automatic clearing to prevent "flying up" of text
            # os.system("cls" if os.name == "nt" else "clear")
            # console.print(Panel("[bold blue]AI Agent CLI[/]", subtitle="v7.0 Enhanced", border_style="dim"))
            
            # Render History
            # try:
            #     state = await agent_app.get_state({"configurable": {"thread_id": thread_id}})
            #     if state and state.values:
            #         render_chat_history(console, state.values.get("messages", []))
            # except Exception:
            #     pass
            
            if last_stats:
                console.print(last_stats, justify="right")
                last_stats = None

            # Sticky Bottom Logic (Safe Version)
            # 1. Ensure we have space at the bottom by printing newlines (forces scroll if full)
            # 2. Move cursor back up to the reserved space
            if sys.stdout.isatty():
               h = shutil.get_terminal_size().lines
               # Reserve 3 lines (Prompt + Toolbar + Buffer)
               # Print 3 newlines to guarantee empty space at bottom
               print("\n" * 3, end="", flush=True) 
               # Move cursor to h-2 (leaving 2 lines for prompt/toolbar)
               print(f"\033[{h-2};0H", end="", flush=True)

            # Stylish Prompt with Toolbar
            user_input = await session.prompt_async(
                get_prompt_message(),
                bottom_toolbar=get_bottom_toolbar,
                refresh_interval=0.5
            )
            user_input = user_input.strip()
            
            if not user_input: continue
            
            # Commands
            cmd = user_input.lower()
            if cmd in ["exit", "quit"]: break
            if cmd in ["clear", "reset"]:
                thread_id = f"session_{int(time.time())}"
                # console.print("[yellow]♻ New session started[/]") # No need to print, next loop clears screen
                continue
            
            if cmd == "/help" or cmd == "/tools":
                show_help(workflow)
                # await session.prompt_async(HTML("<style fg='gray'>Press Enter to continue...</style>"))
                continue

            if cmd == "/refresh":
                os.system("cls" if os.name == "nt" else "clear")
                console.print(Panel(header_text, subtitle="v7.0 Enhanced", border_style="dim"))
                try:
                    state = await agent_app.get_state({"configurable": {"thread_id": thread_id}})
                    if state and state.values:
                        render_chat_history(console, state.values.get("messages", []))
                except Exception:
                    pass
                continue

            # Run Agent
            processor = StreamProcessor(console)
            last_stats = await processor.run(
                agent_app, 
                user_input, 
                thread_id, 
                cfg.max_loops, 
                token_budget=token_budget 
            )
            # console.print()

        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]Cancelled. Type 'exit' to quit.[/]")
            continue
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            if cfg.debug:
                import traceback
                logger.debug(traceback.format_exc())

    # Final Cleanup
    if hasattr(workflow, 'tool_registry') and workflow.tool_registry:
        await workflow.tool_registry.cleanup()
    
    try:
        from tools.system_tools import _net_client
        if _net_client: await _net_client.close()
    except ImportError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass