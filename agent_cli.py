import asyncio
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# --- PROJECT IMPORTS ---
from core.constants import BASE_DIR
from core.ui_theme import AGENT_THEME

# Ensure project modules take precedence
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# --- UI IMPORTS (lazy loaded later if needed) ---
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --- PROMPT IMPORTS ---
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer

# --- LOCAL IMPORTS ---
try:
    from agent import build_agent_app
except ImportError as e:
    if str(Path.cwd()) != str(BASE_DIR):
        sys.path.append(".")
    try:
        from agent import build_agent_app
    except ImportError:
        raise ImportError(f"Could not import 'agent' module. sys.path: {sys.path}. Error: {e}")

from core.cli_utils import format_exception_friendly, get_key_bindings
from core.config import AgentConfig
from core.fuzzy_completer import FuzzyPathCompleter
from core.logging_config import setup_logging
from core.session_utils import repair_session_if_needed

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
    return HTML(
        f'<style bg="#0077c2" fg="white"> Agent </style><style fg="#0077c2"></style><style fg="#ansigreen"> {path_str} </style><style fg="#ansigreen" bold="true">❯</style> '
    )


def get_bottom_toolbar():
    return HTML(" <b>ALT+ENTER</b> Multiline | <b>/tools</b> List | <b>/help</b> Help | <b>exit</b> Quit ")


def show_tools(tools):
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Tool")
    table.add_column("Description")
    for t in tools:
        table.add_row(t.name, (t.description[:60] + "...") if t.description else "No description")
    console.print(Panel(table, title="[bold blue]Available Tools[/]"))


def show_help():
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left", style="bold cyan")
    grid.add_column(justify="left")

    grid.add_row("Command", "Description")
    grid.add_row("-------", "-----------")
    grid.add_row("/tools", "List all available tools")
    grid.add_row("/help", "Show this help message")
    grid.add_row("clear", "Clear screen and reset session")
    grid.add_row("exit", "Exit the application")

    grid.add_row("", "")
    grid.add_row("Keyboard Shortcuts", "")
    grid.add_row("------------------", "")
    grid.add_row("Alt+Enter", "Multiline input")
    grid.add_row("Ctrl+C", "Cancel generation")

    console.print(Panel(grid, title="[bold blue]Help & Usage[/]", border_style="blue"))


# ======================================================
# MAIN
# ======================================================

async def main():
    os.system("cls" if os.name == "nt" else "clear")

    # 1. Load Config
    try:
        # Explicitly set CWD as working directory for relative paths
        # This fixes issue where frozen exe uses its own dir as base for everything
        if getattr(sys, "frozen", False):
            os.chdir(os.getcwd())

        temp_cfg = AgentConfig()
        log_level = logging.DEBUG if temp_cfg.debug else logging.WARNING
        setup_logging(level=log_level)

    except Exception as e:
        console.print(f"[bold red]Config Error:[/] {e}")
        return

    # 2. Initialize Workflow
    tool_registry = None
    tools = []
    try:
        with console.status("[bold green]Initializing system...[/]"):
            agent_app, tool_registry = await build_agent_app(temp_cfg)
            tools = tool_registry.tools
    except Exception as e:
        console.print(f"[bold red]Init Error:[/] {e}")
        return

    # Clear and Print Header with Info
    os.system("cls" if os.name == "nt" else "clear")

    model_name = temp_cfg.gemini_model if temp_cfg.provider == "gemini" else temp_cfg.openai_model

    # Modern Header
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="center")
    grid.add_column(justify="right")
    grid.add_row(
        "[bold cyan] > AI Agent[/] [gray]v0.46b[/]",
        f"[gray]Tools: {len(tools)}[/]",
        f"[gray]{model_name}[/] [cyan]•[/]",
    )
    console.print(Panel(grid, style="panel.border", padding=(0, 1)))

    if temp_cfg.debug:
        console.print("[yellow]🐛 Debug mode enabled[/]")

    # 3. Session Setup
    session = PromptSession(
        history=FileHistory(".history"),
        completer=MergeCompleter(
            [WordCompleter(["/help", "/tools", "exit", "clear"]), FuzzyPathCompleter(root_dir=".")]
        ),
        key_bindings=get_key_bindings(),
        lexer=PygmentsLexer(MarkdownLexer),
        auto_suggest=AutoSuggestFromHistory(),
    )

    thread_id = "main_session"
    last_stats = None
    stream_processor_cls = None

    while True:
        try:
            if last_stats:
                console.print(last_stats, justify="right")
                last_stats = None

            user_input = await session.prompt_async(get_prompt_message(), bottom_toolbar=get_bottom_toolbar)
            user_input = user_input.strip()

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
            if user_input.lower() in ["clear", "reset"]:
                thread_id = f"session_{int(time.time())}"
                os.system("cls" if os.name == "nt" else "clear")
                continue

            if user_input.lower() == "/tools":
                show_tools(tools)
                continue

            if user_input.lower() == "/help":
                show_help()
                continue

            # Auto-fix for interrupted tool calls
            repair_session_if_needed(agent_app, thread_id, console)

            # Initialize State & Config
            initial_state = {
                "messages": [("user", user_input)],
                "steps": 0,
                "token_usage": {},
                "current_task": user_input,
                "critic_status": "",
                "critic_source": "",
                "critic_feedback": "",
            }
            # Worst-case graph path now includes critic checks around each potential exit.
            # Use 6x with a small floor for safety against framework recursion checks.
            recursion_limit = max(12, temp_cfg.max_loops * 6)
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}

            # Start Stream
            stream = agent_app.astream(initial_state, config=config, stream_mode=["messages", "updates"])

            # Process Stream
            if stream_processor_cls is None:
                from core.stream_processor import StreamProcessor as _StreamProcessor

                stream_processor_cls = _StreamProcessor

            processor = stream_processor_cls(console)
            last_stats = await processor.process_stream(stream)

        except (KeyboardInterrupt, asyncio.CancelledError):
            continue
        except Exception as e:
            console.print(f"[bold red]{format_exception_friendly(e)}[/]")

    # Final Cleanup
    if tool_registry:
        await tool_registry.cleanup()
    try:
        from tools.system_tools import _net_client

        if _net_client:
            await _net_client.close()
    except ImportError:
        pass


if __name__ == "__main__":
    asyncio.run(main())

