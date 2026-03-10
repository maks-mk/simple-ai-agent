import asyncio
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.cli_utils import format_exception_friendly, get_key_bindings
from core.config import AgentConfig
from core.constants import BASE_DIR
from core.fuzzy_completer import FuzzyPathCompleter
from core.logging_config import setup_logging
from core.session_utils import repair_session_if_needed
from core.ui_theme import AGENT_THEME

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent import build_agent_app

warnings.filterwarnings("ignore")
console = Console(theme=AGENT_THEME, color_system="truecolor")
logging.getLogger("httpx").setLevel(logging.WARNING)

COMMANDS = ["/help", "/tools", "exit", "clear"]
CLEAR_COMMAND = "cls" if os.name == "nt" else "clear"


class MergeCompleter(Completer):
    def __init__(self, completers):
        self.completers = completers

    def get_completions(self, document, complete_event):
        for completer in self.completers:
            yield from completer.get_completions(document, complete_event)


def clear_screen() -> None:
    os.system(CLEAR_COMMAND)


def get_prompt_message() -> HTML:
    cwd = Path.cwd()
    home = Path.home()
    try:
        parts = ("~",) + cwd.relative_to(home).parts
    except ValueError:
        parts = cwd.parts

    display_parts = [parts[0], "\u2026", *parts[-2:]] if len(parts) > 4 else list(parts)
    path_str = "/".join(display_parts).replace("\\", "/")
    return HTML(
        f'<style bg="#0077c2" fg="white"> Agent </style><style fg="#0077c2"></style><style fg="#ansigreen"> {path_str} </style><style fg="#ansigreen" bold="true">❯</style> '
    )


def get_bottom_toolbar() -> HTML:
    return HTML(" <b>ALT+ENTER</b> Multiline | <b>/tools</b> List | <b>/help</b> Help | <b>exit</b> Quit ")


def render_tools(tools) -> None:
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Tool")
    table.add_column("Description")
    for tool in tools:
        table.add_row(tool.name, (tool.description[:60] + "...") if tool.description else "No description")
    console.print(Panel(table, title="[bold blue]Available Tools[/]"))


def render_help() -> None:
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


def setup_runtime() -> AgentConfig:
    if getattr(sys, "frozen", False):
        os.chdir(os.getcwd())

    config = AgentConfig()
    log_level = logging.DEBUG if config.debug else logging.WARNING
    setup_logging(level=log_level)
    return config


async def initialize_agent(config: AgentConfig):
    with console.status("[bold green]Initializing system...[/]"):
        return await build_agent_app(config)


def render_header(config: AgentConfig, tools) -> None:
    clear_screen()
    model_name = config.gemini_model if config.provider == "gemini" else config.openai_model
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="center")
    grid.add_column(justify="right")
    grid.add_row(
        "[bold cyan] > AI Agent[/] [gray]v0.47b[/]",
        f"[gray]Tools: {len(tools)}[/]",
        f"[gray]{model_name}[/] [cyan]•[/]",
    )
    console.print(Panel(grid, style="panel.border", padding=(0, 1)))
    if config.debug:
        console.print("[yellow]🐛 Debug mode enabled[/]")


def create_session() -> PromptSession:
    return PromptSession(
        history=FileHistory(".history"),
        completer=MergeCompleter([WordCompleter(COMMANDS), FuzzyPathCompleter(root_dir=".")]),
        key_bindings=get_key_bindings(),
        lexer=PygmentsLexer(MarkdownLexer),
        auto_suggest=AutoSuggestFromHistory(),
    )


def build_initial_state(user_input: str) -> dict:
    return {
        "messages": [("user", user_input)],
        "steps": 0,
        "token_usage": {},
        "current_task": user_input,
        "critic_status": "",
        "critic_source": "",
        "critic_feedback": "",
    }


def build_graph_config(thread_id: str, max_loops: int) -> dict:
    recursion_limit = max(12, max_loops * 6)
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


def try_handle_command(user_input: str, tools, reset_session: Callable[[], None]) -> bool:
    normalized = user_input.lower()
    if normalized in ["clear", "reset"]:
        reset_session()
        clear_screen()
        return True
    if normalized == "/tools":
        render_tools(tools)
        return True
    if normalized == "/help":
        render_help()
        return True
    return False


async def run_user_request(agent_app, thread_id: str, user_input: str, config: AgentConfig, stream_processor_cls):
    repair_session_if_needed(agent_app, thread_id, console)
    stream = agent_app.astream(
        build_initial_state(user_input),
        config=build_graph_config(thread_id, config.max_loops),
        stream_mode=["messages", "updates"],
    )
    processor = stream_processor_cls(console)
    return await processor.process_stream(stream)


async def close_runtime_resources(tool_registry) -> None:
    if tool_registry:
        await tool_registry.cleanup()
    try:
        from tools.system_tools import _net_client

        if _net_client:
            await _net_client.aclose()
    except ImportError:
        pass


async def main():
    clear_screen()

    try:
        config = setup_runtime()
    except Exception as e:
        console.print(f"[bold red]Config Error:[/] {e}")
        return

    tool_registry = None
    try:
        agent_app, tool_registry = await initialize_agent(config)
    except Exception as e:
        console.print(f"[bold red]Init Error:[/] {e}")
        return

    tools = tool_registry.tools
    render_header(config, tools)
    session = create_session()

    from core.stream_processor import StreamProcessor

    thread_id = "main_session"
    last_stats = None

    def reset_session() -> None:
        nonlocal thread_id
        thread_id = f"session_{int(time.time())}"

    while True:
        try:
            if last_stats:
                console.print(last_stats, justify="right")
                last_stats = None

            user_input = (await session.prompt_async(get_prompt_message(), bottom_toolbar=get_bottom_toolbar)).strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
            if try_handle_command(user_input, tools, reset_session):
                continue

            last_stats = await run_user_request(agent_app, thread_id, user_input, config, StreamProcessor)
        except (KeyboardInterrupt, asyncio.CancelledError):
            continue
        except Exception as e:
            console.print(f"[bold red]{format_exception_friendly(e)}[/]")

    await close_runtime_resources(tool_registry)


if __name__ == "__main__":
    asyncio.run(main())

