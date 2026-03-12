import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from langgraph.types import Command
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
from core.session_store import SessionSnapshot, SessionStore
from core.session_utils import repair_session_if_needed
from core.ui_theme import AGENT_THEME

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent import build_agent_app

console = Console(theme=AGENT_THEME, color_system="truecolor")
logging.getLogger("httpx").setLevel(logging.WARNING)

COMMANDS = ["/help", "/tools", "/session", "exit", "clear", "reset"]
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
    return HTML(" <b>ALT+ENTER</b> Multiline | <b>/tools</b> List | <b>/session</b> Session | <b>exit</b> Quit ")


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
    grid.add_row("/session", "Show active session and persistence backend")
    grid.add_row("/help", "Show this help message")
    grid.add_row("clear/reset", "Create a new session")
    grid.add_row("exit", "Exit the application")
    grid.add_row("", "")
    grid.add_row("Keyboard Shortcuts", "")
    grid.add_row("------------------", "")
    grid.add_row("Alt+Enter", "Multiline input")
    grid.add_row("Ctrl+C", "Cancel generation")
    console.print(Panel(grid, title="[bold blue]Help & Usage[/]", border_style="blue"))


def render_session_info(snapshot: SessionSnapshot, checkpoint_info: dict) -> None:
    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Session ID", snapshot.session_id)
    table.add_row("Thread ID", snapshot.thread_id)
    table.add_row("Checkpoint", f"{checkpoint_info.get('resolved_backend', 'unknown')}")
    table.add_row("Target", checkpoint_info.get("target", "unknown"))
    table.add_row("Updated", _format_timestamp_local(snapshot.updated_at))
    console.print(Panel(table, title="[bold blue]Session[/]"))


def _format_timestamp_local(value: str) -> str:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def render_runtime_status(tool_registry) -> None:
    lines = tool_registry.get_runtime_status_lines()
    if not lines:
        return
    body = "\n".join(f"- {line}" for line in lines)
    console.print(Panel(body, title="[bold blue]Runtime Status[/]", border_style="blue"))


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
        "[bold cyan] > AI Agent[/] [gray]v0.5b[/]",
        f"[gray]Tools: {len(tools)}[/]",
        f"[gray]{model_name}[/] [cyan]•[/]",
    )
    console.print(Panel(grid, style="panel.border", padding=(0, 1)))
    if config.debug:
        console.print("[yellow]Debug mode enabled[/]")


def create_session() -> PromptSession:
    return PromptSession(
        history=FileHistory(".history"),
        completer=MergeCompleter([WordCompleter(COMMANDS), FuzzyPathCompleter(root_dir=".")]),
        key_bindings=get_key_bindings(),
        lexer=PygmentsLexer(MarkdownLexer),
        auto_suggest=AutoSuggestFromHistory(),
    )


def build_initial_state(user_input: str, session_id: str, safety_mode: str = "default") -> dict:
    return {
        "messages": [("user", user_input)],
        "steps": 0,
        "token_usage": {},
        "current_task": user_input,
        "critic_status": "",
        "critic_source": "",
        "critic_feedback": "",
        "session_id": session_id,
        "run_id": uuid.uuid4().hex,
        "pending_approval": None,
        "last_tool_error": "",
        "last_tool_result": "",
        "safety_mode": safety_mode,
    }


def build_graph_config(thread_id: str, max_loops: int) -> dict:
    recursion_limit = max(12, max_loops * 6)
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


def try_handle_command(
    user_input: str,
    tools,
    reset_session: Callable[[], None],
    show_session: Callable[[], None],
) -> bool:
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
    if normalized == "/session":
        show_session()
        return True
    return False


async def prompt_for_interrupt(session: PromptSession, interrupt_payload: dict) -> dict:
    if interrupt_payload.get("kind") != "tool_approval":
        return {"approved": False}

    tools = interrupt_payload.get("tools", [])
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Tool")
    table.add_column("Args")
    table.add_column("Policy")
    for tool in tools:
        policy = tool.get("policy") or {}
        flags = []
        if policy.get("mutating"):
            flags.append("mutating")
        if policy.get("destructive"):
            flags.append("destructive")
        if policy.get("networked"):
            flags.append("networked")
        table.add_row(
            tool.get("name", "unknown_tool"),
            str(tool.get("args", {})),
            ", ".join(flags) or "protected",
        )
    console.print(Panel(table, title="[bold yellow]Approval Required[/]", border_style="yellow"))
    answer = (
        await session.prompt_async(
            HTML('<style fg="#ansiyellow">Approve tool execution? [y/N]: </style>')
        )
    ).strip().lower()
    return {"approved": answer in {"y", "yes"}}


async def run_user_request(
    agent_app,
    thread_id: str,
    session_id: str,
    user_input: str,
    config: AgentConfig,
    stream_processor_cls,
    prompt_session: PromptSession,
):
    from core.stream_processor import StreamProcessResult

    await repair_session_if_needed(agent_app, thread_id, console)
    payload: dict | Command = build_initial_state(user_input, session_id=session_id)

    while True:
        stream = agent_app.astream(
            payload,
            config=build_graph_config(thread_id, config.max_loops),
            stream_mode=["messages", "updates"],
        )
        processor = stream_processor_cls(console)
        result: StreamProcessResult = await processor.process_stream(stream)
        if result.interrupt is None:
            return result.stats
        payload = Command(resume=await prompt_for_interrupt(prompt_session, result.interrupt))


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

    store = SessionStore(config.session_state_path)

    tool_registry = None
    try:
        agent_app, tool_registry = await initialize_agent(config)
    except Exception as e:
        console.print(f"[bold red]Init Error:[/] {e}")
        return

    checkpoint_info = tool_registry.checkpoint_info
    current_session = store.load_active_session() or store.new_session(
        checkpoint_backend=checkpoint_info.get("resolved_backend", config.checkpoint_backend),
        checkpoint_target=checkpoint_info.get("target", "unknown"),
    )
    store.save_active_session(current_session)

    tools = tool_registry.tools
    render_header(config, tools)
    render_runtime_status(tool_registry)
    render_session_info(current_session, checkpoint_info)
    session = create_session()

    from core.stream_processor import StreamProcessor

    last_stats = None

    def reset_session() -> None:
        nonlocal current_session
        current_session = store.new_session(
            checkpoint_backend=checkpoint_info.get("resolved_backend", config.checkpoint_backend),
            checkpoint_target=checkpoint_info.get("target", "unknown"),
        )
        store.save_active_session(current_session)
        render_header(config, tools)
        render_session_info(current_session, checkpoint_info)

    def show_session() -> None:
        render_session_info(current_session, checkpoint_info)

    while True:
        try:
            if last_stats:
                console.print(last_stats, justify="right")
                last_stats = None

            user_input = (await session.prompt_async(get_prompt_message(), bottom_toolbar=get_bottom_toolbar())).strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
            if try_handle_command(user_input, tools, reset_session, show_session):
                continue

            last_stats = await run_user_request(
                agent_app,
                current_session.thread_id,
                current_session.session_id,
                user_input,
                config,
                StreamProcessor,
                session,
            )
            store.save_active_session(current_session)
        except (KeyboardInterrupt, asyncio.CancelledError):
            continue
        except Exception as e:
            console.print(f"[bold red]{format_exception_friendly(e)}[/]")

    await close_runtime_resources(tool_registry)


if __name__ == "__main__":
    asyncio.run(main())
