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
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from core.cli_utils import format_exception_friendly, get_key_bindings
from core.config import AgentConfig
from core.constants import AGENT_VERSION, BASE_DIR
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


def get_bottom_toolbar(model_name: str = "", tools_count: int = 0) -> HTML:
    model_part = f" │ <b>{model_name}</b>" if model_name else ""
    tools_part = f" │ tools: {tools_count}" if tools_count else ""
    return HTML(
        f" <b>ALT+ENTER</b> multiline"
        f" | <b>/help</b> · <b>/tools</b> · <b>/session</b>"
        f" | <b>exit</b> quit"
        f"{tools_part}{model_part} "
    )


def render_tools(tools) -> None:
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", expand=False)
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Tool", style="tool.name")
    table.add_column("Description", ratio=1)
    table.add_column("Flags", style="dim", no_wrap=True)
    for i, tool in enumerate(tools, 1):
        desc = tool.description or "No description"
        desc = (desc[:72] + "…") if len(desc) > 72 else desc
        # Detect MCP tools by naming convention (they carry a namespace separator)
        source_flag = "[dim]mcp[/]" if ":" in tool.name or hasattr(tool, "_is_mcp") else ""
        table.add_row(str(i), tool.name, desc, source_flag)
    count_str = f"{len(tools)} tool{'s' if len(tools) != 1 else ''}"
    console.print(Panel(table, title=f"[panel.title]Available Tools[/] [dim]({count_str})[/]", border_style="panel.border"))


def render_help() -> None:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=False)
    table.add_column(justify="right", style="bold cyan", no_wrap=True)
    table.add_column(justify="left", style="dim")

    table.add_row("── Commands ──", "")
    table.add_row("/tools",   "List available tools")
    table.add_row("/session", "Show session · thread · backend")
    table.add_row("/help",    "This message")
    table.add_row("clear",    "Start a fresh session")
    table.add_row("exit",     "Quit")
    table.add_row("", "")
    table.add_row("── Keys ──", "")
    table.add_row("Alt+Enter", "Multiline input")
    table.add_row("Ctrl+C",   "Cancel generation")
    table.add_row("↑↓",         "History navigation")
    console.print(Panel(table, title="[panel.title]Help[/]", border_style="panel.border", padding=(0, 1)))


def render_session_info(snapshot: SessionSnapshot, checkpoint_info: dict) -> None:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=False)
    table.add_column(justify="right", style="dim", no_wrap=True)
    table.add_column(justify="left")

    backend = checkpoint_info.get("resolved_backend", "unknown")
    backend_icon = {"sqlite": "▣", "postgres": "○", "memory": "◦"}.get(backend, "□")

    table.add_row("session",  f"[cyan]{snapshot.session_id[:16]}…[/]")
    table.add_row("thread",   f"[dim]{snapshot.thread_id[:16]}…[/]")
    table.add_row("backend",  f"{backend_icon} [bold]{backend}[/]")
    table.add_row("target",   f"[dim]{checkpoint_info.get('target', 'unknown')}[/]")
    table.add_row("updated",  f"[dim]{_format_timestamp_local(snapshot.updated_at)}[/]")
    console.print(Panel(table, title="[panel.title]Session[/]", border_style="panel.border", padding=(0, 1)))


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
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=False)
    table.add_column(no_wrap=True)
    for line in lines:
        # Color lines that indicate errors/warnings vs. successes
        line_lower = line.lower()
        if any(kw in line_lower for kw in ("error", "failed", "unavailable")):
            table.add_row(f"[status.error]✖[/] [dim]{line}[/]")
        elif any(kw in line_lower for kw in ("warn", "disabled", "skip")):
            table.add_row(f"[status.warning]⚠[/] [dim]{line}[/]")
        else:
            table.add_row(f"[status.success]✔[/] [dim]{line}[/]")
    console.print(Panel(table, title="[panel.title]Runtime[/]", border_style="panel.border", padding=(0, 1)))


def setup_runtime() -> AgentConfig:
    if getattr(sys, "frozen", False):
        os.chdir(os.getcwd())

    config = AgentConfig()
    log_level = logging.DEBUG if config.debug else logging.WARNING
    setup_logging(level=log_level)
    return config


async def initialize_agent(config: AgentConfig):
    provider_label = config.provider.capitalize()
    model_name = config.gemini_model if config.provider == "gemini" else config.openai_model
    with console.status(
        f"[init.step]▶[/] [dim]Loading {provider_label} · {model_name}...[/]",
        spinner="dots",
    ):
        result = await build_agent_app(config)
    console.print(f"[init.step]✔[/] [dim]Agent ready[/]")
    return result


def render_header(config: AgentConfig, tools) -> None:
    clear_screen()
    model_name = config.gemini_model if config.provider == "gemini" else config.openai_model
    provider_icon = "[cyan]◆[/]" if config.provider == "gemini" else "[green]◆[/]"
    tools_str = f"[dim]{len(tools)} tools[/]" if tools else "[dim]no tools[/]"
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="center")
    grid.add_column(justify="right")
    grid.add_row(
        f"[bold cyan]AI Agent[/] [dim]v{AGENT_VERSION}[/]",
        tools_str,
        f"[dim]{model_name}[/] {provider_icon}",
    )
    console.print(Panel(grid, style="panel.border", padding=(0, 1)))
    if config.debug:
        console.print(Panel("[yellow]Debug mode active[/]", border_style="panel.warning", padding=(0, 1)))


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


def _format_policy_flags(policy: dict) -> str:
    """Returns colored Rich markup string for tool policy flags."""
    parts = []
    if policy.get("destructive"):
        parts.append("[approval.danger]destructive[/]")
    if policy.get("mutating"):
        parts.append("[approval.mutating]mutating[/]")
    if policy.get("networked"):
        parts.append("[approval.networked]networked[/]")
    return "  ".join(parts) if parts else "[dim]protected[/]"


async def prompt_for_interrupt(session: PromptSession, interrupt_payload: dict) -> dict:
    if interrupt_payload.get("kind") != "tool_approval":
        return {"approved": False}

    req_tools = interrupt_payload.get("tools", [])
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", expand=False)
    table.add_column("Tool", style="tool.name", no_wrap=True)
    table.add_column("Args", style="tool.args")
    table.add_column("Flags", no_wrap=True)
    for tool in req_tools:
        policy = tool.get("policy") or {}
        args_str = str(tool.get("args", {}))
        args_str = (args_str[:80] + "…") if len(args_str) > 80 else args_str
        table.add_row(
            tool.get("name", "unknown_tool"),
            args_str,
            _format_policy_flags(policy),
        )
    console.print(
        Panel(
            table,
            title="[approval.border]⚠  Approval Required[/]",
            border_style="approval.border",
            padding=(0, 1),
        )
    )
    _valid = {"", "y", "yes", "n", "no"}
    while True:
        answer = (
            await session.prompt_async(
                HTML('<style fg="#e0af68"> Approve? [Y/n] </style><style fg="#565f89">(enter = approve) </style><style fg="#e0af68">❯</style> ')
            )
        ).strip().lower()
        if answer in _valid:
            break
        console.print(f"  [status.warning]⚠[/] [dim]Enter y / n / or press Enter[/]")

    approved = answer in {"", "y", "yes"}
    status = "[status.success]approved[/]" if approved else "[status.error]denied[/]"
    console.print(f"  └ {status}")
    return {"approved": approved}


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
        console.print(
            Panel(f"[status.error]Config error:[/] {e}", border_style="panel.error", padding=(0, 1))
        )
        return

    store = SessionStore(config.session_state_path)

    tool_registry = None
    try:
        agent_app, tool_registry = await initialize_agent(config)
    except Exception as e:
        console.print(
            Panel(f"[status.error]Init error:[/] {e}", border_style="panel.error", padding=(0, 1))
        )
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

    model_name = config.gemini_model if config.provider == "gemini" else config.openai_model

    def _toolbar():
        return get_bottom_toolbar(model_name=model_name, tools_count=len(tools))

    turn_count = 0

    while True:
        try:
            if last_stats:
                console.print(last_stats, justify="right")
                last_stats = None

            user_input = (await session.prompt_async(get_prompt_message(), bottom_toolbar=_toolbar)).strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
            if try_handle_command(user_input, tools, reset_session, show_session):
                continue

            # Print a subtle turn separator after the first message
            if turn_count > 0:
                console.print(Rule(style="turn.separator"))
            turn_count += 1

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
            console.print("[dim]  ✕ cancelled[/]")
            continue
        except Exception as e:
            friendly = format_exception_friendly(e)
            console.print(
                Panel(
                    f"[status.error]{friendly}[/]",
                    border_style="panel.error",
                    padding=(0, 1),
                )
            )

    await close_runtime_resources(tool_registry)


if __name__ == "__main__":
    asyncio.run(main())
