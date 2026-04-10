import asyncio
from html import escape
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langgraph.types import Command
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, WordCompleter
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts.choice_input import ChoiceInput
    from prompt_toolkit.styles import Style as PromptStyle
except ImportError:  # pragma: no cover - exercised in import-only environments
    PromptSession = Any
    AutoSuggestFromHistory = None
    HTML = lambda value: value
    KeyBindings = Any
    PromptStyle = None

    class Completer:  # type: ignore[override]
        pass

    class WordCompleter:  # type: ignore[override]
        def __init__(self, *_args, **_kwargs):
            pass

    class FileHistory:  # type: ignore[override]
        def __init__(self, *_args, **_kwargs):
            pass

    class ChoiceInput:  # type: ignore[override]
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("prompt_toolkit is required for interactive approval prompts")

from core.cli_utils import format_exception_friendly, get_key_bindings
from core.config import AgentConfig
from core.constants import AGENT_VERSION, BASE_DIR
from core.logging_config import setup_logging
from core.session_store import SessionSnapshot, SessionStore
from core.session_utils import repair_session_if_needed
from core.tool_policy import ToolMetadata
from core.ui_theme import ACCENT_BLUE, AGENT_THEME, TEXT_MUTED, TEXT_PRIMARY

try:
    from core.fuzzy_completer import FuzzyPathCompleter
except ImportError:  # pragma: no cover - import fallback for non-interactive test envs
    class FuzzyPathCompleter:  # type: ignore[override]
        def __init__(self, *_args, **_kwargs):
            pass

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent import build_agent_app

console = Console(theme=AGENT_THEME, color_system="truecolor")
logging.getLogger("httpx").setLevel(logging.WARNING)

COMMANDS = ["/help", "/tools", "/session", "/new", "/quit", "clear", "reset"]
CLEAR_COMMAND = "cls" if os.name == "nt" else "clear"
APPROVAL_MODE_PROMPT = "prompt"
APPROVAL_MODE_ALWAYS = "always"
PROMPT_TOOLKIT_STYLE = (
    PromptStyle.from_dict(
        {
            "bottom-toolbar": f"fg:{TEXT_MUTED} bg:#101216 noreverse",
        }
    )
    if PromptStyle is not None
    else None
)
APPROVAL_CHOICE_STYLE = (
    PromptStyle.from_dict(
        {
            "input-selection": f"fg:{TEXT_MUTED}",
            "option": f"fg:{TEXT_MUTED}",
            "number": f"fg:{TEXT_MUTED}",
            "selected-option": f"fg:{TEXT_PRIMARY} bold",
            "bottom-toolbar": f"fg:{TEXT_MUTED} bg:#101216 noreverse",
        }
    )
    if PromptStyle is not None
    else None
)


class _ErasingChoiceInput(ChoiceInput):
    """ChoiceInput without numeric prefixes; clears menu after Enter."""

    def _create_application(self):
        try:
            from prompt_toolkit.application import Application
            from prompt_toolkit.filters import (
                Condition,
                is_done,
                renderer_height_is_known,
                to_filter,
            )
            from prompt_toolkit.key_binding import DynamicKeyBindings, KeyBindings, merge_key_bindings
            from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
            from prompt_toolkit.layout.dimension import Dimension
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.utils import suspend_to_background_supported
            from prompt_toolkit.widgets import Box, Frame, Label, RadioList
        except Exception:
            app = super()._create_application()
            app.erase_when_done = True
            return app

        radio_list = RadioList(
            values=self.options,
            default=self.default,
            select_on_focus=True,
            open_character="",
            select_character=self.symbol,
            close_character="",
            show_cursor=False,
            show_numbers=False,
            container_style="class:input-selection",
            default_style="class:option",
            selected_style="",
            checked_style="class:selected-option",
            number_style="class:number",
            show_scrollbar=False,
        )
        container = HSplit(
            [
                Box(
                    Label(text=self.message, dont_extend_height=True),
                    padding_top=0,
                    padding_left=1,
                    padding_right=1,
                    padding_bottom=0,
                ),
                Box(
                    radio_list,
                    padding_top=0,
                    padding_left=3,
                    padding_right=1,
                    padding_bottom=0,
                ),
            ]
        )

        @Condition
        def show_frame_filter() -> bool:
            return to_filter(self.show_frame)()

        show_bottom_toolbar = (
            Condition(lambda: self.bottom_toolbar is not None)
            & ~is_done
            & renderer_height_is_known
        )

        container = ConditionalContainer(
            Frame(container),
            alternative_content=container,
            filter=show_frame_filter,
        )

        bottom_toolbar = ConditionalContainer(
            Window(
                FormattedTextControl(lambda: self.bottom_toolbar, style="class:bottom-toolbar.text"),
                style="class:bottom-toolbar",
                dont_extend_height=True,
                height=Dimension(min=1),
            ),
            filter=show_bottom_toolbar,
        )

        layout = Layout(
            HSplit(
                [
                    container,
                    ConditionalContainer(Window(), filter=show_bottom_toolbar),
                    bottom_toolbar,
                ]
            ),
            focused_element=radio_list,
        )

        kb = KeyBindings()

        @kb.add("enter", eager=True)
        def _accept_input(event):
            event.app.exit(result=radio_list.current_value, style="class:accepted")

        @Condition
        def enable_interrupt() -> bool:
            return to_filter(self.enable_interrupt)()

        @kb.add("c-c", filter=enable_interrupt)
        @kb.add("<sigint>", filter=enable_interrupt)
        def _keyboard_interrupt(event):
            event.app.exit(exception=self.interrupt_exception(), style="class:aborting")

        suspend_supported = Condition(suspend_to_background_supported)

        @Condition
        def enable_suspend() -> bool:
            return to_filter(self.enable_suspend)()

        @kb.add("c-z", filter=suspend_supported & enable_suspend)
        def _suspend(event):
            event.app.suspend_to_background()

        return Application(
            layout=layout,
            full_screen=False,
            mouse_support=self.mouse_support,
            key_bindings=merge_key_bindings([kb, DynamicKeyBindings(lambda: self.key_bindings)]),
            style=self.style,
            erase_when_done=True,
        )


@dataclass(frozen=True)
class ApprovalSummary:
    destructive_count: int
    mutating_count: int
    networked_count: int
    default_approve: bool
    risk_level: str
    impacts: tuple[str, ...]

    @property
    def default_prompt(self) -> str:
        return "[Y/n]" if self.default_approve else "[y/N]"

    @property
    def default_hint(self) -> str:
        return "Enter = approve" if self.default_approve else "Enter = deny"

    @property
    def impact_text(self) -> str:
        return ", ".join(self.impacts) if self.impacts else "local state"


class MergeCompleter(Completer):
    def __init__(self, completers):
        self.completers = completers

    def get_completions(self, document, complete_event):
        for completer in self.completers:
            yield from completer.get_completions(document, complete_event)


def clear_screen() -> None:
    os.system(CLEAR_COMMAND)


def _resolve_console(out: Console | None) -> Console:
    return out or console


def _error_panel(message: str) -> Panel:
    return Panel(Text(message, style="status.error"), border_style="panel.error", padding=(0, 1))


def _html_style(text: str, *, fg: str | None = None, bg: str | None = None, bold: bool = False) -> str:
    attrs = []
    if fg:
        attrs.append(f'fg="{fg}"')
    if bg:
        attrs.append(f'bg="{bg}"')
    if bold:
        attrs.append('bold="true"')
    attr_text = " ".join(attrs)
    safe_text = escape(text)
    if not attr_text:
        return safe_text
    return f"<style {attr_text}>{safe_text}</style>"


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
        f"{_html_style(' Agent ', fg=TEXT_PRIMARY, bg=ACCENT_BLUE, bold=True)}"
        f"{_html_style('', fg=ACCENT_BLUE)}"
        f"{_html_style(f' {path_str} ', fg=TEXT_MUTED)}"
        f"{_html_style('❯', fg=ACCENT_BLUE, bold=True)} "
    )


def get_bottom_toolbar(mode: str = "normal", model_name: str = "", tools_count: int = 0) -> HTML:
    if mode == "approval":
        return HTML(_html_style(" Approval  |  Up/Down select  |  Enter confirm  |  Esc cancel ", fg=TEXT_MUTED))

    parts = [
        "Input Alt+Enter multiline",
        "Commands /help /tools /session /new /quit",
    ]
    status_parts: list[str] = []
    if model_name:
        status_parts.append(model_name)
    if status_parts:
        parts.append("Status " + " · ".join(status_parts))
    return HTML(_html_style(f" {'  |  '.join(parts)} ", fg=TEXT_MUTED))


def _provider_model(config: AgentConfig) -> tuple[str, str]:
    if config.provider == "gemini":
        return "Gemini", config.gemini_model
    return "OpenAI", config.openai_model


def _short_id(value: str, length: int = 16) -> str:
    if len(value) <= length:
        return value
    return f"{value[:length]}…"


def _tool_is_mcp(tool, metadata: ToolMetadata | None) -> bool:
    return bool((metadata and metadata.source == "mcp") or hasattr(tool, "_is_mcp") or ":" in tool.name)


def _format_tool_badges(metadata: ToolMetadata | None, *, is_mcp: bool) -> str:
    metadata = metadata or ToolMetadata(name="unknown", read_only=True)
    parts: list[str] = []
    if is_mcp:
        parts.append("[tool.mcp]mcp[/]")
    if metadata.read_only and not metadata.mutating and not metadata.destructive:
        parts.append("[tool.readonly]read-only[/]")
    if metadata.requires_approval:
        parts.append("[approval.border]approval[/]")
    if metadata.mutating:
        parts.append("[approval.mutating]mutating[/]")
    if metadata.destructive:
        parts.append("[approval.danger]destructive[/]")
    if metadata.networked:
        parts.append("[approval.networked]network[/]")
    return "  ".join(parts) if parts else "[dim]protected[/]"


def _tool_group(tool, metadata: ToolMetadata | None) -> str:
    if _tool_is_mcp(tool, metadata):
        return "MCP"
    if metadata and (metadata.mutating or metadata.destructive or metadata.requires_approval):
        return "Protected"
    return "Read-only"


def _enabled_mcp_servers(tool_registry) -> list[str]:
    names = []
    for status in getattr(tool_registry, "mcp_server_status", []):
        server = status.get("server", "unknown")
        if status.get("error"):
            names.append(f"{server} (error)")
        else:
            names.append(server)
    return names


def _runtime_issue_count(tool_registry) -> int:
    lines = tool_registry.get_runtime_status_lines()
    return sum(
        1
        for line in lines
        if any(keyword in line.lower() for keyword in ("error", "warning", "failed", "unavailable"))
    )


def render_overview(
    config: AgentConfig,
    tool_registry,
    snapshot: SessionSnapshot,
    *,
    show_cheatsheet: bool = False,
    out: Console | None = None,
) -> None:
    out = _resolve_console(out)
    provider_label, model_name = _provider_model(config)
    checkpoint_info = getattr(tool_registry, "checkpoint_info", {}) or {}
    backend = checkpoint_info.get("resolved_backend", config.checkpoint_backend)
    tools_count = len(getattr(tool_registry, "tools", []))
    approvals = "off"
    if config.enable_approvals:
        approvals = "on"
        if getattr(snapshot, "approval_mode", APPROVAL_MODE_PROMPT) == APPROVAL_MODE_ALWAYS:
            approvals = "on (always for this session)"
    mcp_servers = _enabled_mcp_servers(tool_registry)
    mcp_text = ", ".join(mcp_servers) if mcp_servers else "none"
    issue_count = _runtime_issue_count(tool_registry)
    status_text = (
        "ready"
        if issue_count == 0
        else f"degraded ({issue_count} issue{'s' if issue_count != 1 else ''})"
    )

    summary = Text()
    summary.append(provider_label, style="overview.value")
    summary.append("  ·  ", style="dim")
    summary.append(model_name, style="overview.value")
    summary.append("  ·  ", style="dim")
    summary.append(f"tools {tools_count}", style="overview.value")
    summary.append("  ·  ", style="dim")
    summary.append(backend, style="overview.value")
    summary.append("  ·  ", style="dim")
    summary.append(f"approvals {approvals}", style="overview.value")
    if mcp_text != "none":
        summary.append("  ·  ", style="dim")
        summary.append(f"mcp {mcp_text}", style="overview.value")
    summary.append("  ·  ", style="dim")
    summary.append(status_text, style="status.spinner" if issue_count == 0 else "status.error")

    details = None
    if not show_cheatsheet:
        details = Text()
        details.append(f"session {_short_id(snapshot.session_id)}", style="dim")
        details.append("  ·  ", style="dim")
        details.append(f"thread {_short_id(snapshot.thread_id)}", style="dim")

    body = Group(summary, details) if details else summary
    out.print(
        Panel(
            body,
            title=f"[panel.title]AI Agent[/] [gold3]v{AGENT_VERSION}[/]",
            border_style="panel.border",
            padding=(0, 1),
        )
    )

def render_tools(tool_registry, out: Console | None = None) -> None:
    out = _resolve_console(out)
    tools = getattr(tool_registry, "tools", [])
    metadata_map = getattr(tool_registry, "tool_metadata", {})
    grouped = {"Read-only": [], "Protected": [], "MCP": []}

    for tool in tools:
        metadata = metadata_map.get(tool.name)
        grouped[_tool_group(tool, metadata)].append((tool, metadata))

    table = Table(box=box.ROUNDED, show_header=True, header_style="table.header", expand=False)
    table.add_column("Group", style="overview.label", no_wrap=True)
    table.add_column("Tool", style="tool.name", no_wrap=True)
    table.add_column("Description", ratio=1)
    table.add_column("Flags", style="dim", no_wrap=True)

    order = ("Read-only", "Protected", "MCP")
    added_rows = 0
    for group_name in order:
        items = sorted(grouped[group_name], key=lambda item: item[0].name)
        if not items:
            continue
        if added_rows:
            table.add_section()
        for tool, metadata in items:
            desc = tool.description or "No description"
            desc = (desc[:72] + "…") if len(desc) > 72 else desc
            table.add_row(
                group_name,
                tool.name,
                desc,
                _format_tool_badges(metadata, is_mcp=_tool_is_mcp(tool, metadata)),
            )
            added_rows += 1

    count_str = f"{len(tools)} tool{'s' if len(tools) != 1 else ''}"
    out.print(
        Panel(
            table,
            title=f"[panel.title]Tools[/] [dim]({count_str})[/]",
            border_style="panel.border",
            padding=(0, 1),
        )
    )


def render_help(out: Console | None = None) -> None:
    out = _resolve_console(out)
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=False)
    table.add_column(justify="right", style="overview.label", no_wrap=True)
    table.add_column(justify="left", style="dim")

    table.add_row("Start work", "Type a request and press Enter")
    table.add_row("Inspect tools", "/tools shows read-only, protected, and MCP tools")
    table.add_row("Inspect session", "/session shows the current runtime overview")
    table.add_row("Reset", "/new starts a fresh session  ·  clear/reset still work")
    table.add_row("Exit", "/quit closes the CLI")
    table.add_row("", "")
    table.add_row("Input", "Alt+Enter adds a new line  ·  ↑↓ reuses history")
    table.add_row("Approvals", "↑↓ choose  ·  Enter confirms  ·  Esc cancels")
    table.add_row("Stop", "Ctrl+C cancels the current generation")
    out.print(Panel(table, title="[panel.title]Help[/]", border_style="panel.border", padding=(0, 1)))


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
        f"[init.step]●[/] [init.info]Loading {provider_label} · {model_name}...[/]",
        spinner="dots",
    ):
        result = await build_agent_app(config)
    console.print(f"[init.step]●[/] [status.success]Agent ready[/]")
    return result


def create_session() -> PromptSession:
    return PromptSession(
        history=FileHistory(".history"),
        completer=MergeCompleter([WordCompleter(COMMANDS), FuzzyPathCompleter(root_dir=".")]),
        key_bindings=get_key_bindings(),
        auto_suggest=AutoSuggestFromHistory(),
        style=PROMPT_TOOLKIT_STYLE,
    )


def build_initial_state(user_input: str, session_id: str, safety_mode: str = "default") -> dict:
    return {
        "messages": [("user", user_input)],
        "steps": 0,
        "token_usage": {},
        "current_task": user_input,
        "retry_count": 0,
        "retry_reason": "",
        "turn_outcome": "",
        "final_issue": "",
        "session_id": session_id,
        "run_id": uuid.uuid4().hex,
        "turn_id": 1,
        "pending_approval": None,
        "open_tool_issue": None,
        "last_tool_error": "",
        "last_tool_result": "",
        "safety_mode": safety_mode,
    }


def build_graph_config(thread_id: str, max_loops: int) -> dict:
    recursion_limit = max(12, max_loops * 6)
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


def try_handle_command(
    user_input: str,
    tool_registry,
    reset_session: Callable[[], None],
    show_session: Callable[[], None],
) -> bool:
    normalized = user_input.lower()
    if normalized in ["clear", "reset", "/new"]:
        reset_session()
        return True
    if normalized == "/tools":
        render_tools(tool_registry)
        return True
    if normalized == "/help":
        render_help()
        return True
    if normalized == "/session":
        show_session()
        return True
    return False


def _summarize_approval_request(req_tools: list[dict]) -> ApprovalSummary:
    destructive_count = 0
    mutating_count = 0
    networked_count = 0
    impacts: set[str] = set()

    for tool in req_tools:
        policy = tool.get("policy") or {}
        destructive = bool(policy.get("destructive"))
        mutating = bool(policy.get("mutating"))
        networked = bool(policy.get("networked"))
        impact_scope = str(policy.get("impact_scope") or "unknown").strip() or "unknown"

        destructive_count += int(destructive)
        mutating_count += int(mutating)
        networked_count += int(networked)

        if destructive or mutating or impact_scope != "unknown":
            impacts.add(impact_scope.replace("_", " "))
        if networked:
            impacts.add("network")

    risk_kinds = sum(int(count > 0) for count in (destructive_count, mutating_count, networked_count))
    mixed_risk = risk_kinds > 1
    default_approve = destructive_count == 0 and not mixed_risk
    risk_level = "high" if destructive_count or mixed_risk else "medium" if (mutating_count or networked_count) else "low"

    return ApprovalSummary(
        destructive_count=destructive_count,
        mutating_count=mutating_count,
        networked_count=networked_count,
        default_approve=default_approve,
        risk_level=risk_level,
        impacts=tuple(sorted(impacts)),
    )


def _normalize_approval_mode(value: str | None) -> str:
    if value == APPROVAL_MODE_ALWAYS:
        return APPROVAL_MODE_ALWAYS
    return APPROVAL_MODE_PROMPT


def _approval_target_text(tool: dict) -> str:
    args = tool.get("args") or {}
    if not isinstance(args, dict):
        return ""

    for key in ("path", "target", "destination", "source", "cwd", "url", "pid"):
        value = args.get(key)
        if value:
            text = str(value)
            return (text[:72] + "…") if len(text) > 72 else text
    return ""


def _approval_panel_body(req_tools: list[dict], _summary: ApprovalSummary) -> Group | str:
    if not req_tools:
        return "[approval.summary]Action pending approval[/]"

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="table.header",
        expand=True,
        pad_edge=False,
    )
    table.add_column("Action", style="tool.name", no_wrap=True)
    table.add_column("Target", style="overview.value")

    for tool in req_tools:
        tool_name = tool.get("name", "unknown_tool")
        target = _approval_target_text(tool)
        table.add_row(tool_name, target or "local action")
    return table


def _approval_panel_style(summary: ApprovalSummary) -> str:
    if summary.risk_level == "low":
        return "approval.networked"
    return "approval.border"


async def _run_approval_selector(summary: ApprovalSummary) -> str | None:
    default_choice = "yes" if summary.default_approve else "no"
    kb = KeyBindings()

    @kb.add("escape")
    def _cancel(event) -> None:
        event.app.exit(result=None)

    @kb.add("c-c")
    def _cancel_interrupt(event) -> None:
        event.app.exit(result=None)

    prompt = _ErasingChoiceInput(
        message="↳ approve:",
        options=[
            ("yes", "yes          — once"),
            ("no", "no           — deny"),
            ("always", "always       — this session"),
        ],
        default=default_choice,
        mouse_support=True,
        symbol="›",
        style=APPROVAL_CHOICE_STYLE,
        bottom_toolbar=lambda: get_bottom_toolbar(mode="approval"),
        show_frame=False,
        enable_interrupt=False,
        key_bindings=kb,
    )
    return await prompt.prompt_async()


def render_user_turn(user_input: str, out: Console | None = None) -> None:
    out = _resolve_console(out)
    line = Text()
    #line.append("You", style="turn.user")
    #line.append("  ")
    #line.append(user_input)
    out.print(line)


def render_turn_header(
    turn_number: int,
    model_name: str,
    snapshot: SessionSnapshot,
    out: Console | None = None,
) -> None:
    out = _resolve_console(out)
    header = Table.grid(expand=True, padding=(0, 1))
    header.add_column(style="overview.label", no_wrap=True)
    header.add_column(ratio=1, style="overview.value")
    header.add_row(
        "Turn",
        f"{turn_number}  ·  model {model_name}  ·  thread {_short_id(snapshot.thread_id, 12)}  ·  session {_short_id(snapshot.session_id, 12)}",
    )
    out.print(Panel(header, border_style="panel.border", padding=(0, 1)))


def render_assistant_turn(out: Console | None = None) -> None:
    return None


async def prompt_for_interrupt(
    interrupt_payload: dict,
    current_session: SessionSnapshot,
    session_store: SessionStore,
    *,
    model_name: str = "",
    tools_count: int = 0,
    out: Console | None = None,
    selector=None,
) -> dict:
    out = _resolve_console(out)
    if interrupt_payload.get("kind") != "tool_approval":
        return {"approved": False}

    req_tools = interrupt_payload.get("tools", [])
    summary = _summarize_approval_request(req_tools)
    current_session.approval_mode = _normalize_approval_mode(getattr(current_session, "approval_mode", APPROVAL_MODE_PROMPT))
    if current_session.approval_mode == APPROVAL_MODE_ALWAYS:
        out.print("  [tool.badge]↳[/] [status.success]approved:[/] [overview.value]always[/]")
        return {"approved": True}

    selected = await (selector(summary) if selector else _run_approval_selector(summary))
    approved = False
    if selected == "always":
        current_session.approval_mode = APPROVAL_MODE_ALWAYS
        session_store.save_active_session(current_session)
        approved = True
    elif selected == "yes":
        approved = True

    if selected == "always":
        out.print("  [tool.badge]↳[/] [status.success]approved:[/] [overview.value]always[/]")
    elif selected == "yes":
        out.print("  [tool.badge]↳[/] [status.success]approved:[/] [overview.value]yes[/]")
    elif selected == "no":
        out.print("  [tool.badge]↳[/] [status.error]denied:[/] [overview.value]no[/]")
    else:
        out.print("  [tool.badge]↳[/] [status.error]denied:[/] [overview.value]cancel[/]")
    return {"approved": approved}


async def run_user_request(
    agent_app,
    thread_id: str,
    session_id: str,
    user_input: str,
    config: AgentConfig,
    stream_processor_cls,
    current_session: SessionSnapshot,
    session_store: SessionStore,
    *,
    model_name: str = "",
    tools_count: int = 0,
    tool_metadata: dict[str, ToolMetadata] | None = None,
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
        processor = stream_processor_cls(console, tool_metadata=tool_metadata)
        result: StreamProcessResult = await processor.process_stream(stream)
        if result.interrupt is None:
            return result.stats
        payload = Command(
            resume=await prompt_for_interrupt(
                result.interrupt,
                current_session,
                session_store,
                model_name=model_name,
                tools_count=tools_count,
            )
        )


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
        console.print(_error_panel(f"Config error: {e}"))
        return

    store = SessionStore(config.session_state_path)

    tool_registry = None
    try:
        agent_app, tool_registry = await initialize_agent(config)
    except Exception as e:
        console.print(_error_panel(f"Init error: {e}"))
        return

    checkpoint_info = tool_registry.checkpoint_info
    current_session = store.load_active_session() or store.new_session(
        checkpoint_backend=checkpoint_info.get("resolved_backend", config.checkpoint_backend),
        checkpoint_target=checkpoint_info.get("target", "unknown"),
    )
    current_session.approval_mode = _normalize_approval_mode(getattr(current_session, "approval_mode", APPROVAL_MODE_PROMPT))
    store.save_active_session(current_session)

    tools = tool_registry.tools
    render_overview(config, tool_registry, current_session, show_cheatsheet=True)
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
        clear_screen()
        render_overview(config, tool_registry, current_session, show_cheatsheet=True)

    def show_session() -> None:
        render_overview(config, tool_registry, current_session, show_cheatsheet=False)

    model_name = config.gemini_model if config.provider == "gemini" else config.openai_model

    def _toolbar():
        return get_bottom_toolbar(mode="normal", model_name=model_name, tools_count=len(tools))

    turn_count = 0

    while True:
        try:
            if last_stats:
                console.print(last_stats, justify="right")
                last_stats = None

            user_input = (await session.prompt_async(get_prompt_message(), bottom_toolbar=_toolbar)).strip()
            if not user_input:
                continue
            if user_input.lower() == "/quit":
                break
            if try_handle_command(user_input, tool_registry, reset_session, show_session):
                continue

            if turn_count > 0:
                console.print(Rule(style="turn.separator"))
            turn_count += 1
            render_user_turn(user_input)
            render_assistant_turn()

            last_stats = await run_user_request(
                agent_app,
                current_session.thread_id,
                current_session.session_id,
                user_input,
                config,
                StreamProcessor,
                current_session,
                store,
                model_name=model_name,
                tools_count=len(tools),
                tool_metadata=getattr(tool_registry, "tool_metadata", {}),
            )
            store.save_active_session(current_session)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("[dim]  ✕ cancelled[/]")
            continue
        except Exception as e:
            friendly = format_exception_friendly(e)
            console.print(_error_panel(friendly))

    await close_runtime_resources(tool_registry)


if __name__ == "__main__":
    asyncio.run(main())
