import asyncio
from html import escape
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langgraph.types import Command
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts.choice_input import ChoiceInput
from rich import box
from rich.console import Console, Group
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
from core.tool_policy import ToolMetadata
from core.ui_theme import ACCENT_BLUE, AGENT_THEME, TEXT_MUTED, TEXT_PRIMARY

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent import build_agent_app

console = Console(theme=AGENT_THEME, color_system="truecolor")
logging.getLogger("httpx").setLevel(logging.WARNING)

COMMANDS = ["/help", "/tools", "/session", "/new", "/quit", "clear", "reset"]
CLEAR_COMMAND = "cls" if os.name == "nt" else "clear"
APPROVAL_MODE_PROMPT = "prompt"
APPROVAL_MODE_ALWAYS = "always"


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
        return HTML(
            f" {_html_style('Approval', fg=TEXT_PRIMARY, bold=True)}"
            f" | {_html_style('↑↓', fg=ACCENT_BLUE, bold=True)} choose"
            f" | {_html_style('Enter', fg=ACCENT_BLUE, bold=True)} confirm"
            f" | {_html_style('Esc', fg=ACCENT_BLUE, bold=True)} cancel "
        )

    model_part = (
        f" | {_html_style(model_name, fg=TEXT_PRIMARY, bold=True)}"
        if model_name
        else ""
    )
    tools_part = (
        f" | {_html_style(f'tools: {tools_count}', fg=TEXT_MUTED)}"
        if tools_count
        else ""
    )
    return HTML(
        f" {_html_style('ALT+ENTER', fg=ACCENT_BLUE, bold=True)} multiline"
        f" | {_html_style('/help', fg=ACCENT_BLUE, bold=True)} · {_html_style('/tools', fg=ACCENT_BLUE, bold=True)}"
        f" · {_html_style('/session', fg=ACCENT_BLUE, bold=True)} · {_html_style('/new', fg=ACCENT_BLUE, bold=True)}"
        f" | {_html_style('/quit', fg=ACCENT_BLUE, bold=True)} exit"
        f"{tools_part}{model_part} "
    )


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
    provider_icon = "[status.spinner]◆[/]"
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
        "[status.spinner]ready[/]"
        if issue_count == 0
        else f"[status.error]degraded ({issue_count} issue{'s' if issue_count != 1 else ''})[/]"
    )

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="overview.label", justify="right", no_wrap=True)
    table.add_column(style="overview.value", ratio=1)
    table.add_column(style="overview.label", justify="right", no_wrap=True)
    table.add_column(style="overview.value", ratio=1)

    table.add_row("Provider", f"{provider_icon} {provider_label}", "Model", model_name)
    table.add_row("Backend", backend, "Tools", str(tools_count))
    table.add_row("Session", _short_id(snapshot.session_id), "Thread", _short_id(snapshot.thread_id))
    table.add_row("Approvals", approvals, "MCP", mcp_text)
    table.add_row("Status", status_text, "Config", "debug" if config.debug else "standard")

    out.print(
        Panel(
            table,
            title=f"[panel.title]AI Agent[/] [dim]v{AGENT_VERSION}[/]",
            border_style="panel.border",
            padding=(0, 1),
        )
    )
    if show_cheatsheet:
        out.print(
            "[dim]/help[/] workflow guide  ·  [dim]/tools[/] inspect capabilities  ·  [dim]/session[/] overview  ·  [dim]/new[/] fresh session  ·  [dim]Alt+Enter[/] multiline"
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
    table.add_row("Approvals", "↑↓ choose Yes/No/Always  ·  Enter confirms  ·  Esc cancels")
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


def _summarize_approval_request(req_tools: list[dict]) -> ApprovalSummary:
    destructive_count = 0
    mutating_count = 0
    networked_count = 0
    impacts: set[str] = set()

    for tool in req_tools:
        policy = tool.get("policy") or {}
        name = (tool.get("name") or "").lower()
        destructive = bool(policy.get("destructive"))
        mutating = bool(policy.get("mutating"))
        networked = bool(policy.get("networked"))

        destructive_count += int(destructive)
        mutating_count += int(mutating)
        networked_count += int(networked)

        if destructive or mutating:
            if any(token in name for token in ("process", "pid", "port", "shell", "exec", "command")):
                impacts.add("processes")
            elif any(token in name for token in ("file", "directory", "path", "write", "edit", "delete", "download")):
                impacts.add("files")
            else:
                impacts.add("local state")
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


def _approval_compact_action(tool: dict) -> str:
    name = str(tool.get("name") or "action")
    target = _approval_target_text(tool)
    friendly = {
        "write_file": "Save file",
        "edit_file": "Edit file",
        "safe_delete_file": "Delete file",
        "delete_file": "Delete file",
        "download_file": "Download file",
        "cli_exec": "Run command",
    }.get(name, name)
    return f"{friendly}: {target}" if target else friendly


def _approval_summary_line(summary: ApprovalSummary) -> str:
    parts = []
    if summary.destructive_count:
        parts.append(f"[approval.danger]{summary.destructive_count} destructive[/]")
    if summary.networked_count:
        parts.append(f"[approval.networked]{summary.networked_count} networked[/]")
    if summary.impacts and (summary.destructive_count or summary.networked_count):
        parts.append(f"[dim]Affects:[/] {summary.impact_text}")
    if not parts and summary.mutating_count:
        parts.append("[dim]This will change local files.[/]")
    return "  ".join(parts)


def _use_detailed_approval_panel(summary: ApprovalSummary, req_tools: list[dict]) -> bool:
    return len(req_tools) > 1 or summary.destructive_count > 0 or summary.networked_count > 0


def _approval_panel_style(summary: ApprovalSummary) -> str:
    if summary.risk_level == "high":
        return "approval.danger"
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

    prompt = ChoiceInput(
        message="Approval choice:",
        options=[
            ("yes", "Yes     Approve this batch"),
            ("no", "No      Deny this batch"),
            ("always", "Always  Approve future protected actions in this session"),
        ],
        default=default_choice,
        mouse_support=True,
        symbol="(*)",
        show_frame=False,
        enable_interrupt=False,
        key_bindings=kb,
    )
    return await prompt.prompt_async()


def render_user_turn(user_input: str, out: Console | None = None) -> None:
    out = _resolve_console(out)
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=7, no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_row("[turn.user]You[/]", Text(user_input))
    out.print(grid)


def render_assistant_turn(out: Console | None = None) -> None:
    out = _resolve_console(out)
    out.print("[turn.assistant]Agent[/]")


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
        out.print(
            "  [approval.summary]Action[/] [status.success]auto-approved[/] "
            "[dim](always for this session · /new resets)[/]"
        )
        return {"approved": True}

    summary_line = _approval_summary_line(summary)
    panel_style = _approval_panel_style(summary)
    panel_body: Group | str
    if _use_detailed_approval_panel(summary, req_tools):
        table = Table(box=box.ROUNDED, show_header=True, header_style="table.header", expand=False)
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
        panel_body = Group(summary_line, table)
    else:
        tool = req_tools[0] if req_tools else {}
        action_line = f"[approval.summary]{_approval_compact_action(tool)}[/]"
        panel_body = action_line

    out.print(
        Panel(
            panel_body,
            title=f"[{panel_style}]⚠  Approval Required[/]",
            border_style=panel_style,
            padding=(0, 1),
        )
    )
    out.print("  [dim]Choose Yes, No or Always. Enter confirms.[/]")

    selected = await (selector(summary) if selector else _run_approval_selector(summary))
    approved = False
    if selected == "always":
        current_session.approval_mode = APPROVAL_MODE_ALWAYS
        session_store.save_active_session(current_session)
        approved = True
    elif selected == "yes":
        approved = True

    status = "[status.success]approved[/]" if approved else "[status.error]denied[/]"
    out.print(f"  [approval.summary]Action[/] {status}")
    if selected == "always":
        out.print("  [dim]I will stop asking in this session. Use /new to reset.[/]")
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
