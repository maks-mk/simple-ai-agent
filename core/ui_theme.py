from __future__ import annotations

from rich.style import Style
from rich.text import Text
from rich.theme import Theme

ACCENT_BLUE = "#6EA8FF"
TEXT_PRIMARY = "#F5F7FA"
TEXT_SECONDARY = "#CDD4DE"
TEXT_MUTED = "#97A0AB"
TEXT_DIM = "#6F7884"
BORDER = "#3C4452"
SEPARATOR = "#2B313C"
AMBER_WARNING = "#E8A838"
ERROR_RED = "#FF5A5F"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def blend_hex(start_hex: str, end_hex: str, factor: float) -> str:
    factor = max(0.0, min(1.0, factor))
    start = _hex_to_rgb(start_hex)
    end = _hex_to_rgb(end_hex)
    blended = tuple(
        round(start[index] + (end[index] - start[index]) * factor)
        for index in range(3)
    )
    return _rgb_to_hex(blended)


def shimmer_supported(color_system: str | None) -> bool:
    return color_system in {"256", "truecolor", "windows"}


def build_shimmer_text(label: str, *, phase: float, italic: bool = False, enabled: bool = True) -> Text:
    base_style = Style(color=TEXT_MUTED, bold=True, italic=italic)
    if not label:
        return Text("", style=base_style)
    if not enabled or len(label.strip()) < 2:
        return Text(label, style=base_style)

    glow_width = max(2.0, min(4.0, len(label) / 3))
    travel = max(len(label) + int(glow_width * 2), 1)
    center = (phase * 8.0) % travel - glow_width

    text = Text()
    for index, char in enumerate(label):
        if char.isspace():
            text.append(char, style=base_style)
            continue

        distance = abs(index - center)
        blend = max(0.0, 1.0 - (distance / glow_width))
        style = Style(
            color=blend_hex(TEXT_MUTED, TEXT_PRIMARY, blend),
            bold=True,
            italic=italic,
        )
        text.append(char, style=style)
    return text


AGENT_THEME = Theme({
    "status.spinner": Style(color=ACCENT_BLUE, bold=True),
    "status.label": Style(color=TEXT_MUTED, bold=True),
    "status.text": Style(color=TEXT_SECONDARY),
    "status.error": Style(color=ERROR_RED, bold=True),
    "status.warning": Style(color=TEXT_PRIMARY, bold=True),
    "status.success": Style(color=TEXT_PRIMARY, bold=True),

    "tool.name": Style(color=ACCENT_BLUE, bold=True),
    "tool.args": Style(color=TEXT_MUTED, italic=True),
    "tool.result": Style(color=TEXT_PRIMARY),
    "tool.error": Style(color=ERROR_RED, bold=True),
    "tool.timing": Style(color=TEXT_DIM, italic=True),
    "tool.badge": Style(color=TEXT_DIM, bold=True),
    "tool.readonly": Style(color=TEXT_DIM),
    "tool.mcp": Style(color=ACCENT_BLUE, bold=True),

    "agent.thought": Style(color=TEXT_MUTED, italic=True),
    "agent.node": Style(color=ACCENT_BLUE, italic=True),

    "approval.border": Style(color=AMBER_WARNING, bold=True),
    "approval.danger": Style(color=ERROR_RED, bold=True),
    "approval.mutating": Style(color=AMBER_WARNING, bold=True),
    "approval.networked": Style(color=ACCENT_BLUE),
    "approval.summary": Style(color=TEXT_PRIMARY, bold=True),

    "turn.separator": Style(color=SEPARATOR),
    "turn.user": Style(color=ACCENT_BLUE, bold=True),
    "turn.assistant": Style(color=TEXT_PRIMARY, bold=True),

    "stats.text": Style(color=TEXT_DIM, italic=True),
    "stats.time": Style(color=TEXT_MUTED),
    "stats.tokens": Style(color=ACCENT_BLUE),

    "init.step": Style(color=ACCENT_BLUE, bold=True),
    "init.info": Style(color=TEXT_MUTED, dim=True),

    "panel.border": Style(color=BORDER),
    "panel.title": Style(color=TEXT_PRIMARY, bold=True),
    "panel.error": Style(color=ERROR_RED),
    "panel.warning": Style(color=AMBER_WARNING),
    "overview.label": Style(color=TEXT_MUTED, bold=True),
    "overview.value": Style(color=TEXT_PRIMARY),
    "table.header": Style(color=TEXT_PRIMARY, bold=True),

    "code.block": Style(color=TEXT_PRIMARY),
    "markdown.code": Style(color=TEXT_PRIMARY),

    "markdown.h1": Style(color=TEXT_PRIMARY, bold=True),
    "markdown.h2": Style(color=TEXT_PRIMARY, bold=True),
    "markdown.h3": Style(color=ACCENT_BLUE, bold=True),
    "markdown.h4": Style(color=ACCENT_BLUE, bold=True),
    "markdown.link": Style(color=ACCENT_BLUE, underline=True, bold=True),
    "markdown.link_url": Style(color=ACCENT_BLUE, underline=True),
})
