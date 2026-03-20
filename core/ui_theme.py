from rich.theme import Theme
from rich.style import Style

# Define the central theme for the agent
# Modern, professional color palette (Tokyonight / Catppuccin inspired)
AGENT_THEME = Theme({
    # General status
    "status.spinner": Style(color="#5ad1e6", bold=True),
    "status.text": Style(color="#77c5ff"),
    "status.error": Style(color="#ff6b6b", bold=True),
    "status.warning": Style(color="#f4b942", bold=True),
    "status.success": Style(color="#7fd17f", bold=True),

    # Tool execution
    "tool.name": Style(color="#ffd166", bold=True),
    "tool.args": Style(color="#7c8799", italic=True),
    "tool.result": Style(color="#5ed0b1"),
    "tool.error": Style(color="#ff6b6b", bold=True),
    "tool.timing": Style(color="#6f7b8e", italic=True),
    "tool.badge": Style(color="#f4b942", bold=True),
    "tool.readonly": Style(color="#7fd17f", bold=True),
    "tool.mcp": Style(color="#77c5ff", bold=True),

    # Agent interaction
    "agent.thought": Style(color="#7c8799", italic=True),
    "agent.node": Style(color="#77c5ff", italic=True),

    # Approval UI
    "approval.border": Style(color="#f4b942"),
    "approval.danger": Style(color="#ff6b6b", bold=True),
    "approval.mutating": Style(color="#f4b942", bold=True),
    "approval.networked": Style(color="#77c5ff"),
    "approval.summary": Style(color="#d6deeb", bold=True),

    # Conversation turns
    "turn.separator": Style(color="#334155"),
    "turn.user": Style(color="#0077c2", bold=True),
    "turn.assistant": Style(color="#5ed0b1", bold=True),

    # Stats / metrics
    "stats.text": Style(color="#6f7b8e", italic=True),
    "stats.time": Style(color="#7c8799"),
    "stats.tokens": Style(color="#77c5ff"),

    # Init / startup
    "init.step": Style(color="#7fd17f"),
    "init.info": Style(color="#77c5ff", dim=True),

    # Panels and Layouts
    "panel.border": Style(color="#3c495d"),
    "panel.title": Style(color="#77c5ff", bold=True),
    "panel.error": Style(color="#ff6b6b"),
    "panel.warning": Style(color="#f4b942"),
    "overview.label": Style(color="#8c98a8", bold=True),
    "overview.value": Style(color="#d6deeb"),

    # Code
    "code.block": Style(color="#d6deeb"),
    "markdown.code": Style(color="#ffb86c"),

    # Markdown readability on dark backgrounds
    "markdown.h1": Style(color="#77c5ff", bold=True),
    "markdown.h2": Style(color="#77c5ff", bold=True),
    "markdown.h3": Style(color="#7fd17f", bold=True),
    "markdown.h4": Style(color="#7fd17f", bold=True),
    "markdown.link": Style(color="#5ed0b1", underline=True, bold=True),
    "markdown.link_url": Style(color="#77c5ff", underline=True),
})
