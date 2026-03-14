from rich.theme import Theme
from rich.style import Style

# Define the central theme for the agent
# Modern, professional color palette (Tokyonight / Catppuccin inspired)
AGENT_THEME = Theme({
    # General status
    "status.spinner": Style(color="cyan", bold=True),
    "status.text": Style(color="#7dcfff"),  # Light Blue
    "status.error": Style(color="#f7768e", bold=True),  # Red
    "status.warning": Style(color="#e0af68", bold=True),  # Yellow/Orange
    "status.success": Style(color="#9ece6a", bold=True),  # Green

    # Tool execution
    "tool.name": Style(color="#bb9af7", bold=True),  # Purple
    "tool.args": Style(color="#565f89", italic=True),  # Comment-like grey-blue
    "tool.result": Style(color="#73daca"),  # Teal
    "tool.error": Style(color="#db4b4b", bold=True),
    "tool.timing": Style(color="#414868", italic=True),  # Subtle dim timing
    "tool.badge": Style(color="#9d7cd8"),  # Muted purple for tool badge

    # Agent interaction
    "agent.thought": Style(color="#565f89", italic=True),  # Subtle thought color
    "agent.node": Style(color="#7dcfff", italic=True),    # Active node hint

    # Approval UI
    "approval.border": Style(color="#e0af68"),           # Warning yellow border
    "approval.danger": Style(color="#f7768e", bold=True), # Destructive flag
    "approval.mutating": Style(color="#e0af68", bold=True), # Mutating flag
    "approval.networked": Style(color="#7dcfff"),          # Network flag

    # Conversation turns
    "turn.separator": Style(color="#292e42"),  # Barely-visible divider
    "turn.user": Style(color="#0077c2", bold=True),

    # Stats / metrics
    "stats.text": Style(color="#414868", italic=True),  # Dim inline stats
    "stats.time": Style(color="#565f89"),
    "stats.tokens": Style(color="#7aa2f7"),

    # Init / startup
    "init.step": Style(color="#9ece6a"),
    "init.info": Style(color="#7dcfff", dim=True),

    # Panels and Layouts
    "panel.border": Style(color="#3b4261"),  # Dark Grey Blue
    "panel.title": Style(color="#7aa2f7", bold=True),
    "panel.error": Style(color="#f7768e"),   # Error panel border
    "panel.warning": Style(color="#e0af68"), # Warning panel border

    # Code
    "code.block": Style(color="#c0caf5"),
    "markdown.code": Style(color="#ff9e64"),  # Orange for inline code

    # Markdown readability on dark backgrounds
    "markdown.h1": Style(color="#7dcfff", bold=True),
    "markdown.h2": Style(color="#7dcfff", bold=True),
    "markdown.h3": Style(color="#9ece6a", bold=True),
    "markdown.h4": Style(color="#9ece6a", bold=True),
    "markdown.link": Style(color="#73daca", underline=True, bold=True),
    "markdown.link_url": Style(color="#7aa2f7", underline=True),
})
