from rich.theme import Theme
from rich.style import Style

# Define the central theme for the agent
# Modern, professional color palette (Tokyonight / Catppuccin inspired)
AGENT_THEME = Theme({
    # General status
    "status.spinner": Style(color="cyan", bold=True),
    "status.text": Style(color="#7dcfff"), # Light Blue
    "status.error": Style(color="#f7768e", bold=True), # Red
    "status.warning": Style(color="#e0af68", bold=True), # Yellow/Orange
    "status.success": Style(color="#9ece6a", bold=True), # Green
    
    # Tool execution
    "tool.name": Style(color="#bb9af7", bold=True), # Purple
    "tool.args": Style(color="#565f89", italic=True), # Comment-like grey-blue
    "tool.result": Style(color="#73daca"), # Teal
    "tool.error": Style(color="#db4b4b", bold=True),
    
    # Agent interaction
    "agent.thought": Style(color="#565f89", italic=True), # Subtle thought color
    
    # Panels and Layouts
    "panel.border": Style(color="#3b4261"), # Dark Grey Blue
    "panel.title": Style(color="#7aa2f7", bold=True),
    
    # Code
    "code.block": Style(color="#c0caf5"),
    "markdown.code": Style(color="#ff9e64"), # Orange for inline code
})
