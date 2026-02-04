from rich.theme import Theme
from rich.style import Style

# Define the central theme for the agent
# This allows consistent coloring across the application
AGENT_THEME = Theme({
    # General status
    "status.spinner": Style(color="cyan", bold=True),
    "status.text": Style(color="cyan"),
    "status.error": Style(color="red", bold=True),
    "status.warning": Style(color="yellow", bold=True),
    "status.success": Style(color="green", bold=True),
    
    # Tool execution
    "tool.name": Style(color="cyan", bold=True),
    "tool.args": Style(color="white", dim=True),
    "tool.result": Style(color="cyan", dim=True),
    "tool.error": Style(color="red"),
    
    # Agent interaction
    "agent.thought": Style(color="yellow", italic=True),
    "agent.say": Style(color="blue", bold=True),
    "user.say": Style(color="green", bold=True),
    
    # Panels and Layouts
    "panel.border": Style(color="blue"),
    "panel.title": Style(color="blue", bold=True),
    
    # Code
    "code.block": Style(color="white"),
})

# Helper function to get the console with theme
def get_console():
    from rich.console import Console
    return Console(theme=AGENT_THEME)
