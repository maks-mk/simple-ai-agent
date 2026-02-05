
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from pydantic.fields import FieldInfo
from pydantic import SecretStr

# Ensure PYTHONPATH
sys.path.append(str(Path(__file__).parent))

from core.config import AgentConfig

def check_all_env():
    # Force terminal to ensure output is captured
    console = Console(force_terminal=True)
    print("Starting configuration check...")
    
    try:
        # Load Config
        config = AgentConfig()
        
        table = Table(title="🔍 .env & Configuration Check", show_header=True, header_style="bold magenta")
        table.add_column("Config Field", style="cyan", no_wrap=True)
        table.add_column("Env Variable (Alias)", style="blue")
        table.add_column("Current Value", style="green")
        table.add_column("Status", style="bold")

        # Get all fields
        fields = AgentConfig.model_fields
        
        for field_name, field_info in fields.items():
            # Determine Env Alias
            env_alias = field_info.alias if field_info.alias else field_name.upper()
            
            # Get Value
            value = getattr(config, field_name)
            
            # Mask Secrets
            display_value = str(value)
            status = "[green]OK[/]"
            
            if isinstance(value, SecretStr):
                display_value = "********"
                if not value.get_secret_value():
                    status = "[red]MISSING[/]"
                    display_value = "None"
            elif value is None:
                 # Check if it is required
                 if field_info.is_required():
                     status = "[red]MISSING (Required)[/]"
                     display_value = "None"
                 else:
                     status = "[dim]Empty (Optional)[/]"
                     display_value = "None"
            
            # Special check for paths
            if isinstance(value, Path):
                if value.exists():
                    status = "[green]FOUND[/]"
                else:
                    status = "[yellow]NOT FOUND (File)[/]"
            
            table.add_row(field_name, env_alias, display_value, status)
            
        console.print(table)
        
        # Provider Check
        console.print(f"\n[bold]Active Provider:[/] [yellow]{config.provider}[/]")
        if config.provider == "openai":
            if config.openai_api_key:
                console.print("✅ OpenAI Key is set.")
            else:
                console.print("❌ OpenAI Key is MISSING!")
        elif config.provider == "gemini":
            if config.gemini_api_key:
                console.print("✅ Gemini Key is set.")
            else:
                console.print("❌ Gemini Key is MISSING!")
                
    except Exception as e:
        console.print(f"[bold red]Critical Error loading config:[/] {e}")

if __name__ == "__main__":
    check_all_env()
