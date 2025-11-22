import sys
import os
import subprocess
import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

def run_cli():
    """Запускает CLI версию агента."""
    console.print("[bold cyan]🚀 Запуск CLI интерфейса...[/]")
    try:
        subprocess.run([sys.executable, "agent_cli.py"], check=False)
    except KeyboardInterrupt:
        pass

def run_ui():
    """Запускает Streamlit UI."""
    console.print("[bold cyan]🚀 Запуск Web UI (Streamlit)...[/]")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui.py"], check=False)
    except KeyboardInterrupt:
        pass

def main():
    if os.name == "nt": os.system("cls")
    else: os.system("clear")

    console.print(Panel.fit(
        "[bold blue]🤖 AI Agent Launcher[/]\n"
        "[dim]Выберите режим работы[/]",
        style="blue"
    ))

    # Если аргументы переданы в командной строке
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ["cli", "terminal"]:
            run_cli()
        elif mode in ["ui", "web", "streamlit"]:
            run_ui()
        else:
            console.print(f"[red]Неизвестный режим: {mode}[/]")
        return

    # Интерактивное меню
    console.print("1. [bold green]🖥️  CLI (Терминал)[/]")
    console.print("2. [bold blue]🌐 Web UI (Браузер)[/]")
    console.print("3. [dim]Выход[/]")

    choice = Prompt.ask("\nВаш выбор", choices=["1", "2", "3"], default="1")

    if choice == "1":
        run_cli()
    elif choice == "2":
        run_ui()
    else:
        console.print("👋 Пока!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
