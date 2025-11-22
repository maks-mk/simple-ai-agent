import os
import asyncio
import warnings
import time
import logging
from typing import Optional

# === IMPORTS FROM CORE AGENT ===
from agent import create_agent_graph, AgentConfig, logger

# === LANGCHAIN ===
from langchain_core.messages import HumanMessage

# === RICH UI ===
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner

# === PROMPT TOOLKIT ===
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer
from prompt_toolkit.history import FileHistory

# === SETUP ===
warnings.filterwarnings("ignore", category=DeprecationWarning)
console = Console()

async def interactive_loop(agent):
    config = {"configurable": {"thread_id": "main"}}
    
    # ИСПРАВЛЕНО: Простая схема стилей
    style = Style.from_dict({
        "myself": "#00ffff bold",  # cyan
    })
    
    bindings = KeyBindings()
    @bindings.add('escape', 'enter')
    def _(event): event.current_buffer.insert_text("\n")
    @bindings.add('enter')
    def _(event): event.current_buffer.validate_and_handle()
    
    session = PromptSession(
        multiline=True, 
        key_bindings=bindings, 
        style=style, 
        history=FileHistory(".agent_history"),
        prompt_continuation=lambda w, l, c: ". ", 
        lexer=PygmentsLexer(MarkdownLexer),
    )
    
    console.print("\n[bold green]Чат начат[/] (Enter = Отправить, Esc+Enter = Новая строка, 'exit' = Выход)\n")

    while True:
        try:
            console.print(Rule(style="dim cyan"))
            user_input = await session.prompt_async([("class:myself", "You > ")])
            user_input = user_input.strip()
            
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break
            if user_input.lower() in ["reset", "clear"]:
                config["configurable"]["thread_id"] = f"session-{time.time()}"
                console.print("[yellow]♻ Контекст сброшен[/]")
                continue

            accumulated_text = ""
            start_time = time.time()
            
            # Используем transient=False (по умолчанию), чтобы текст остался после завершения Live
            with Live(Spinner("dots", text="Думаю...", style="cyan"), refresh_per_second=12, console=console) as live:
                async for event in agent.astream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages"
                ):
                    message, metadata = event
                    node = metadata.get("langgraph_node")
                    
                    if node == "agent" and message.content:
                        chunk = message.content
                        if isinstance(chunk, list):
                            chunk = "".join([p["text"] for p in chunk if "text" in p])
                        
                        if chunk:
                            accumulated_text += chunk
                            # Обновляем текст в реальном времени
                            live.update(Padding(Markdown(accumulated_text), (0, 1, 0, 1)))

                    if node == "agent" and hasattr(message, "tool_calls") and message.tool_calls:
                        # Если есть накопленный текст перед вызовом инструмента, фиксируем его
                        if accumulated_text.strip():
                            live.console.print(Padding(Markdown(accumulated_text), (0, 1, 0, 1)))
                            accumulated_text = "" # Очищаем буфер
                        
                        for tc in message.tool_calls:
                            live.update(Spinner("earth", text=f"[bold cyan]Выполняю:[/] {tc['name']}", style="cyan"))
                    
                    elif node == "tools":
                        name = getattr(message, "name", "tool")
                        res_str = str(message.content)
                        preview = res_str[:150] + "..." if len(res_str) > 150 else res_str
                        # Печатаем результат инструмента отдельно
                        live.console.print(Padding(f"[dim green]✓ {name}: {preview}[/]", (0, 0, 0, 4)))
                        live.update(Spinner("dots", text="Анализирую результат...", style="cyan"))

            console.print(f"[dim right]⏱ {time.time() - start_time:.1f}s[/]")

        except KeyboardInterrupt:
            console.print("\n[bold red]Прервано[/]")
            break
        except Exception as e:
            logger.exception("Runtime Error")
            console.print(f"\n[bold red]Ошибка цикла: {e}[/]")

async def main():
    if os.name == "nt": os.system("cls")
    else: os.system("clear")
    
    console.print(Panel.fit("[bold blue]AI Agent CLI[/] [dim](MCP + Local Tools)[/]", style="blue"))

    try:
        console.print(Rule("Инициализация", style="blue"))
        
        # Используем общую фабрику
        config = AgentConfig()

        # Выводим информацию о модели
        model_name = config.gemini_model if config.provider == "gemini" else config.openai_model
        console.print(f"[dim]Провайдер:[/] [bold cyan]{config.provider.upper()}[/] | [dim]Модель:[/] [bold cyan]{model_name}[/]")
        
        agent = await create_agent_graph(config)
        
        console.print(Panel(f"[green]Агент готов к работе[/]", style="green"))
        
        await interactive_loop(agent)
    except Exception as e:
        console.print(f"[bold red]Критическая ошибка запуска: {e}[/]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
