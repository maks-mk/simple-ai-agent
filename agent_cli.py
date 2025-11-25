import os
import asyncio
import warnings
import time
from typing import Dict, Any, Optional

# === IMPORTS FROM CORE AGENT ===
from agent import create_agent_graph, AgentConfig, logger

# === LANGCHAIN ===
from langchain_core.messages import HumanMessage, BaseMessage

# === RICH UI ===
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text

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

# === HELPER CLASSES ===

class TokenTracker:
    """Класс для накопления статистики использования токенов в потоке."""
    def __init__(self):
        self.usage_stats: Dict[str, Dict[str, int]] = {}

    def update(self, message: BaseMessage):
        """Обновляет статистику на основе метаданных сообщения."""
        if not hasattr(message, "usage_metadata") or not message.usage_metadata:
            return

        msg_id = getattr(message, "id", "unknown")
        new_usage = message.usage_metadata

        if msg_id in self.usage_stats:
            current = self.usage_stats[msg_id]
            # Берем MAX, так как в стриме данные могут приходить кумулятивно
            self.usage_stats[msg_id] = {
                "input_tokens": max(current.get("input_tokens", 0), new_usage.get("input_tokens", 0)),
                "output_tokens": max(current.get("output_tokens", 0), new_usage.get("output_tokens", 0)),
            }
        else:
            self.usage_stats[msg_id] = new_usage

    def display(self, duration: float) -> str:
        """Формирует строку статистики для вывода."""
        total_in = sum(s.get("input_tokens", 0) for s in self.usage_stats.values())
        total_out = sum(s.get("output_tokens", 0) for s in self.usage_stats.values())
        
        # Собираем текст статистики
        text = f"⏱ {duration:.1f}s"
        if total_in + total_out > 0:
            text += f" | 🪙 In: {total_in} / Out: {total_out}"
            
        # Оборачиваем весь текст в [bright_black] (ярко-черный = серый)
        # [dim] можно добавить дополнительно, если нужно еще тусклее
        return f"[bright_black]{text}[/]"
# === MAIN LOGIC ===

def setup_key_bindings() -> KeyBindings:
    """Настройка горячих клавиш: Enter отправляет, Alt+Enter переносит строку."""
    kb = KeyBindings()

    @kb.add('enter')
    def _(event):
        # Если буфер пуст, ничего не делаем
        if not event.current_buffer.text.strip():
            return
        event.current_buffer.validate_and_handle()

    @kb.add('escape', 'enter') # Alt+Enter
    def _(event):
        event.current_buffer.insert_text("\n")
        
    return kb

async def interactive_loop(agent):
    """Главный цикл взаимодействия с пользователем."""
    
    # Конфигурация сессии LangGraph
    thread_id = "main"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Стили prompt_toolkit
    style = Style.from_dict({
        "myself": "#00ffff bold",
    })
    
    session = PromptSession(
        multiline=True, 
        key_bindings=setup_key_bindings(), 
        style=style, 
        history=FileHistory(".agent_history"),
        prompt_continuation=lambda w, l, c: ". ", 
        lexer=PygmentsLexer(MarkdownLexer),
    )
    
    console.print("\n[bold green]Чат начат[/] (Enter = Отправить, Alt+Enter = Новая строка)\n")

    while True:
        try:
            console.print(Rule(style="dim cyan"))
            user_input = await session.prompt_async([("class:myself", "You > ")])
            user_input = user_input.strip()
            
            # Команды управления
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break
            if user_input.lower() in ["reset", "clear"]:
                thread_id = f"session-{time.time()}"
                config["configurable"]["thread_id"] = thread_id
                console.print("[yellow]♻  Контекст сброшен (новая сессия)[/]")
                continue

            # Инициализация переменных для обработки ответа
            accumulated_text = ""
            start_time = time.time()
            tracker = TokenTracker()
            
            # Визуализация процесса
            with Live(Spinner("dots", text="Думаю...", style="cyan"), refresh_per_second=12, console=console) as live:
                
                async for event in agent.astream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages"
                ):
                    message, metadata = event
                    node = metadata.get("langgraph_node")
                    
                    # 1. Обновляем токены
                    tracker.update(message)

                    # 2. Обработка текстового ответа от Агента
                    if node == "agent":
                        # Если сообщение содержит вызовы инструментов
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            # Если был накоплен текст до вызова инструмента, выводим его
                            if accumulated_text.strip():
                                live.console.print(Padding(Markdown(accumulated_text), (0, 1, 0, 1)))
                                accumulated_text = "" 
                            
                            for tc in message.tool_calls:
                                live.update(Spinner("earth", text=f"[bold cyan]Выполняю:[/] {tc['name']}", style="cyan"))
                        
                        # Если сообщение содержит контент (текст)
                        elif message.content:
                            chunk = message.content
                            # Обработка списка (иногда бывает в мультимодальных ответах)
                            if isinstance(chunk, list):
                                chunk = "".join([p["text"] for p in chunk if "text" in p])
                            
                            if chunk:
                                accumulated_text += chunk
                                live.update(Padding(Markdown(accumulated_text), (0, 1, 0, 1)))

                    # 3. Обработка результатов инструментов
                    elif node == "tools":
                        name = getattr(message, "name", "tool")
                        
                        # Форматирование вывода инструмента
                        res_str = str(message.content)
                        if len(res_str) > 200:
                            preview = res_str[:200] + f"... [dim](+{len(res_str)-200} chars)[/]"
                        else:
                            preview = res_str
                            
                        # Вывод блока инструмента "над" текущим спиннером
                        live.console.print(Padding(f"[dim green]✓ {name}: {preview}[/]", (0, 0, 0, 4)))
                        live.update(Spinner("dots", text="Анализирую результат...", style="cyan"))

            # Вывод финальной статистики
            console.print(tracker.display(time.time() - start_time))

        except KeyboardInterrupt:
            console.print("\n[bold red]Прервано пользователем[/]")
            break
        except Exception as e:
            logger.exception("Runtime Error in CLI loop")
            console.print(f"\n[bold red]Ошибка цикла: {e}[/]")
            # Небольшая задержка перед повтором, чтобы не спамить ошибками
            await asyncio.sleep(1)

async def main():
    # Очистка консоли
    os.system("cls" if os.name == "nt" else "clear")
    
    console.print(Panel.fit("[bold blue]AI Agent CLI[/] [dim](LangGraph + MCP)[/]", style="blue"))

    try:
        console.print(Rule("Инициализация", style="blue"))
        
        # 1. Загрузка конфига через новый фабричный метод
        config = AgentConfig.from_env()

        # Вывод инфо
        model_name = config.gemini_model if config.provider == "gemini" else config.openai_model
        console.print(f"[dim]Провайдер:[/] [bold cyan]{config.provider.upper()}[/] | [dim]Модель:[/] [bold cyan]{model_name}[/]")
        
        # 2. Создание графа
        agent = await create_agent_graph(config)
        
        console.print(Panel(f"[green]Агент готов к работе[/]", style="green"))
        
        # 3. Запуск цикла
        await interactive_loop(agent)
        
    except Exception as e:
        console.print(f"[bold red]Критическая ошибка запуска: {e}[/]")
        # logger.exception("Critical startup error") # Раскомментировать если нужно логировать старт

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass