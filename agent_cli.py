import sys
from pathlib import Path
from core.constants import BASE_DIR

# Используем insert(0) для приоритета модулей проекта над системными
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import os
import asyncio
import warnings
import time
import logging
import re
from typing import Optional

# --- UI IMPORTS ---
from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.padding import Padding
from rich.table import Table
from rich import box

# --- PROMPT IMPORTS ---
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer
from prompt_toolkit.history import FileHistory

# --- LANGCHAIN IMPORTS ---
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk, SystemMessage

# --- LOCAL IMPORTS ---
try:
    from agent import AgentWorkflow, logger
except ImportError as e:
    # Если импорт не удался даже с BASE_DIR, пробуем текущую директорию как fallback
    # но только если BASE_DIR отличается от cwd
    if str(Path.cwd()) != str(BASE_DIR):
        sys.path.append(".")
    try:
        from agent import AgentWorkflow, logger
    except ImportError:
        raise ImportError(f"Could not import 'agent' module. sys.path: {sys.path}. Error: {e}")

from core.cli_utils import (
    TokenTracker, 
    clean_markdown_text, 
    parse_thought, 
    format_tool_output, 
    get_key_bindings
)
from core.config import AgentConfig

# --- CONFIG ---
warnings.filterwarnings("ignore")
console = Console()
logging.getLogger("httpx").setLevel(logging.WARNING)

# ======================================================
# STREAM PROCESSOR
# ======================================================

class StreamProcessor:
    """Стабильный процессор стриминга. Не теряет текст, так как использует накопление."""
    
    def __init__(self):
        self.tracker = TokenTracker()
        self.full_text = ""          
        self.printed_len = 0         
        self.printed_tool_ids = set()
        self.tool_buffer = {}        
        self.status_text = "Thinking..."
        self.start_time = time.time()

    async def run(self, agent_app, user_input: str, thread_id: str, max_loops: int):
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_loops * 4}
        
        try:
            with Live(Spinner("dots", text=self.status_text, style="cyan"), 
                      refresh_per_second=10, 
                      console=console, 
                      transient=True) as live:
                
                async for mode, payload in agent_app.astream(
                    {"messages": [HumanMessage(content=user_input)], "steps": 0},
                    config=config,
                    stream_mode=["messages", "updates"]
                ):
                    await asyncio.sleep(0.005) 

                    # 1. ОБНОВЛЕНИЯ ОТ УЗЛОВ
                    if mode == "updates":
                        self.tracker.update_from_node_update(payload)
                        self._commit_printed_text(live)

                        if "agent" in payload:
                            messages = payload["agent"].get("messages", [])
                            if not isinstance(messages, list): messages = [messages]
                            last_msg = messages[-1] if messages else None
                            
                            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                                for tc in last_msg.tool_calls:
                                    self.tool_buffer[tc["id"]] = {"name": tc["name"], "args": tc["args"]}

                    # 2. ПОТОК СООБЩЕНИЙ
                    elif mode == "messages":
                        msg, metadata = payload
                        node = metadata.get("langgraph_node")
                        self.tracker.update_from_message(msg)
                        
                        # Self-correction / validator feedback
                        # В текущем графе нет отдельного узла "validator".
                        # Подсказки валидации приходят как SystemMessage из узла "tools".
                        if node == "tools" and isinstance(msg, SystemMessage):
                            error_preview = str(msg.content).split('\n')[0]
                            live.console.print(Padding(f"🔧 [bold magenta]Self-Correction:[/bold magenta] {error_preview}", (0, 0, 0, 4)))
                            self.status_text = "Correcting strategy..."
                            
                        # Quality Gate
                        if node == "agent" and isinstance(msg, SystemMessage):
                            warning_preview = msg.content
                            if len(warning_preview) > 100: warning_preview = warning_preview[:97] + "..."
                            live.console.print(Padding(f"🛡️ [bold orange3]Quality Gate:[/bold orange3] {warning_preview}", (0, 0, 0, 4)))
                            self.status_text = "Safety protocol triggered..."
                            
                        if node == "agent" and isinstance(msg, (AIMessage, AIMessageChunk)):
                            if msg.tool_calls:
                                self._commit_printed_text(live)
                                for tc in msg.tool_calls:
                                    self._handle_tool_call(tc, live)
                            
                            if msg.content:
                                chunk = msg.content if isinstance(msg.content, str) else ""
                                if isinstance(msg.content, list):
                                    chunk = "".join(x.get("text", "") for x in msg.content if isinstance(x, dict))
                                self.full_text += chunk

                        elif node == "tools" and isinstance(msg, ToolMessage):
                            self._handle_tool_result(msg, live)
                            
                    self._update_live_display(live)

        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[bold red]🛑 Stopped by user[/]")
            return 

        self._commit_printed_text(None)
        console.print(self.tracker.render(time.time() - self.start_time), justify="right")

    def _handle_tool_call(self, tc, live):
        t_id, t_name = tc.get("id"), tc.get("name")
        args = tc.get("args", {})
        self.tool_buffer[t_id] = {"name": t_name, "args": args}

        arg_str = ""
        if isinstance(args, dict):
            priority_keys = ["query", "queries", "path", "file_path", "url", "urls", "filename"]
            for key in priority_keys:
                if key in args:
                    val = args[key]
                    arg_str = str(val) if isinstance(val, list) else str(val)
                    break
            if not arg_str and args:
                arg_str = str(list(args.values())[0])
        elif isinstance(args, str):
            arg_str = args

        arg_display = ""
        if arg_str:
            clean_arg = str(arg_str).strip().replace("\n", " ")
            if len(clean_arg) > 50: clean_arg = clean_arg[:47] + "..."
            arg_display = f" [dim]{clean_arg}[/]"

        self.status_text = f"[bold cyan]Thinking:[/] {t_name}{arg_display}"
            
    def _handle_tool_result(self, msg, live):
        t_id = msg.tool_call_id
        
        # Печатаем вызов, если его еще не было
        if t_id in self.tool_buffer and t_id not in self.printed_tool_ids:
            info = self.tool_buffer[t_id]
            t_name = info["name"]
            args = info["args"]
            
            # --- Формирование строки аргументов ---
            arg_str = ""
            if isinstance(args, dict):
                priority_keys = ["query", "queries", "path", "file_path", "url", "urls", "filename"]
                for key in priority_keys:
                    if key in args:
                        val = args[key]
                        arg_str = str(val) if isinstance(val, list) else str(val)
                        break
                if not arg_str and args:
                    arg_str = str(list(args.values())[0])
            elif isinstance(args, str):
                arg_str = args

            arg_display = ""
            if arg_str:
                clean_arg = str(arg_str).strip().replace("\n", " ")
                if len(clean_arg) > 60: clean_arg = clean_arg[:57] + "..."
                arg_display = f" [dim]{clean_arg}[/]"
            # --------------------------------------

            live.console.print(Padding(f"🌍 [bold cyan]Call:[/] {t_name}{arg_display}", (0, 0, 0, 2)))
            self.printed_tool_ids.add(t_id)

        # Печатаем результат
        content_str = str(msg.content)
        is_error = getattr(msg, "status", "") == "error" or content_str.startswith(("Error", "Ошибка"))
        icon = "❌" if is_error else "✅"
        color = "red" if is_error else "green"
        summary = format_tool_output(msg.name, content_str, is_error)
        
        live.console.print(Padding(f"[{color}]{icon} {msg.name}:[/] [dim]{summary}[/]", (0, 0, 0, 4)))
        self.status_text = "Analyzing..."

    def _commit_printed_text(self, live: Optional[Live]):
        _, clean_full, _ = parse_thought(self.full_text)
        
        if len(clean_full) > self.printed_len:
            new_text = clean_full[self.printed_len:]
            cleaned_chunk = clean_markdown_text(new_text)
            
            target = live.console if live else console
            target.print(Padding(Markdown(cleaned_chunk), (0, 0, 0, 2)))
            self.printed_len = len(clean_full)

    def _update_live_display(self, live: Live):
        # Используем parse_thought из утилит (вернет thought, clean_text, has_thought)
        thought_content, clean_full, has_thought = parse_thought(self.full_text)
        
        if has_thought and thought_content:
            self.status_text = f"[yellow italic]{thought_content[-60:]}...[/]"
        
        pending = clean_full[self.printed_len:]
        renderable = Spinner("dots", text=self.status_text, style="cyan")
        
        if pending.strip():
             renderable = Group(
                Padding(Markdown(clean_markdown_text(pending)), (0, 0, 0, 2)),
                renderable
             )
        live.update(renderable)

# ======================================================
# MAIN ENTRY POINT
# ======================================================

async def main():
    os.system("cls" if os.name == "nt" else "clear")
    console.print(Panel("[bold blue]AI Agent CLI[/]", subtitle="v6.3b"))

    # 1. Загружаем конфиг
    temp_cfg = AgentConfig()
    
    # 2. НАСТРОЙКА ЛОГИРОВАНИЯ
    # Это влияет ТОЛЬКО на внутренние сообщения (Router, Init, Summary)
    if temp_cfg.debug:
        # В режиме DEBUG показываем всё нутро
        logger.setLevel(logging.DEBUG)
        console.print("[yellow]🐛 Debug mode enabled (Internal Logs Visible)[/]")
    else:
        # В обычном режиме скрываем технические логи
        # Но UI (StreamProcessor) продолжит работать и показывать инструменты!
        logger.setLevel(logging.WARNING) 
        
        # Гасим шумы от библиотек
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    try:
        # Спиннер инициализации
        with console.status("[bold green]Initializing system...[/]", spinner="dots"):
            workflow = AgentWorkflow()
            await workflow.initialize_resources()
            agent_app = workflow.build_graph()
        console.print("[bold green]System Ready![/]")

    except Exception as e:
        console.print(f"[bold red]Init Error:[/] {e}")
        if temp_cfg.debug:
            import traceback
            traceback.print_exc()
        return

    # 3. Вывод информации о модели
    cfg = workflow.config
    console.print(
        f"[dim]Model:[/] [bold cyan]{cfg.gemini_model if cfg.provider == 'gemini' else cfg.openai_model}[/] "
        f"[dim]Temp:[/] [bold cyan]{cfg.temperature}[/] "
        f"[dim]Tools:[/] [bold cyan]{len(workflow.tools)}[/] "
    )
    console.print("[bold blue]Enter[/] [bold green]↵[/] — send  |  [bold blue]Alt+Enter[/] [bold yellow]⎇ ↵[/] — new line | [green]/tools[/] | [green]/help\n")

    session = PromptSession(
        history=FileHistory(".history"),
        style=Style.from_dict({"prompt": "bold cyan"}),
        key_bindings=get_key_bindings(),
        lexer=PygmentsLexer(MarkdownLexer)
    )

    thread_id = "main_session"

    while True:
        try:
            cwd_name = Path.cwd().name
            user_input = await session.prompt_async(f"User (./{cwd_name}) > ")
            user_input = user_input.strip()
            
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break
            if user_input.lower() in ["clear", "reset"]:
                thread_id = f"session_{int(time.time())}"
                console.print("[yellow]♻ New session started[/]")
                continue
            
            if user_input.lower() in ["/help", "/tools"]:
                # ... (код отрисовки таблицы Help, как был) ...
                table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
                table.add_column("Tool", style="green")
                table.add_column("Description")
                for t in workflow.tools:
                    desc = t.description.split("\n")[0] if t.description else "No description"
                    if len(desc) > 60: desc = desc[:57] + "..."
                    table.add_row(t.name, desc)

                if user_input.lower() == "/tools":
                    console.print(Panel(table, title="[bold blue]Available Tools[/]", border_style="blue"))
                else:
                    console.print(Panel(
                        Group(
                            Markdown("### 🎮 Commands"),
                            Markdown("- `/help` - Show this menu"),
                            Markdown("- `/tools` - Show available tools"),
                            Markdown("- `exit` / `quit` - Close application"),
                            Markdown("- `clear` / `reset` - Start new session"),
                            Markdown("- `Alt+Enter` - Multi-line input"),
                            Markdown("---"),
                            Markdown("### 🛠 Available Tools"),
                            table
                        ),
                        title="[bold blue]Help Menu[/]",
                        border_style="blue"
                    ))
                continue

            # Запуск процессора
            # Он будет печатать Call/Result, так как это часть UI
            processor = StreamProcessor()
            await processor.run(agent_app, user_input, thread_id, cfg.max_loops)
            console.print()

        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]Cancelled. Type 'exit' to quit.[/]")
            continue
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            if cfg.debug:
                import traceback
                logger.debug(traceback.format_exc())
        finally:
            if 'workflow' in locals() and workflow:
                # Гарантируем закрытие соединений MCP и NetworkClient
                if workflow.tool_registry:
                    await workflow.tool_registry.cleanup()
                
                # Закрываем NetworkClient если он был инициализирован
                try:
                    from tools.system_tools import _net_client
                    if _net_client:
                        await _net_client.close()
                except ImportError:
                    pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass