import sys
import os
import asyncio
import warnings
import time
import logging
import re  # Добавлен импорт
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dotenv import load_dotenv

# --- PROJECT IMPORTS ---
from core.constants import BASE_DIR

# Ensure project modules take precedence
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# --- UI IMPORTS ---
from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.padding import Padding
from rich.table import Table
from rich import box
from rich.syntax import Syntax  # Добавлен импорт

# --- PROMPT IMPORTS ---
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML

# --- LANGCHAIN IMPORTS ---
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk, SystemMessage

# --- LOCAL IMPORTS ---
try:
    from agent import AgentWorkflow, logger
except ImportError as e:
    # Fallback for running from non-root directory
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
# Suppress noisy logs from libs
logging.getLogger("httpx").setLevel(logging.WARNING)

# ======================================================
# STREAM PROCESSOR
# ======================================================

class StreamProcessor:
    """
    Handles real-time streaming of agent execution events and messages.
    """
    
    def __init__(self, console: Console):
        self.console = console
        self.tracker = TokenTracker()
        self.full_text = ""          
        self.printed_len = 0         
        self.printed_tool_ids: Set[str] = set()
        self.tool_buffer: Dict[str, Dict[str, Any]] = {}        
        self.status_text = "Thinking..."
        self.start_time = time.time()

    async def run(self, agent_app, user_input: str, thread_id: str, max_loops: int, token_budget: int = 0):
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_loops * 4}
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)], 
            "steps": 0,
            "token_budget": token_budget, 
            "token_used": 0 
        }
        
        try:
            with Live(Spinner("dots", text=self.status_text, style="cyan"), 
                      refresh_per_second=10, 
                      console=self.console, 
                      transient=True) as live:
                
                async for mode, payload in agent_app.astream(
                    initial_state,
                    config=config,
                    stream_mode=["messages", "updates"]
                ):
                    # Slight delay to allow UI updates
                    await asyncio.sleep(0.005) 

                    if mode == "updates":
                        self._handle_updates(payload, live)
                    elif mode == "messages":
                        self._handle_messages(payload, live)
                            
                    self._update_live_display(live)

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.console.print("\n[bold red]🛑 Stopped by user[/]")
            return 

        # Final commit of any remaining text
        self._commit_printed_text(None)
        
        # Show stats
        duration = time.time() - self.start_time
        self.console.print(self.tracker.render(duration), justify="right")

    def _handle_updates(self, payload: Dict, live: Live):
        """Processes state updates from the graph."""
        self.tracker.update_from_node_update(payload)
        self._commit_printed_text(live)

        if "agent" in payload:
            messages = payload["agent"].get("messages", [])
            if not isinstance(messages, list): messages = [messages]
            last_msg = messages[-1] if messages else None
            
            # Capture tool calls for display
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    self.tool_buffer[tc["id"]] = {"name": tc["name"], "args": tc["args"]}

    def _handle_messages(self, payload: tuple, live: Live):
        """Processes streamed messages."""
        msg, metadata = payload
        node = metadata.get("langgraph_node")
        self.tracker.update_from_message(msg)
        
        # 1. System Alerts & Feedback
        if isinstance(msg, SystemMessage):
            self._handle_system_message(msg, node, live)
            
        # 2. Agent Output (Thinking & Tool Calls)
        if node == "agent" and isinstance(msg, (AIMessage, AIMessageChunk)):
            if msg.tool_calls:
                self._commit_printed_text(live)
                for tc in msg.tool_calls:
                    self._handle_tool_call(tc, live)
            
            if msg.content:
                chunk = self._extract_text_content(msg.content)
                self.full_text += chunk

        # 3. Tool Outputs
        elif node == "tools" and isinstance(msg, ToolMessage):
            self._handle_tool_result(msg, live)

    def _handle_system_message(self, msg: SystemMessage, node: str, live: Live):
        content = str(msg.content)
        
        if node == "tools":
            # Self-correction feedback
            error_preview = content.split('\n')[0]
            live.console.print(Padding(f"🔧 [bold magenta]Self-Correction:[/bold magenta] {error_preview}", (0, 0, 0, 4)))
            self.status_text = "Correcting strategy..."
        
        elif node == "token_budget_guard":
             live.console.print(Padding(f"💰 [bold red]Budget Alert:[/bold red] Context limit reached. Switching to wrap-up mode.", (0, 0, 0, 4)))
             self.status_text = "Budget exhausted..."

        elif node == "agent":
            # Quality Gate warnings
            warning_preview = content
            if len(warning_preview) > 100: warning_preview = warning_preview[:97] + "..."
            live.console.print(Padding(f"🛡️ [bold orange3]Quality Gate:[/bold orange3] {warning_preview}", (0, 0, 0, 4)))
            self.status_text = "Safety protocol triggered..."

    def _extract_text_content(self, content: Any) -> str:
        if isinstance(content, str): 
            return content
        if isinstance(content, list):
            return "".join(x.get("text", "") for x in content if isinstance(x, dict))
        return ""

    def _handle_tool_call(self, tc: Dict, live: Live):
        t_id, t_name = tc.get("id"), tc.get("name")
        args = tc.get("args", {})
        self.tool_buffer[t_id] = {"name": t_name, "args": args}

        arg_str = self._format_tool_args(args)
        arg_display = f" [dim]{arg_str}[/]" if arg_str else ""

        self.status_text = f"[bold cyan]Thinking:[/] {t_name}{arg_display}"
            
    def _handle_tool_result(self, msg: ToolMessage, live: Live):
        from rich.panel import Panel
    
        t_id = msg.tool_call_id
        content_str = str(msg.content)
        is_error = getattr(msg, "status", "") == "error" or content_str.startswith(("Error", "Ошибка"))
    
    # Цветовая схема по категориям инструментов для быстрой идентификации
        tool_meta = {
            "default": {"icon": "🔧", "color": "cyan"},
            "search": {"icon": "🔍", "color": "magenta"},
            "file": {"icon": "📄", "color": "blue"},
            "write": {"icon": "✏️", "color": "yellow"},
            "web": {"icon": "🌐", "color": "green"},
            "exec": {"icon": "⚡", "color": "red"}
        }
    
    # Определяем категорию
        category = "default"
        if msg.name:
            name_lower = msg.name.lower()
            if any(k in name_lower for k in ["search", "query", "find"]):
                category = "search"
            elif any(k in name_lower for k in ["read", "view", "list", "dir", "cat"]):
                category = "file"
            elif any(k in name_lower for k in ["write", "save", "edit", "create"]):
                category = "write"
            elif any(k in name_lower for k in ["web", "crawl", "fetch", "http"]):
                category = "web"
            elif any(k in name_lower for k in ["exec", "run", "bash", "shell", "cmd"]):
                category = "exec"
    
        style = tool_meta[category]
    
    # Выводим заголовок вызова (только если первый раз)
        if t_id in self.tool_buffer and t_id not in self.printed_tool_ids:
            info = self.tool_buffer[t_id]
            t_name = info["name"]
            args = info["args"]
            arg_str = self._format_tool_args(args)
        
        # Компактная строка: Иконка + Имя + Аргументы серым
            header = f"{style['icon']} [{style['color']}]{t_name}[/]"
            if arg_str:
                header += f" [dim]· {arg_str}[/]"
            
            live.console.print(Padding(header, (0, 0, 0, 2)))
            self.printed_tool_ids.add(t_id)
    
    # Получаем форматированный результат
        summary = format_tool_output(msg.name, content_str, is_error)
    
        if is_error:
        # Ошибки выделяются панелью с красной рамкой для максимальной заметности
        # format_tool_output уже возвращает текст с [red] разметкой
            error_panel = Panel(
                f"[bold]❌ {msg.name} failed[/]\n{summary}",
                border_style="red",
                padding=(0, 2),
                width=min(100, self.console.width - 6)
            )
            live.console.print(Padding(error_panel, (0, 0, 0, 4)))
        else:
        # Успешные результаты соединяются с заголовком через "└─" (tree-style)
        # Цвет соединителя совпадает с категорией инструмента
            connector = f"[{style['color']}]└─[/]"
            live.console.print(
                Padding(
                    f"{connector} {summary}",
                    (0, 0, 0, 4)
                )
            )
    
        self.status_text = "Analyzing..."
    
    def _format_tool_args(self, args: Any) -> str:
        arg_str = ""
        if isinstance(args, dict):
            # Prioritize common keys for display
            priority_keys = ["query", "queries", "path", "file_path", "url", "urls", "filename"]
            for key in priority_keys:
                if key in args:
                    val = args[key]
                    arg_str = str(val) if isinstance(val, list) else str(val)
                    break
            if not arg_str and args:
                # If no priority key, just take the first value
                arg_str = str(list(args.values())[0])
        elif isinstance(args, str):
            arg_str = args
            
        clean_arg = str(arg_str).strip().replace("\n", " ")
        if len(clean_arg) > 50: 
            return clean_arg[:47] + "..."
        return clean_arg

    def _extract_and_format_code(self, text: str) -> Group:
        """
        Заменяет markdown code blocks на Rich Syntax с подсветкой.
        Обычный текст рендерится как Markdown.
        """
        pattern = r'```(\w+)?\n(.*?)```'
        
        parts = []
        last_end = 0
        
        for match in re.finditer(pattern, text, re.DOTALL):
            # Добавляем текст до блока кода как Markdown
            if match.start() > last_end:
                pre_text = text[last_end:match.start()]
                if pre_text.strip():
                    parts.append(Markdown(pre_text))
            
            # Извлекаем язык и код
            lang = match.group(1) or "text"
            code = match.group(2)
            
            # Автоопределение языка для популярных случаев, если не указан явно
            if lang == "text" and code.strip():
                first_line = code.strip().split('\n')[0]
                if any(kw in first_line for kw in ['def ', 'class ', 'import ', 'print(']):
                    lang = "python"
                elif any(kw in first_line for kw in ['function', 'const ', 'let ', 'var ', '=>']):
                    lang = "javascript"
                elif '<' in first_line and '>' in first_line:
                    lang = "html"
                elif '{' in first_line and '}' in first_line:
                    lang = "json"
            
            # Добавляем блок кода с подсветкой синтаксиса
            syntax = Syntax(
                code, 
                lang, 
                theme="monokai", 
                line_numbers=True,
                word_wrap=True,
                padding=(1, 2)
            )
            parts.append(syntax)
            
            last_end = match.end()
        
        # Добавляем оставшийся текст
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining.strip():
                parts.append(Markdown(remaining))
        
        return Group(*parts) if parts else Group(Markdown(text))

    def _commit_printed_text(self, live: Optional[Live]):
        """Commits the accumulated text to the console."""
        _, clean_full, _ = parse_thought(self.full_text)
        
        if len(clean_full) > self.printed_len:
            new_text = clean_full[self.printed_len:]
            cleaned_chunk = clean_markdown_text(new_text)
            
            target = live.console if live else self.console
            # Используем новый метод с подсветкой синтаксиса вместо обычного Markdown
            formatted_content = self._extract_and_format_code(cleaned_chunk)
            target.print(Padding(formatted_content, (0, 0, 0, 2)))
            self.printed_len = len(clean_full)

    def _update_live_display(self, live: Live):
        """Updates the spinner text with the latest thought."""
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

def show_help(workflow: AgentWorkflow):
    """Displays the help menu."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Tool", style="green")
    table.add_column("Description")
    
    for t in workflow.tools:
        desc = t.description.split("\n")[0] if t.description else "No description"
        if len(desc) > 60: desc = desc[:57] + "..."
        table.add_row(t.name, desc)

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

async def main():
    os.system("cls" if os.name == "nt" else "clear")
    
    # Force load .env from the executable directory
    load_dotenv(BASE_DIR / '.env')
    
    console.print(Panel("[bold blue]AI Agent CLI[/]", subtitle="v6.5b"))

    # 1. Load Config
    try:
        temp_cfg = AgentConfig()
        token_budget = temp_cfg.token_budget
        
        # Set Debug Mode
        if temp_cfg.debug:
            logger.setLevel(logging.DEBUG)
            console.print("[yellow]🐛 Debug mode enabled (Internal Logs Visible)[/]")
        else:
            logger.setLevel(logging.WARNING)
            
    except Exception as e:
        console.print(f"[bold red]Config Error:[/] {e}")
        return

    # 2. Initialize Workflow
    try:
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

    # 3. Show Status Bar
    cfg = workflow.config
    console.print(
        f"[dim]Model:[/] [bold cyan]{cfg.gemini_model if cfg.provider == 'gemini' else cfg.openai_model}[/] "
        f"[dim]Temp:[/] [bold cyan]{cfg.temperature}[/] "
        f"[dim]Tools:[/] [bold cyan]{len(workflow.tools)}[/] "
        f"[dim]Budget:[/] [bold cyan]{token_budget}[/]"
    )
    console.print("[bold blue]Enter[/] [bold green]↵[/] — send  |  [bold blue]Alt+Enter[/] [bold yellow]⎇ ↵[/] — new line | [green]/tools[/] | [green]/help\n")

    # 4. Input Loop
    session = PromptSession(
        history=FileHistory(".history"),
        style=Style.from_dict({"prompt": "green"}),
        key_bindings=get_key_bindings(),
        lexer=PygmentsLexer(MarkdownLexer)
    )

    thread_id = "main_session"

    while True:
        try:
            cwd_name = Path.cwd().name
            user_input = await session.prompt_async(
                f".../{cwd_name}> ",
                placeholder=HTML('<gray>Type your message...</gray>')
            )
            user_input = user_input.strip()
            
            if not user_input: continue
            
            # Commands
            cmd = user_input.lower()
            if cmd in ["exit", "quit"]: break
            if cmd in ["clear", "reset"]:
                thread_id = f"session_{int(time.time())}"
                console.print("[yellow]♻ New session started[/]")
                continue
            
            if cmd == "/help" or cmd == "/tools":
                show_help(workflow)
                continue

            # Run Agent
            processor = StreamProcessor(console)
            await processor.run(
                agent_app, 
                user_input, 
                thread_id, 
                cfg.max_loops, 
                token_budget=token_budget 
            )
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
            # Clean up resources on exit (only if loop breaks)
            pass

    # Final Cleanup
    if hasattr(workflow, 'tool_registry') and workflow.tool_registry:
        await workflow.tool_registry.cleanup()
    
    # Close global network client if exists
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