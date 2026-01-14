import os
import asyncio
import warnings
import time
import re
from typing import Dict, Tuple, Optional, Set, Any
import logging

from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.padding import Padding
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer
from prompt_toolkit.history import FileHistory

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk

try:
    from agent import AgentWorkflow, logger
except ImportError:
    import sys
    sys.path.append(".")
    from agent import AgentWorkflow, logger

# Импорт tiktoken для эвристики UI
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODER = None

warnings.filterwarnings("ignore")
console = Console()

_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)

# ======================================================
# TOKEN TRACKER (Версия 3.0 - Updates + Metadata)
# ======================================================
class TokenTracker:
    def __init__(self):
        self.max_input = 0
        self.total_output = 0
        self._seen_ids = set()
        self._streaming_text = "" 

    def update_from_message(self, msg: Any):
        """Обновление из стрима сообщений"""
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            self._apply_metadata(msg.usage_metadata, getattr(msg, "id", None))
        
        # Эвристика для live-режима
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            content = msg.content
            chunk = ""
            if isinstance(content, str):
                chunk = content
            elif isinstance(content, list):
                chunk = "".join(x.get("text", "") for x in content if isinstance(x, dict))
            
            if isinstance(msg, AIMessageChunk):
                self._streaming_text += chunk
            elif not msg.usage_metadata:
                self._streaming_text = chunk

    def update_from_node_update(self, update: Dict):
        """Обновление из результата узла (Гарантированные данные)"""
        agent_data = update.get("agent")
        if not agent_data: return

        messages = agent_data.get("messages", [])
        if not isinstance(messages, list): messages = [messages]

        for msg in messages:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                self._apply_metadata(msg.usage_metadata, getattr(msg, "id", None))

    def _apply_metadata(self, usage: Dict, msg_id: str = None):
        is_new = True
        if msg_id and msg_id in self._seen_ids:
            is_new = False
        
        in_t = usage.get("input_tokens", 0)
        if in_t > self.max_input:
            self.max_input = in_t
        
        out_t = usage.get("output_tokens", 0)
        if out_t > 0:
            if is_new:
                self.total_output += out_t
                if msg_id: self._seen_ids.add(msg_id)
                self._streaming_text = ""

    def render(self, duration: float) -> str:
        display_out = self.total_output
        if self._streaming_text:
            est = 0
            if _ENCODER: est = len(_ENCODER.encode(self._streaming_text))
            else: est = len(self._streaming_text) // 3
            display_out += est

        txt = f"⏱ {duration:.1f}s"
        txt += f" | In: {self.max_input} Out: {display_out}"
        return f"[bright_black]{txt}[/]"

# ======================================================
# UI UTILS
# ======================================================
def get_key_bindings():
    kb = KeyBindings()
    @kb.add('enter')
    def _(event):
        buf = event.current_buffer
        if not buf.text.strip(): return
        buf.validate_and_handle()
    @kb.add('escape', 'enter')
    def _(event):
        event.current_buffer.insert_text("\n")
    return kb

def parse_thought(text: str) -> Tuple[str, str, bool]:
    match = _THOUGHT_RE.search(text)
    if match:
        return match.group(1).strip(), _THOUGHT_RE.sub('', text).strip(), True
    if "<thought>" in text and "</thought>" not in text:
        start = text.find("<thought>") + len("<thought>")
        return text[start:].strip(), text[:text.find("<thought>")], False
    return "", text, False

def print_padded_markdown(text: str, padding: tuple = (1, 1)):
    if not text.strip(): return
    clean_text = re.sub(r'\n{3,}', '\n\n', text)
    console.print(Padding(Markdown(clean_text), padding))

def format_tool_output(name: str, content: str, is_error: bool) -> str:
    """UX Magic: Делает вывод инструмента кратким и понятным."""
    content = str(content).strip()
    
    if is_error:
        # Если ошибка короткая, показываем целиком, иначе обрезаем
        return f"[red]{content[:120]}...[/]" if len(content) > 120 else f"[red]{content}[/]"
    
    if "web_search" in name:
        count = content.count("http")
        return f"Found {count} results"
    
    elif "fetch" in name or "read" in name:
        size = len(content)
        return f"Loaded {size} chars"
    
    elif "write" in name or "save" in name:
        return "File saved successfully"
    
    elif "list" in name:
        items = len(content.split("\n"))
        return f"Listed {items} items"
        
    # Fallback: показываем начало текста
    preview = (content[:80] + "...") if len(content) > 80 else content
    return preview.replace("\n", " ")

# ======================================================
# STREAM LOOP
# ======================================================
async def process_stream(agent_app, user_input: str, thread_id: str, max_loops: int = 25):
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_loops * 4
    }
    
    tracker = TokenTracker()
    start_time = time.time()
    accumulated_text = ""
    printed_thoughts = set()
    printed_tool_ids = set()
    
    spinner_status = "Thinking..."
    
    try:
        # Используем transient=True, чтобы спиннер исчезал после завершения
        with Live(Spinner("dots", text=spinner_status, style="cyan"), 
                  refresh_per_second=12, 
                  console=console, 
                  transient=True) as live:
            
            async for mode, payload in agent_app.astream(
                {"messages": [HumanMessage(content=user_input)], "steps": 0},
                config=config,
                stream_mode=["messages", "updates"]
            ):
                # ВАЖНО: Даем время циклу событий отрисовать анимацию
                await asyncio.sleep(0.001)

                # --- UPDATE TRACKER ---
                if mode == "updates":
                    tracker.update_from_node_update(payload)
                    # Обновляем UI (текст + спиннер), чтобы показать актуальный статус
                    renderable = Group(
                        Padding(Markdown(accumulated_text or ""), (0, 1)),
                        Spinner("dots", text=spinner_status, style="cyan")
                    )
                    live.update(renderable)
                    continue

                # --- MESSAGE STREAM ---
                if mode == "messages":
                    msg, metadata = payload
                    node = metadata.get("langgraph_node")
                    
                    tracker.update_from_message(msg)

                    # 1. АГЕНТ (МЫСЛИ + ВЫЗОВЫ)
                    if node == "agent" and isinstance(msg, (AIMessage, AIMessageChunk)):
                        # A. Инструменты (Запрос)
                        if msg.tool_calls:
                            if accumulated_text.strip():
                                 _, clean, _ = parse_thought(accumulated_text)
                                 if clean.strip():
                                     live.console.print(Padding(Markdown(clean), (0, 1)))
                                     accumulated_text = ""

                            for tc in msg.tool_calls:
                                t_id = tc.get("id")
                                t_name = tc.get("name")
                                if t_id and t_name and t_id not in printed_tool_ids:
                                    live.console.print(Padding(f"🌍 [bold cyan]Call:[/] {t_name}", (0, 0, 0, 2)))
                                    printed_tool_ids.add(t_id)
                                    spinner_status = f"[bold cyan]Calling:[/] {t_name}"
                        
                        # B. Текст (Мысли/Ответ)
                        if msg.content:
                            chunk = msg.content if isinstance(msg.content, str) else ""
                            if isinstance(msg.content, list):
                                chunk = "".join(x.get("text", "") for x in msg.content if isinstance(x, dict))

                            if isinstance(msg, AIMessageChunk):
                                accumulated_text += chunk
                            else:
                                if not accumulated_text: accumulated_text = chunk
                            
                            thought, clean_text, is_complete = parse_thought(accumulated_text)
                            
                            if thought:
                                spinner_status = f"[yellow italic]{thought}...[/]"
                                if is_complete and thought not in printed_thoughts:
                                    live.console.print(Padding(f"➤ [italic yellow]{thought}[/]", (0, 0, 0, 2)))
                                    printed_thoughts.add(thought)
                                    accumulated_text = clean_text
                            elif clean_text.strip() and "<thought>" not in accumulated_text:
                                spinner_status = "Typing..."
                                # Рендерим Группу: Текст сверху, Спиннер снизу
                                pretty_md = re.sub(r'\n{3,}', '\n\n', clean_text)
                                live.update(Group(
                                    Padding(Markdown(pretty_md), (1, 1)),
                                    Spinner("dots", text=spinner_status, style="cyan")
                                ))
                                continue

                    # 2. ИНСТРУМЕНТЫ (ОТВЕТЫ - UX FIX)
                    elif node == "tools" and isinstance(msg, ToolMessage):
                        content_str = str(msg.content)
                        is_error = False
                        
                        if getattr(msg, "status", "") == "error":
                            is_error = True
                        elif content_str.startswith(("Error", "Ошибка")):
                            is_error = True
                            
                        icon = "❌" if is_error else "✅"
                        color = "red" if is_error else "green"
                        
                        summary = format_tool_output(msg.name, content_str, is_error)
                        
                        live.console.print(Padding(f"[{color}]{icon} {msg.name}:[/] [dim]{summary}[/]", (0, 0, 0, 4)))
                        spinner_status = "Analyzing..."

                # Обновляем дефолтное состояние (если не попали в ветку с текстом)
                if accumulated_text.strip():
                     _, clean_text, _ = parse_thought(accumulated_text)
                     pretty_md = re.sub(r'\n{3,}', '\n\n', clean_text)
                     renderable = Group(
                        Padding(Markdown(pretty_md), (1, 1)),
                        Spinner("dots", text=spinner_status, style="cyan")
                     )
                else:
                    renderable = Spinner("dots", text=spinner_status, style="cyan")
                
                live.update(renderable)

    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[bold red]🛑 Stopped by user[/]")
        return 

    # Финальный вывод
    _, final_clean, _ = parse_thought(accumulated_text)
    if final_clean.strip():
        print_padded_markdown(final_clean, padding=(0, 1, 1, 1))
    
    console.print(tracker.render(time.time() - start_time), justify="right")

# ======================================================
# MAIN
# ======================================================
async def main():
    os.system("cls" if os.name == "nt" else "clear")
    console.print(Panel("[bold blue]AI Agent CLI[/]", subtitle="v3.4"))

    previous_level = logger.getEffectiveLevel()
    logger.setLevel(logging.WARNING)

    try:
        with console.status("[bold green]Initializing system...[/]", spinner="dots"):
            workflow = AgentWorkflow()
            await workflow.initialize_resources()
            agent_app = workflow.build_graph()
        console.print("[bold green]System Ready![/]")

    except Exception as e:
        console.print(f"[bold red]Init Error:[/] {e}")
        return
    finally:
        logger.setLevel(previous_level)
        
    model = workflow.config.gemini_model if workflow.config.provider == "gemini" else workflow.config.openai_model
    tools = len(workflow.tools)
    max_loops = workflow.config.max_loops

    console.print(f"[dim]Model:[/] [bold cyan]{model}[/] [dim]Tools:[/] [bold cyan]{tools}[/] [dim]Max Loops:[/] [bold cyan]{max_loops}[/]")
    console.print("[bold blue]Enter[/] [bold green]↵[/] — send  |  [bold blue]Alt+Enter[/] [bold yellow]⎇↵[/] — new line\n")

    session = PromptSession(
        history=FileHistory(".history"),
        style=Style.from_dict({"prompt": "bold cyan"}),
        key_bindings=get_key_bindings(),
        lexer=PygmentsLexer(MarkdownLexer)
    )

    thread_id = "main_session"

    while True:
        try:
            user_input = await session.prompt_async("You > ")
            user_input = user_input.strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break
            if user_input.lower() in ["clear", "reset"]:
                thread_id = f"session_{int(time.time())}"
                console.print("[yellow]♻ New session started[/]")
                continue

            await process_stream(agent_app, user_input, thread_id, max_loops=max_loops)
            console.print()

        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]Cancelled. Type 'exit' to quit.[/]")
            continue
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass