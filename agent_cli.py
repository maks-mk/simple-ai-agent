import os
import asyncio
import warnings
import time
import re
import logging
from typing import Dict, Tuple, Any, Set, Optional

# --- UI IMPORTS ---
from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.padding import Padding
from rich.text import Text

# --- PROMPT IMPORTS ---
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer
from prompt_toolkit.history import FileHistory

# --- LANGCHAIN IMPORTS ---
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk

# --- LOCAL IMPORTS ---
try:
    from agent import AgentWorkflow, logger
except ImportError:
    import sys
    sys.path.append(".")
    from agent import AgentWorkflow, logger

# --- OPTIONAL IMPORTS ---
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODER = None

# --- CONFIG ---
warnings.filterwarnings("ignore")
console = Console()
logging.getLogger("httpx").setLevel(logging.WARNING)

# ======================================================
# 1. TEXT PROCESSING UTILITIES
# ======================================================

_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)

def clean_markdown_text(text: str) -> str:
    """
    Убирает лишние отступы и двойные переносы строк перед списками.
    Решает проблему визуальных 'дыр' в Rich Markdown.
    """
    if not text: return text
    
    # 1. Схлопываем множественные переносы (оставляем максимум 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 2. Убираем пустую строку перед элементами списка (•, -, *, 1.)
    text = re.sub(r'\n\s*\n(\s*[•\-\*]|\d+\.)', r'\n\1', text)
    
    return text

def parse_thought(text: str) -> Tuple[str, str, bool]:
    """Отделяет скрытые мысли <thought> от основного текста."""
    match = _THOUGHT_RE.search(text)
    if match: 
        return match.group(1).strip(), _THOUGHT_RE.sub('', text).strip(), True
    
    if "<thought>" in text and "</thought>" not in text:
        start = text.find("<thought>") + len("<thought>")
        return text[start:].strip(), text[:text.find("<thought>")], False
        
    return "", text, False

# ======================================================
# 2. UI UTILITIES
# ======================================================

class TokenTracker:
    def __init__(self):
        self.max_input = 0
        self.total_output = 0
        self._seen_ids = set()
        self._streaming_text = "" 

    def update_from_message(self, msg: Any):
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            self._apply_metadata(msg.usage_metadata, getattr(msg, "id", None))
        
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            content = msg.content
            chunk = ""
            if isinstance(content, str): chunk = content
            elif isinstance(content, list):
                chunk = "".join(x.get("text", "") for x in content if isinstance(x, dict))
            
            if isinstance(msg, AIMessageChunk): self._streaming_text += chunk
            elif not msg.usage_metadata: self._streaming_text = chunk

    def update_from_node_update(self, update: Dict):
        agent_data = update.get("agent")
        if not agent_data: return
        messages = agent_data.get("messages", [])
        if not isinstance(messages, list): messages = [messages]
        for msg in messages:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                self._apply_metadata(msg.usage_metadata, getattr(msg, "id", None))

    def _apply_metadata(self, usage: Dict, msg_id: str = None):
        is_new = True
        if msg_id and msg_id in self._seen_ids: is_new = False
        
        in_t = usage.get("input_tokens", 0)
        if in_t > self.max_input: self.max_input = in_t
        
        out_t = usage.get("output_tokens", 0)
        if out_t > 0:
            if is_new:
                self.total_output += out_t
                if msg_id: self._seen_ids.add(msg_id)
                self._streaming_text = ""

    def render(self, duration: float) -> str:
        display_out = self.total_output
        if self._streaming_text:
            est = len(_ENCODER.encode(self._streaming_text)) if _ENCODER else len(self._streaming_text) // 3
            display_out += est
        return f"⏱ {duration:.1f}s | In: {self.max_input} Out: {display_out}"

def format_tool_output(name: str, content: str, is_error: bool) -> str:
    content = str(content).strip()
    if is_error: 
        return f"[red]{content[:120]}...[/]" if len(content) > 120 else f"[red]{content}[/]"
    
    if "web_search" in name: return f"Found {content.count('http')} results"
    elif "fetch" in name or "read" in name: return f"Loaded {len(content)} chars"
    elif "write" in name or "save" in name: return "File saved successfully"
    elif "list" in name: return f"Listed {len(content.splitlines())} items"
    
    return (content[:80] + "...") if len(content) > 80 else content

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

# ======================================================
# 3. STREAM PROCESSOR (STABLE LOGIC)
# ======================================================

class StreamProcessor:
    """Стабильный процессор стриминга. Не теряет текст, так как использует накопление."""
    
    def __init__(self):
        self.tracker = TokenTracker()
        self.full_text = ""          # Весь текст ответа целиком
        self.printed_len = 0         # Сколько символов мы уже вывели "навечно"
        self.printed_tool_ids = set()
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
                    await asyncio.sleep(0.005) # Даем время Rich обновиться

                    # 1. ОБНОВЛЕНИЯ ОТ УЗЛОВ (Конец шага)
                    if mode == "updates":
                        self.tracker.update_from_node_update(payload)
                        # Шаг завершен: безопасно печатаем весь накопленный текст
                        self._commit_printed_text(live)

                    # 2. ПОТОК СООБЩЕНИЙ (Стриминг токенов)
                    elif mode == "messages":
                        msg, metadata = payload
                        node = metadata.get("langgraph_node")
                        self.tracker.update_from_message(msg)

                        if node == "agent" and isinstance(msg, (AIMessage, AIMessageChunk)):
                            # Если модель решила вызвать инструмент - сначала печатаем весь текст до этого момента
                            if msg.tool_calls:
                                self._commit_printed_text(live)
                                for tc in msg.tool_calls:
                                    self._handle_tool_call(tc, live)
                            
                            # Накапливаем текст
                            if msg.content:
                                chunk = msg.content if isinstance(msg.content, str) else ""
                                if isinstance(msg.content, list):
                                    chunk = "".join(x.get("text", "") for x in msg.content if isinstance(x, dict))
                                
                                # Простое накопление. Merge здесь не нужен, так как LangGraph не дублирует стрим.
                                self.full_text += chunk

                        elif node == "tools" and isinstance(msg, ToolMessage):
                            self._handle_tool_result(msg, live)
                            
                    # Обновляем "живой" хвост текста (то, что еще не запечатано)
                    self._update_live_display(live)

        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[bold red]🛑 Stopped by user[/]")
            return 

        # Финальный вывод остатка
        self._commit_printed_text(None) # None = печать в основную консоль
        console.print(self.tracker.render(time.time() - self.start_time), justify="right")

    def _handle_tool_call(self, tc, live):
        t_id, t_name = tc.get("id"), tc.get("name")
        if t_id and t_name and t_id not in self.printed_tool_ids:
            # Выводим уведомление о туле
            live.console.print(Padding(f"🌍 [bold cyan]Call:[/] {t_name}", (0, 0, 0, 2)))
            self.printed_tool_ids.add(t_id)
            self.status_text = f"[bold cyan]Calling:[/] {t_name}"

    def _handle_tool_result(self, msg, live):
        content_str = str(msg.content)
        is_error = getattr(msg, "status", "") == "error" or content_str.startswith(("Error", "Ошибка"))
        icon = "❌" if is_error else "✅"
        color = "red" if is_error else "green"
        summary = format_tool_output(msg.name, content_str, is_error)
        
        live.console.print(Padding(f"[{color}]{icon} {msg.name}:[/] [dim]{summary}[/]", (0, 0, 0, 4)))
        self.status_text = "Analyzing..."

    def _commit_printed_text(self, live: Optional[Live]):
        """
        Берет накопившийся текст, чистит его от тегов <thought>
        и печатает ту часть, которая еще не была напечатана.
        """
        _, clean_full, _ = parse_thought(self.full_text)
        
        # Если есть новый текст для печати
        if len(clean_full) > self.printed_len:
            new_text = clean_full[self.printed_len:]
            
            # Чистим Markdown (убираем лишние отступы)
            cleaned_chunk = clean_markdown_text(new_text)
            
            # Печатаем
            target = live.console if live else console
            target.print(Padding(Markdown(cleaned_chunk), (0, 0, 0, 2)))
            
            self.printed_len = len(clean_full)

    def _update_live_display(self, live: Live):
        """Показывает только статус (спиннер) и последние несколько слов."""
        _, clean_full, _ = parse_thought(self.full_text)
        
        # Обновляем текст статуса из <thought> тегов
        thought_match = _THOUGHT_RE.search(self.full_text)
        if thought_match:
            thought_content = thought_match.group(1).strip()
            self.status_text = f"[yellow italic]{thought_content[-60:]}...[/]"
        
        # Хвост, который еще не запечатан.
        # Это то, что пользователь видит "в процессе набора".
        pending = clean_full[self.printed_len:]
        
        renderable = Spinner("dots", text=self.status_text, style="cyan")
        
        if pending.strip():
             renderable = Group(
                Padding(Markdown(clean_markdown_text(pending)), (0, 0, 0, 2)),
                renderable
             )
            
        live.update(renderable)

# ======================================================
# 4. MAIN LOOP
# ======================================================

async def main():
    os.system("cls" if os.name == "nt" else "clear")
    console.print(Panel("[bold blue]AI Agent CLI[/]", subtitle="v4.5b"))

    # Suppress Logs during init
    prev_level = logger.getEffectiveLevel()
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
        logger.setLevel(prev_level)
        
    # Info Block
    cfg = workflow.config
    console.print(
        f"[dim]Model:[/] [bold cyan]{cfg.gemini_model if cfg.provider == 'gemini' else cfg.openai_model}[/] "
        f"[dim]Temp:[/] [bold cyan]{cfg.temperature}[/] "
        f"[dim]Tools:[/] [bold cyan]{len(workflow.tools)}[/] "
    )
    console.print("[bold blue]Enter[/] [bold green]↵[/] — send  |  [bold blue]Alt+Enter[/] [bold yellow]⎇ ↵[/] — new line\n")

    # Prompt Session
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

            processor = StreamProcessor()
            await processor.run(agent_app, user_input, thread_id, cfg.max_loops)
            console.print()

        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]Cancelled. Type 'exit' to quit.[/]")
            continue
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            import traceback
            logger.debug(traceback.format_exc())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass