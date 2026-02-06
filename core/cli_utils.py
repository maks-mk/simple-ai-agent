import re
from typing import Tuple, Dict, Any
from langchain_core.messages import AIMessage, AIMessageChunk
from prompt_toolkit.key_binding import KeyBindings

# --- OPTIONAL IMPORTS (Tiktoken) ---
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODER = None

# ======================================================
# TEXT PROCESSING
# ======================================================

_THOUGHT_RE = re.compile(r"<(thought|think)>(.*?)</\1>", re.DOTALL)

def clean_markdown_text(text: str) -> str:
    """Убирает лишние отступы и двойные переносы строк."""
    if not text: return text
    # Схлопываем множественные переносы
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Убираем пустую строку перед элементами списка - REMOVED to fix list rendering
    # text = re.sub(r'\n\s*\n(\s*[•\-\*]|\d+\.)', r'\n\1', text)
    return text

def parse_thought(text: str) -> Tuple[str, str, bool]:
    """Отделяет скрытые мысли <thought>/<think> от основного текста."""
    match = _THOUGHT_RE.search(text)
    if match: 
        return match.group(2).strip(), _THOUGHT_RE.sub('', text).strip(), True
    
    # Fallback for unclosed tags (streaming)
    for tag in ["<thought>", "<think>"]:
        if tag in text:
            close_tag = tag.replace("<", "</")
            if close_tag not in text:
                start = text.find(tag) + len(tag)
                return text[start:].strip(), text[:text.find(tag)], False
        
    return "", text, False

def format_tool_output(name: str, content: str, is_error: bool) -> str:
    """Форматирует результат инструмента для компактного вывода."""
    content = str(content).strip()
    if is_error: 
        # Smart Hints for common errors
        hint = ""
        lower_content = content.lower()
        if "401" in lower_content or "unauthorized" in lower_content:
            hint = " (Hint: Check your API keys in .env)"
        elif "not found" in lower_content and ("file" in lower_content or "dir" in lower_content):
            hint = " (Hint: Check path relative to workspace)"
        elif "disabled" in lower_content:
            hint = " (Hint: Check .env configuration)"
        elif "connection" in lower_content or "timeout" in lower_content:
            hint = " (Hint: Network issue, try again)"
            
        summary = f"{content[:120]}..." if len(content) > 120 else content
        return f"[red]{summary}[/][yellow italic]{hint}[/]"
    
    # Специальные форматтеры для разных типов инструментов
    if "web_search" in name: 
        count = content.count('http')
        return f"Found {count} results" if count > 0 else "No results found"
        
    elif "crawl_site" in name:
        # Extract page count and depth if available
        # Format: [Crawl completed. X pages processed. max_depth: Y]
        import re
        pages_match = re.search(r"(\d+) pages processed", content)
        depth_match = re.search(r"max_depth: (\d+)", content)
        
        pages = pages_match.group(1) if pages_match else "?"
        depth = depth_match.group(1) if depth_match else "?"
        
        if pages != "?" or depth != "?":
             return f"Crawled {pages} pages (depth: {depth})"
        return "Crawl completed"

    elif "list" in name and "directory" in name:
        lines = content.splitlines()
        count = len(lines)
        preview = ", ".join([l.strip() for l in lines[:3]])
        if count > 3:
            return f"Listed {count} items: {preview}, ..."
        return f"Listed {count} items: {preview}"

    elif "read" in name:
        size = len(content)
        lines = len(content.splitlines())
        return f"Read {lines} lines ({size} chars)"
        
    elif "write" in name or "save" in name: 
        return "File saved successfully"
    
    elif "delete" in name:
        return "Deleted successfully"

    # Default fallback
    clean_content = content.replace("\n", " ")
    return (clean_content[:80] + "...") if len(clean_content) > 80 else clean_content

def get_key_bindings():
    """Настройка Alt+Enter для переноса строки."""
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
# TOKEN TRACKER
# ======================================================

class TokenTracker:
    def __init__(self):
        self.max_input = 0
        self._usage_map: Dict[str, int] = {}  # msg_id -> output_tokens
        self._streaming_text = "" 
        self.source_label = "Est" 

    def update_from_message(self, msg: Any):
        """
        Обновляет счетчики на основе сообщения (чанка или полного).
        Использует гибридный подход: 
        1. Метаданные (если есть) - сохраняем по ID сообщения.
        2. Текст (streaming) - накапливаем для оценки.
        """
        # 1. Update Metadata
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            self._apply_metadata(msg.usage_metadata, getattr(msg, "id", None), msg)
        
        # 2. Accumulate Text for Estimation
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            content = msg.content
            chunk = ""
            if isinstance(content, str): chunk = content
            elif isinstance(content, list):
                chunk = "".join(x.get("text", "") for x in content if isinstance(x, dict))
            
            # Всегда накапливаем текст для независимой оценки
            self._streaming_text += chunk

    def update_from_node_update(self, update: Dict):
        """
        Обновляет счетчики из обновления состояния (обычно содержит полные сообщения с метаданными).
        """
        agent_data = update.get("agent")
        if not agent_data: return
        messages = agent_data.get("messages", [])
        if not isinstance(messages, list): messages = [messages]
        for msg in messages:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                self._apply_metadata(msg.usage_metadata, getattr(msg, "id", None), msg)

    def _apply_metadata(self, usage: Dict, msg_id: str = None, msg: Any = None):
        # Check source
        source = usage.get("token_source")
        if not source and msg and hasattr(msg, "additional_kwargs"):
            source = msg.additional_kwargs.get("token_source")

        if source == "Manual":
            self.source_label = "Manual"
        
        in_t = usage.get("input_tokens", 0)
        if in_t > self.max_input: self.max_input = in_t
        
        out_t = usage.get("output_tokens", 0)
        if out_t > 0:
            if msg_id:
                # Store usage per message ID.
                # Assuming cumulative updates from provider (standard for LangChain usage_metadata).
                # Even if delta, max() is safer than sum() for unknown behavior, 
                # but typically usage_metadata IS the total for that message.
                current = self._usage_map.get(msg_id, 0)
                if out_t > current:
                    self._usage_map[msg_id] = out_t
                    self.source_label = "Provider"
            else:
                # No ID (rare) - ignore to avoid double counting or implementation complexity
                pass

    def render(self, duration: float) -> str:
        # 1. Calculate Estimate
        est = 0
        if self._streaming_text:
            if _ENCODER:
                est = len(_ENCODER.encode(self._streaming_text))
            else:
                est = len(self._streaming_text) // 3
        
        # 2. Calculate Total Metadata Output
        total_output_metadata = sum(self._usage_map.values())
        
        # 3. Hybrid Decision Logic
        display_out = 0
        label = self.source_label

        if total_output_metadata > 0:
            # Если метаданные есть, но они подозрительно малы по сравнению с эстимейтом (например < 20%)
            # Это признак того, что провайдер вернул токены только за последний чанк/сообщение, а не за все.
            # (как в случае с Out: 4 при большом тексте)
            if est > 100 and total_output_metadata < (est * 0.2):
                 display_out = est
                 label = "Hybrid/Est"
            else:
                 display_out = total_output_metadata
                 label = "Provider"
        else:
            display_out = est
            label = "Est"

        return f"⏱ {duration:.1f}s | In: {self.max_input} Out: {display_out} [dim]({label})[/]"
