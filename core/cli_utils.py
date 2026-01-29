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

_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)

def clean_markdown_text(text: str) -> str:
    """Убирает лишние отступы и двойные переносы строк."""
    if not text: return text
    # Схлопываем множественные переносы
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Убираем пустую строку перед элементами списка
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
        self.total_output = 0
        self._seen_ids = set()
        self._streaming_text = "" 
        self.source_label = "Provider" 

    def update_from_message(self, msg: Any):
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            self._apply_metadata(msg.usage_metadata, getattr(msg, "id", None), msg)
        
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
                self._apply_metadata(msg.usage_metadata, getattr(msg, "id", None), msg)

    def _apply_metadata(self, usage: Dict, msg_id: str = None, msg: Any = None):
        # DEBUG: Print incoming usage
        #print(f"DEBUG: Token Update | ID: {msg_id} | Usage: {usage}")
        
        is_new = True
        if msg_id and msg_id in self._seen_ids: is_new = False
        
        # Check source in usage_metadata OR additional_kwargs
        source = usage.get("token_source")
        if not source and msg and hasattr(msg, "additional_kwargs"):
            source = msg.additional_kwargs.get("token_source")

        if source == "Manual":
            self.source_label = "Manual"
        
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
            
        return f"⏱ {duration:.1f}s | In: {self.max_input} Out: {display_out} [dim]({self.source_label})[/]"