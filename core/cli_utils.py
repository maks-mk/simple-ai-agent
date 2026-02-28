import re
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from langchain_core.messages import AIMessage, AIMessageChunk
from prompt_toolkit.key_binding import KeyBindings

# ======================================================
# PRE-COMPILED REGEXES (Optimization)
# ======================================================
_THOUGHT_RE = re.compile(r"<(thought|think)>(.*?)</\1>", re.DOTALL)
_CLEAN_MD_RE = re.compile(r'\n{3,}')
_CRAWL_PAGES_RE = re.compile(r"(\d+) pages processed")
_CRAWL_DEPTH_RE = re.compile(r"max_depth: (\d+)")

# ======================================================
# TEXT PROCESSING
# ======================================================

def truncate_value(value: str, max_length: int = 60) -> str:
    """Truncate a string value if it exceeds max_length."""
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value

def abbreviate_path(path_str: str, max_length: int = 60) -> str:
    """Abbreviate a file path intelligently - show basename or relative path."""
    try:
        path = Path(path_str)
        # If it's just a filename (no directory parts), return as-is
        if len(path.parts) == 1:
            return path_str

        # Try to get relative path from current working directory
        try:
            rel_str = str(path.relative_to(Path.cwd()))
            # Use relative if it's shorter and not too long
            if len(rel_str) < len(path_str) and len(rel_str) <= max_length:
                return rel_str
        except (ValueError, OSError):
            pass

        # If absolute path is reasonable length, use it
        if len(path_str) <= max_length:
            return path_str
    except Exception:
        pass
    
    # Fallback: just show basename (filename only) if path is too long
    return truncate_value(path_str, max_length)

def format_tool_display(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """
    Format tool calls for display with tool-specific smart formatting.
    Based on deepagents-cli UI.
    """
    # Tool-specific formatting - show the most important argument(s)
    if tool_name in {"read_file", "write_file", "edit_file", "Read", "Write", "SearchReplace", "tail_file"}:
        path_value = tool_args.get("file_path") or tool_args.get("path")
        if path_value:
            return f"{tool_name}({abbreviate_path(str(path_value))})"

    elif tool_name in {"web_search", "WebSearch"}:
        if "query" in tool_args:
            return f'{tool_name}("{truncate_value(str(tool_args["query"]), 80)}")'

    elif tool_name in {"grep", "Grep", "glob", "Glob", "search_in_file", "search_in_directory", "find_file"}:
        if "pattern" in tool_args:
            pattern_val = tool_args.get("pattern") or tool_args.get("name_pattern")
            return f'{tool_name}("{truncate_value(str(pattern_val), 70)}")'

    elif tool_name in {"execute", "RunCommand", "cli_exec"}:
        if "command" in tool_args:
            return f'{tool_name}("{truncate_value(str(tool_args["command"]), 100)}")'

    elif tool_name in {"ls", "LS", "list_directory"}:
        if tool_args.get("path"):
            return f"{tool_name}({abbreviate_path(str(tool_args['path']))})"
        return f"{tool_name}()"

    elif tool_name in {"fetch_url", "WebFetch", "fetch_content", "download_file"}:
        url_val = tool_args.get("url") or tool_args.get("urls")
        if url_val:
            return f'{tool_name}("{truncate_value(str(url_val), 80)}")'

    # Fallback: generic formatting
    args_str = ", ".join(f"{k}={truncate_value(str(v), 50)}" for k, v in tool_args.items())
    return f"{tool_name}({args_str})"

def clean_markdown_text(text: str) -> str:
    """Убирает лишние отступы и двойные переносы строк."""
    if not text: return text
    # Используем пре-скомпилированный regex для скорости
    return _CLEAN_MD_RE.sub('\n\n', text)

def parse_thought(text: str) -> Tuple[str, str, bool]:
    """Отделяет скрытые мысли <thought>/<think> от основного текста."""
    # 1. Сначала удаляем ВСЕ полностью завершенные мысли (чтобы они не ломали флаг)
    clean_text = _THOUGHT_RE.sub('', text)
    
    # 2. Проверяем, есть ли сейчас открытая (незавершенная) мысль
    for tag in ("<thought>", "<think>"):
        start_idx = clean_text.find(tag)
        if start_idx != -1:
            close_tag = tag.replace("<", "</")
            if close_tag not in clean_text:
                content_start = start_idx + len(tag)
                # Возвращаем: (текущая мысль, чистый текст до нее, флаг "идет мысль")
                return clean_text[content_start:].strip(), clean_text[:start_idx], True
                
    return "", clean_text.strip(), False
    
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
    name_lower = name.lower()
    
    if "web_search" in name_lower: 
        count = content.count('http')
        return f"Found {count} results" if count > 0 else "No results found"
        
    elif "crawl_site" in name_lower:
        # Используем пре-скомпилированные регулярные выражения
        pages_match = _CRAWL_PAGES_RE.search(content)
        depth_match = _CRAWL_DEPTH_RE.search(content)
        
        pages = pages_match.group(1) if pages_match else "?"
        depth = depth_match.group(1) if depth_match else "?"
        
        if pages != "?" or depth != "?":
             return f"Crawled {pages} pages (depth: {depth})"
        return "Crawl completed"
        
    elif "cli_exec" in name_lower or "shell" in name_lower:
        # Разбиваем вывод на строки и убираем пустые
        lines =[line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return "Command executed (no output)"
        
        # Берем первую значимую строку для превью
        first_line = lines[0].replace("[stderr]", "").strip()
        preview = first_line[:60] + "..." if len(first_line) > 60 else first_line
        
        # Если строк много, добавляем счетчик
        if len(lines) > 1:
            return f"{preview} [dim](+{len(lines)-1} lines)[/]"
        return preview

    elif "list" in name_lower and "directory" in name_lower:
        lines = content.splitlines()
        count = len(lines)
        preview = ", ".join([l.strip() for l in lines[:3]])
        if count > 3:
            return f"Listed {count} items: {preview}, ..."
        return f"Listed {count} items: {preview}"

    elif "read" in name_lower:
        return f"Read {len(content.splitlines())} lines ({len(content)} chars)"
        
    elif "write" in name_lower or "save" in name_lower: 
        return "File saved successfully"
        
    elif "edit_file" in name_lower:
        return "File edited successfully"
    
    elif "delete" in name_lower:
        return "Deleted successfully"
        
    elif "fetch" in name_lower or "download" in name_lower:
        return f"Fetched content ({len(content)} chars)"

    return content[:150] + "..." if len(content) > 150 else content

def format_exception_friendly(e: Exception) -> str:
    """Возвращает читаемое сообщение об ошибке вместо traceback."""
    err_str = str(e)
    err_type = type(e).__name__
    
    # 1. Rate Limits
    if "429" in err_str or "RateLimit" in err_type or "QuotaExceeded" in err_type or "ResourceExhausted" in err_type:
        return "⚠ Rate Limit Exceeded (429). Please wait a moment or check your API quota."
    
    # 2. Auth Errors
    if "401" in err_str or "403" in err_str or "Authentication" in err_type:
        return "⚠ Authentication Failed. Check your API KEY in .env."
    
    # 3. Context Length
    if "context_length_exceeded" in err_str or "too many tokens" in err_str.lower():
        return "⚠ Context Limit Reached. Use 'reset' to start fresh."
    
    # 4. Network
    if "ConnectError" in err_type or "Timeout" in err_type or "ReadTimeout" in err_type:
        return "⚠ Network Error. Connection failed or timed out."

    # 5. Fallback: Shorten extremely long messages (often JSON dumps)
    if len(err_str) > 300:
        return f"⚠ Error ({err_type}): {err_str[:300]}...[truncated]"
        
    return f"⚠ Error ({err_type}): {err_str}"

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
    __slots__ = ('max_input', 'total_output', '_streaming_len', '_seen_msg_ids')

    def __init__(self):
        self.max_input = 0
        self.total_output = 0
        self._streaming_len = 0
        self._seen_msg_ids: set = set()  # Дедупликация metadata по id сообщения

    def update_from_message(self, msg: Any):
        """Updates counters based on message content or metadata."""
        
        # 1. ВСЕГДА считаем длину контента, даже если есть метаданные
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            content = msg.content
            chunk_len = 0
            if isinstance(content, str): 
                chunk_len = len(content)
            elif isinstance(content, list):
                # Faster sum via generator expression
                chunk_len = sum(len(x.get("text", "")) for x in content if isinstance(x, dict))
            
            # Для потока (чанков) суммируем. Для целого сообщения заменяем (если поток не работал)
            if isinstance(msg, AIMessageChunk):
                self._streaming_len += chunk_len
            elif self._streaming_len == 0:
                self._streaming_len = chunk_len

        # 2. Применяем метаданные, если они есть
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            self._apply_metadata(msg.usage_metadata, msg_id=getattr(msg, "id", None))

    def update_from_node_update(self, update: Dict):
        """Updates counters from state updates (final messages with metadata)."""
        agent_data = update.get("agent")
        if not agent_data: return
        
        messages = agent_data.get("messages",[])
        if not isinstance(messages, list): 
            messages = [messages]
            
        for msg in messages:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                self._apply_metadata(msg.usage_metadata, msg_id=getattr(msg, "id", None))

    def _apply_metadata(self, usage: Dict, msg_id: str = None):
        # Дедупликация: одно и то же сообщение приходит через messages и updates
        if msg_id:
            if msg_id in self._seen_msg_ids:
                return
            self._seen_msg_ids.add(msg_id)
        
        in_t = usage.get("input_tokens", 0)
        if in_t > self.max_input: 
            self.max_input = in_t
        
        out_t = usage.get("output_tokens", 0)
        # Суммируем output_tokens (каждый вызов LLM генерирует новые токены)
        self.total_output += out_t

    def render(self, duration: float) -> str:
        display_out = self.total_output
        
        # ЗАЩИТА ОТ БАГОВ ПРОВАЙДЕРА (Xiaomi mimo-v2-flash и другие)
        # Агент мог сделать несколько шагов, тогда API вернет 2-3 токена. 
        # Проверяем "плотность" токенов. В норме 1 токен = 3-4 символа.
        # Если API заявляет, что 1 токен вместил больше 10 символов текста - это баг API.
        if self._streaming_len > 10 and display_out < (self._streaming_len // 10):
            display_out = self._streaming_len // 3
        
        in_display = str(self.max_input) if self.max_input > 0 else "?"
        
        return f"[dim]• {duration:.1f}s   In: {in_display}   Out: {display_out}[/]"
        