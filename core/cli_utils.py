import re
from typing import Tuple, Dict, Any
from langchain_core.messages import AIMessage, AIMessageChunk
from prompt_toolkit.key_binding import KeyBindings

# ======================================================
# TEXT PROCESSING
# ======================================================

_THOUGHT_RE = re.compile(r"<(thought|think)>(.*?)</\1>", re.DOTALL)

def truncate_value(value: str, max_length: int = 60) -> str:
    """Truncate a string value if it exceeds max_length."""
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value

def abbreviate_path(path_str: str, max_length: int = 60) -> str:
    """Abbreviate a file path intelligently - show basename or relative path."""
    from pathlib import Path
    try:
        path = Path(path_str)
        # If it's just a filename (no directory parts), return as-is
        if len(path.parts) == 1:
            return path_str

        # Try to get relative path from current working directory
        try:
            rel_path = path.relative_to(Path.cwd())
            rel_str = str(rel_path)
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
    if tool_name in {"read_file", "write_file", "edit_file", "Read", "Write", "SearchReplace"}:
        # File operations: show the primary file path argument
        path_value = tool_args.get("file_path") or tool_args.get("path")
        if path_value:
            path = abbreviate_path(str(path_value))
            return f"{tool_name}({path})"

    elif tool_name in {"web_search", "WebSearch"}:
        # Web search: show the query string
        if "query" in tool_args:
            query = truncate_value(str(tool_args["query"]), 80)
            return f'{tool_name}("{query}")'

    elif tool_name in {"grep", "Grep", "glob", "Glob"}:
        # Search tools: show pattern
        if "pattern" in tool_args:
            pattern = truncate_value(str(tool_args["pattern"]), 70)
            return f'{tool_name}("{pattern}")'

    elif tool_name in {"execute", "RunCommand"}:
        # Execute: show the command
        if "command" in tool_args:
            command = truncate_value(str(tool_args["command"]), 100)
            return f'{tool_name}("{command}")'

    elif tool_name in {"ls", "LS"}:
        # ls: show directory, or empty if current directory
        if tool_args.get("path"):
            path = abbreviate_path(str(tool_args["path"]))
            return f"{tool_name}({path})"
        return f"{tool_name}()"

    elif tool_name in {"fetch_url", "WebFetch"}:
        if "url" in tool_args:
            url = truncate_value(str(tool_args["url"]), 80)
            return f'{tool_name}("{url}")'

    # Fallback: generic formatting
    args_str = ", ".join(
        f"{k}={truncate_value(str(v), 50)}" for k, v in tool_args.items()
    )
    return f"{tool_name}({args_str})"

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
        
    elif "cli_exec" in name:
        # Разбиваем вывод на строки и убираем пустые
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        
        if not lines:
            return "Command executed (no output)"
        
        # Берем первую значимую строку для превью
        first_line = lines[0]
        # Если в первой строке есть "[stderr]", убираем его для красоты
        first_line = first_line.replace("[stderr]", "").strip()
        
        # Обрезаем, если длинная
        preview = first_line[:60] + "..." if len(first_line) > 60 else first_line
        
        # Если строк много, добавляем счетчик
        if len(lines) > 1:
            return f"{preview} [dim](+{len(lines)-1} lines)[/]"
        
        return preview

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
        
    elif "edit_file" in name:
        return "File edited successfully"
    
    elif "delete" in name:
        return "Deleted successfully"
        
    elif "fetch" in name or "download" in name:
        size = len(content)
        return f"Fetched content ({size} chars)"

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
        return f"⚠ Error ({err_type}): {err_str[:300]}... [truncated]"
        
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
    def __init__(self):
        self.max_input = 0
        self.total_output = 0
        self._streaming_len = 0 

    def update_from_message(self, msg: Any):
        """
        Updates counters based on message content or metadata.
        """
        # 1. Prefer Metadata if available
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            self._apply_metadata(msg.usage_metadata)
            return

        # 2. Fallback: Estimate from content length (for streaming)
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            content = msg.content
            chunk_len = 0
            if isinstance(content, str): 
                chunk_len = len(content)
            elif isinstance(content, list):
                chunk_len = sum(len(x.get("text", "")) for x in content if isinstance(x, dict))
            
            self._streaming_len += chunk_len

    def update_from_node_update(self, update: Dict):
        """
        Updates counters from state updates (final messages with metadata).
        """
        agent_data = update.get("agent")
        if not agent_data: return
        messages = agent_data.get("messages", [])
        if not isinstance(messages, list): messages = [messages]
        for msg in messages:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                self._apply_metadata(msg.usage_metadata)

    def _apply_metadata(self, usage: Dict):
        in_t = usage.get("input_tokens", 0)
        if in_t > self.max_input: self.max_input = in_t
        
        # Assume output_tokens in metadata is the total for the response
        out_t = usage.get("output_tokens", 0)
        if out_t > self.total_output:
            self.total_output = out_t

    def render(self, duration: float) -> str:
        # Display: Prefer exact metadata, else estimate (chars / 3)
        display_out = self.total_output
        if display_out == 0 and self._streaming_len > 0:
            display_out = self._streaming_len // 3
        
        in_display = str(self.max_input) if self.max_input > 0 else "?"
        
        return f"[dim]• {duration:.1f}s   In: {in_display}   Out: {display_out}[/]"
