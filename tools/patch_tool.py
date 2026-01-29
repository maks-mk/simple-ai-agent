import logging
import asyncio
from pathlib import Path
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

def _smart_replace_sync(path: str, target_text: str, replacement_text: str) -> str:
    try:
        file_path = Path(path).resolve()
        
        # --- Security Check (Path Traversal) ---
        if not file_path.is_relative_to(Path.cwd()):
            return f"ACCESS DENIED: Path '{path}' is outside working directory."
        # ---------------------------------------
        
        if not file_path.exists(): return f"Error: File '{path}' not found."
        if not file_path.is_file(): return f"Error: '{path}' is not a file."
            
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return "Error: File binary or unknown encoding."

        # Попытка 1: Точное совпадение
        if target_text in content:
            new_content = content.replace(target_text, replacement_text)
            file_path.write_text(new_content, encoding='utf-8')
            return "Success: Text replaced (exact match)."

        # Попытка 2: Нормализация переносов
        content_norm = content.replace('\r\n', '\n')
        target_norm = target_text.replace('\r\n', '\n')
        
        if target_norm in content_norm:
            new_content_norm = content_norm.replace(target_norm, replacement_text)
            file_path.write_text(new_content_norm, encoding='utf-8')
            return "Success: Text replaced (normalized line endings)."

        snippet = target_text[:50].replace('\n', '\\n')
        return f"Error: Could not find target text starting with: '{snippet}...'"

    except Exception as e:
        return f"Error: System Error during edit: {e}"

@tool("smart_replace")
async def smart_replace(path: str, target_text: str, replacement_text: str) -> str:
    """
    Precise search-and-replace for files. Best for surgical code edits or config updates.
    Handles line-ending differences (LF/CRLF) automatically to prevent errors.
    
    Args:
        path: Relative path to the file.
        target_text: Unique text block to find.
        replacement_text: New text to insert.
    """
    return await asyncio.to_thread(_smart_replace_sync, path, target_text, replacement_text)