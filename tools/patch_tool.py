import logging
from pathlib import Path
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool("smart_replace")
def smart_replace(path: str, target_text: str, replacement_text: str) -> str:
    """
    Smartly replaces a specific text snippet in a file with new text.
    Use this tool for editing code or text files when you need to change specific blocks.
    It is more robust than standard 'edit_file' as it handles line-ending differences automatically.
    
    Args:
        path: Relative path to the file.
        target_text: The exact text block you want to replace (copy it from the file).
        replacement_text: The new text to insert in place of target_text.
    """
    try:
        file_path = Path(path).resolve()
        
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
        return f"System Error during edit: {e}"