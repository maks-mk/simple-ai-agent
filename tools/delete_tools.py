import asyncio
import logging
import shutil
import errno
from pathlib import Path
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

def _validate_path(relative_path: str) -> tuple[bool, str | None, Path | None]:
    """Вспомогательная функция для проверки безопасности путей."""
    try:
        root_dir = Path.cwd()
        # 1. Проверка на абсолютный путь
        if Path(relative_path).is_absolute():
            return False, f"ACCESS DENIED: Absolute paths not allowed: '{relative_path}'", None

        # 2. Разрешаем путь относительно root_dir
        target_path = (root_dir / relative_path).resolve()
        
        # 3. Проверка на выход за пределы (Path Traversal)
        if not target_path.is_relative_to(root_dir):
            return False, f"ACCESS DENIED: Path '{relative_path}' is outside working directory.", None

        return True, None, target_path
    except Exception as e:
        return False, f"Path Error: {e}", None

@tool("safe_delete_file")
async def safe_delete_file(file_path: str) -> str:
    """
    Safely deletes a file within the working directory.
    Args:
        file_path: Relative path to the file to delete.
    """
    is_valid, error, target = _validate_path(file_path)
    if not is_valid: 
        return f"Error: {error}"

    if not target.exists():
        return f"Error: File not found: {file_path}"
    if target.is_dir():
        return f"Error: {file_path} is a directory. Use safe_delete_directory."

    try:
        # Use asyncio.to_thread for blocking I/O operations
        await asyncio.to_thread(target.unlink)
        logger.info(f"FILE DELETED: {target}")
        return f"Success: File {file_path} deleted."
    except Exception as e:
        logger.error(f"Delete File Error: {e}")
        return f"Error: {str(e)}"

@tool("safe_delete_directory")
async def safe_delete_directory(dir_path: str, recursive: bool = False) -> str:
    """
    Safely deletes a directory. 
    Args:
        dir_path: Relative path to the directory.
        recursive: Set True to delete non-empty directories.
    """
    is_valid, error, target = _validate_path(dir_path)
    if not is_valid: 
        return f"Error: {error}"

    if not target.exists():
        return f"Error: Directory not found: {dir_path}"
    if not target.is_dir():
        return f"Error: {dir_path} is a file."

    try:
        if recursive:
            await asyncio.to_thread(shutil.rmtree, target)
            msg = f"Directory {dir_path} deleted recursively."
        else:
            await asyncio.to_thread(target.rmdir)
            msg = f"Empty directory {dir_path} deleted."
        
        logger.info(f"DIR DELETED: {target}")
        return f"Success: {msg}"

    except OSError as e:
        if e.errno == errno.ENOTEMPTY:
            return "Error: Directory is not empty. Set recursive=True to delete it."
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Delete Dir Error: {e}")
        return f"Error: {str(e)}"