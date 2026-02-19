import asyncio
import logging
import shutil
import errno
from pathlib import Path
from langchain_core.tools import tool
from core.errors import format_error, ErrorType

logger = logging.getLogger(__name__)

def _validate_path(relative_path: str) -> tuple[bool, str | None, Path | None]:
    """Вспомогательная функция для проверки безопасности путей."""
    try:
        root_dir = Path.cwd()
        # 1. Проверка на абсолютный путь
        if Path(relative_path).is_absolute():
            return False, format_error(ErrorType.ACCESS_DENIED, f"Absolute paths not allowed: '{relative_path}'"), None

        # 2. Разрешаем путь относительно root_dir
        target_path = (root_dir / relative_path).resolve()
        
        # 3. Проверка на выход за пределы (Path Traversal)
        try:
            target_path.relative_to(root_dir)
        except ValueError:
            return False, format_error(ErrorType.ACCESS_DENIED, f"Path '{relative_path}' is outside working directory."), None

        return True, None, target_path
    except Exception as e:
        return False, format_error(ErrorType.VALIDATION, f"Path Error: {e}"), None

@tool("safe_delete_file")
async def safe_delete_file(file_path: str) -> str:
    """
    Deletes a file in the working directory.
    Args: file_path (relative).
    """
    is_valid, error, target = _validate_path(file_path)
    if not is_valid: 
        return error

    if not target.exists():
        return format_error(ErrorType.NOT_FOUND, f"File not found: {file_path}")
    if target.is_dir():
        return format_error(ErrorType.VALIDATION, f"{file_path} is a directory. Use safe_delete_directory.")

    try:
        # Use asyncio.to_thread for blocking I/O operations
        await asyncio.to_thread(target.unlink)
        logger.info(f"FILE DELETED: {target}")
        return f"Success: File {file_path} deleted."
    except Exception as e:
        logger.error(f"Delete File Error: {e}")
        return format_error(ErrorType.EXECUTION, str(e))

@tool("safe_delete_directory")
async def safe_delete_directory(dir_path: str, recursive: bool = False) -> str:
    """
    Deletes a directory in the working directory.
    Args: dir_path (relative), recursive (bool).
    """
    is_valid, error, target = _validate_path(dir_path)
    if not is_valid: 
        return error

    if not target.exists():
        return format_error(ErrorType.NOT_FOUND, f"Directory not found: {dir_path}")
    if not target.is_dir():
        return format_error(ErrorType.VALIDATION, f"{dir_path} is a file.")

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
            return format_error(ErrorType.VALIDATION, "Directory is not empty. Set recursive=True to delete it.")
        return format_error(ErrorType.EXECUTION, str(e))
    except Exception as e:
        logger.error(f"Delete Dir Error: {e}")
        return format_error(ErrorType.EXECUTION, str(e))
