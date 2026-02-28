import os
import re
import psutil
from pathlib import Path
from typing import Dict, Optional, Any
from core.errors import format_error, ErrorType

# Пре-компилируем регулярку для скорости
_PID_RE = re.compile(r"PID[:\s]+(\d+)")

def validate_tool_result(tool_name: str, args: Dict[str, Any], result: str) -> Optional[str]:
    """
    Validates the side effects of tool execution.
    Returns an error message if validation fails, else None.
    """
    if result.startswith("ERROR"):
        return None  # Already failed, no need to validate side effects

    try:
        # Группируем работу с созданием/редактированием файлов
        if tool_name in ("write_file", "edit_file"):
            path = args.get("path")
            if not path:
                return None
                
            p = Path(path).resolve()
            if not p.exists():
                return format_error(ErrorType.VALIDATION, f"File {path} not found or was not created.")
                
            if tool_name == "write_file":
                if p.stat().st_size == 0 and args.get("content"):
                    return format_error(ErrorType.VALIDATION, f"File {path} is empty after write.")
            else:  # edit_file
                new_string = args.get("new_string")
                if new_string and new_string not in p.read_text(encoding='utf-8'):
                    return format_error(ErrorType.VALIDATION, f"Edit failed: new content not found in {path}.")
        
        # Группируем операции удаления
        elif tool_name in ("safe_delete_file", "safe_delete_directory"):
             path = args.get("path")
             if path and Path(path).exists():
                 target_type = "File" if "file" in tool_name else "Directory"
                 return format_error(ErrorType.VALIDATION, f"{target_type} {path} still exists after deletion.")

        elif tool_name == "run_background_process":
            # Извлекаем PID используя скомпилированный regex
            match = _PID_RE.search(result)
            if match:
                pid = int(match.group(1))
                if not psutil.pid_exists(pid):
                     return format_error(ErrorType.VALIDATION, f"Process PID {pid} not found immediately after start.")

        elif tool_name == "cli_exec":
            # Резерв для будущей логики exit codes
            pass

    except Exception as e:
        return format_error(ErrorType.VALIDATION, f"Validation check failed: {e}")

    return None