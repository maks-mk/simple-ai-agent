import os
from pathlib import Path
import psutil
from typing import Dict, Optional, Any
from core.errors import format_error, ErrorType

def validate_tool_result(tool_name: str, args: Dict[str, Any], result: str) -> Optional[str]:
    """
    Validates the side effects of tool execution.
    Returns an error message if validation fails, else None.
    """
    if result.startswith("ERROR"):
        return None # Already failed, no need to validate side effects

    try:
        if tool_name == "write_file":
            path = args.get("path")
            if path:
                p = Path(path).resolve()
                if not p.exists():
                    return format_error(ErrorType.VALIDATION, f"File {path} was not created.")
                if p.stat().st_size == 0 and args.get("content"):
                     return format_error(ErrorType.VALIDATION, f"File {path} is empty after write.")

        elif tool_name == "edit_file":
            path = args.get("path")
            new_string = args.get("new_string")
            if path and new_string:
                p = Path(path).resolve()
                if not p.exists():
                     return format_error(ErrorType.VALIDATION, f"File {path} not found.")
                
                content = p.read_text(encoding='utf-8')
                if new_string not in content:
                    return format_error(ErrorType.VALIDATION, f"Edit failed: new content not found in {path}.")
        
        elif tool_name == "safe_delete_file":
             path = args.get("path")
             if path and Path(path).exists():
                 return format_error(ErrorType.VALIDATION, f"File {path} still exists after deletion.")

        elif tool_name == "safe_delete_directory":
             path = args.get("path")
             if path and Path(path).exists():
                 return format_error(ErrorType.VALIDATION, f"Directory {path} still exists after deletion.")

        elif tool_name == "run_background_process":
            # Extract PID from result "Success: Process started with PID 1234."
            import re
            match = re.search(r"PID[:\s]+(\d+)", result)
            if match:
                pid = int(match.group(1))
                if not psutil.pid_exists(pid):
                     return format_error(ErrorType.VALIDATION, f"Process PID {pid} not found immediately after start.")

        elif tool_name == "cli_exec":
            # Check for non-zero exit code in output if present
            # The tool output might contain "Exit Code: X"
            if "Exit Code:" in result and "Exit Code: 0" not in result:
                 # It's an execution error, likely already handled by the tool returning ERROR[EXECUTION]
                 # But if not, we can catch it here.
                 pass

    except Exception as e:
        return format_error(ErrorType.VALIDATION, f"Validation check failed: {e}")

    return None
