import subprocess
import asyncio
import os
import platform
from typing import Optional
from langchain_core.tools import tool

from core.utils import truncate_output
from core.errors import format_error, ErrorType
from core.safety_policy import SafetyPolicy

# Лимиты для защиты контекста
DEFAULT_TIMEOUT = 120

# Global safety policy
_SAFETY_POLICY: Optional[SafetyPolicy] = None

def set_safety_policy(policy: SafetyPolicy):
    global _SAFETY_POLICY
    _SAFETY_POLICY = policy

@tool("cli_exec")
async def cli_exec(command: str) -> str:
    """
    Executes a shell command on the host machine.
    WARNING: Use with caution. This tool runs commands directly in the OS shell.
    Supports pipe (|), redirects (>), and chain operators (&&).
    
    Args:
        command: The shell command to execute (e.g., 'ls -la', 'git status', 'npm install').
    """
    # Check if shell allowed
    if _SAFETY_POLICY and not _SAFETY_POLICY.allow_shell:
        return format_error(ErrorType.ACCESS_DENIED, "Shell execution is disabled by SafetyPolicy.")

    return await asyncio.to_thread(_run_shell_command, command)

def _run_shell_command(command: str) -> str:
    if not command.strip():
        return format_error(ErrorType.VALIDATION, "Command cannot be empty.")

    # Определяем shell в зависимости от ОС
    is_windows = platform.system() == "Windows"
    
    try:
        # Запускаем процесс
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT,
            cwd=os.getcwd() # Всегда выполняем в текущей рабочей директории
        )

        # Формируем вывод
        output_parts = []
        
        # Stdout
        if result.stdout:
            output_parts.append(result.stdout.strip())
            
        # Stderr (помечаем явно, если он есть)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr.strip()}")

        output = "\n".join(output_parts)

        # Если вывода нет
        if not output and result.returncode == 0:
            output = "Command executed successfully (no output)."
        elif not output and result.returncode != 0:
            output = "Command failed with no output."

        # Добавляем код возврата, если ошибка
        if result.returncode != 0:
            output = format_error(ErrorType.EXECUTION, f"Command failed with Exit Code {result.returncode}.\nOutput:\n{output}")

        # Обрезаем, если слишком длинно (защита контекста LLM)
        limit = _SAFETY_POLICY.max_tool_output if _SAFETY_POLICY else 5000
        output = truncate_output(output, limit, source="shell")

        return output

    except subprocess.TimeoutExpired:
        return format_error(ErrorType.TIMEOUT, f"Command timed out after {DEFAULT_TIMEOUT} seconds.")
    except Exception as e:
        return format_error(ErrorType.EXECUTION, f"Error executing command: {str(e)}")
