import asyncio
import os
from typing import Optional
from langchain_core.tools import tool

from core.utils import truncate_output
from core.errors import format_error, ErrorType
from core.safety_policy import SafetyPolicy

# Константы
DEFAULT_TIMEOUT = 120

# Глобальные настройки
_SAFETY_POLICY: Optional[SafetyPolicy] = None
_WORKING_DIRECTORY: str = os.getcwd()  # По умолчанию текущая папка процесса

def set_safety_policy(policy: SafetyPolicy):
    """Sets the global safety policy for shell execution."""
    global _SAFETY_POLICY
    _SAFETY_POLICY = policy

def set_working_directory(cwd: str):
    """
    Syncs the shell's working directory with the FilesystemManager's workspace.
    Call this when initializing the agent to ensure tools look at the same folders.
    """
    global _WORKING_DIRECTORY
    _WORKING_DIRECTORY = cwd

@tool("cli_exec")
async def cli_exec(command: str) -> str:
    """
    Executes a shell command on the host machine.
    
    IMPORTANT RULES FOR LLM:
    1. STATELESSNESS: Commands are stateless. `cd folder` in one call will NOT affect the next call. 
       If you need to change directories, chain commands: e.g., `cd folder && npm install`.
    2. NO INTERACTIVE COMMANDS: DO NOT run commands that require user input (e.g., `nano`, `vim`, `python` without args, `less`, `top`). 
       They will hang until timeout!
    3. BACKGROUND TASKS: Do not run blocking servers (e.g., `npm start`, `python -m http.server`) unless you background them or use an appropriate tool.
    4. LONG SCRIPTS: For complex logic, write a script file using `write_file` and then execute it.
    
    Supports pipe (|), redirects (>), and chain operators (&&).
    
    Args:
        command: The shell command to execute (e.g., 'ls -la', 'git status').
    """
    if _SAFETY_POLICY and not _SAFETY_POLICY.allow_shell:
        return format_error(ErrorType.ACCESS_DENIED, "Shell execution is disabled by SafetyPolicy.")

    if not command.strip():
        return format_error(ErrorType.VALIDATION, "Command cannot be empty.")

    try:
        # Используем нативный asyncio.subprocess вместо потоков
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=_WORKING_DIRECTORY
        )

        try:
            # Ждем выполнения с таймаутом
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), 
                timeout=DEFAULT_TIMEOUT
            )
        except asyncio.TimeoutError:
            # Если команда зависла (например, LLM запустила `nano`), жестко убиваем процесс
            try:
                process.kill()
            except OSError:
                pass
            return format_error(
                ErrorType.TIMEOUT, 
                f"Command timed out after {DEFAULT_TIMEOUT} seconds. Did you run an interactive command (like nano/vim) or a blocking server?"
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        output_parts =[]
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"[stderr]\n{stderr}")

        output = "\n".join(output_parts)
        
        if process.returncode != 0:
            error_msg = f"Command failed with Exit Code {process.returncode}."
            if output:
                error_msg += f"\nOutput:\n{output}"
            else:
                error_msg += " (No output)"
            return format_error(ErrorType.EXECUTION, error_msg)

        if not output:
            output = "Command executed successfully (no output)."
        
        limit = _SAFETY_POLICY.max_tool_output if _SAFETY_POLICY else 5000
        return truncate_output(output, limit, source="shell")

    except Exception as e:
        return format_error(ErrorType.EXECUTION, f"Error executing command: {str(e)}")