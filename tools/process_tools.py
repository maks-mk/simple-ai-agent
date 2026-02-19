import asyncio
import logging
import subprocess
import shlex
import psutil
import platform
import atexit
from typing import Dict, Any, Optional
from langchain_core.tools import tool

from core.safety_policy import SafetyPolicy
from core.errors import format_error, ErrorType

logger = logging.getLogger(__name__)

# Global registry for background processes
# Format: {pid: Popen_object}
_BACKGROUND_PROCESSES: Dict[int, subprocess.Popen] = {}

# Global safety policy
_SAFETY_POLICY: Optional[SafetyPolicy] = None

def set_safety_policy(policy: SafetyPolicy):
    global _SAFETY_POLICY
    _SAFETY_POLICY = policy

def _cleanup_zombies():
    """Removes finished processes from registry."""
    to_remove = []
    for pid, proc in _BACKGROUND_PROCESSES.items():
        if proc.poll() is not None:
            to_remove.append(pid)
    for pid in to_remove:
        del _BACKGROUND_PROCESSES[pid]

def _shutdown_all():
    """Kills all tracked processes on exit."""
    for pid, proc in _BACKGROUND_PROCESSES.items():
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except:
            proc.kill()

atexit.register(_shutdown_all)

@tool("run_background_process")
def run_background_process(command: str, cwd: str = ".") -> str:
    """
    Starts a process in the background. Returns PID.
    Useful for starting servers, watchers, or long-running tasks.
    
    Args:
        command: The shell command to execute.
        cwd: Working directory (default ".").
    """
    # Clean up dead processes first
    _cleanup_zombies()
    
    # Check limit
    limit = _SAFETY_POLICY.max_background_processes if _SAFETY_POLICY else 5
    if len(_BACKGROUND_PROCESSES) >= limit:
        return format_error(ErrorType.LIMIT_EXCEEDED, f"Maximum background processes ({limit}) reached.")

    try:
        # Split command safely
        if platform.system() == "Windows":
            # On Windows, we often need shell=True for complex commands, 
            # but for simple executables, shlex might not be perfect.
            # Using shell=True for flexibility in agent context.
            process = subprocess.Popen(
                command, 
                shell=True, 
                cwd=cwd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        else:
            args = shlex.split(command)
            process = subprocess.Popen(
                args, 
                cwd=cwd,
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                start_new_session=True # Detach from parent
            )
        
        pid = process.pid
        _BACKGROUND_PROCESSES[pid] = process
        
        logger.info(f"Background process started: {command} (PID: {pid})")
        return f"Success: Process started with PID {pid}."
        
    except Exception as e:
        return format_error(ErrorType.EXECUTION, f"Error starting process: {e}")

@tool("stop_background_process")
def stop_background_process(pid: int) -> str:
    """
    Stops a background process by PID.
    """
    _cleanup_zombies()
    
    try:
        if pid in _BACKGROUND_PROCESSES:
            proc = _BACKGROUND_PROCESSES[pid]
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            
            if pid in _BACKGROUND_PROCESSES:
                del _BACKGROUND_PROCESSES[pid]
            return f"Success: Process {pid} stopped."
            
        # Try to kill by PID even if not in our registry (e.g. found via find_process)
        if psutil.pid_exists(pid):
            try:
                p = psutil.Process(pid)
                p.terminate()
                return f"Success: External process {pid} terminated."
            except psutil.AccessDenied:
                return format_error(ErrorType.ACCESS_DENIED, f"Cannot terminate process {pid}.")
            
        return format_error(ErrorType.NOT_FOUND, f"Process {pid} not found.")
        
    except Exception as e:
        return format_error(ErrorType.EXECUTION, f"Error stopping process: {e}")

@tool("find_process_by_port")
def find_process_by_port(port: int) -> str:
    """
    Finds a process PID listening on a specific port.
    """
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        return f"Found process '{proc.name()}' (PID: {proc.pid}) on port {port}."
            except (psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return format_error(ErrorType.NOT_FOUND, f"No process found listening on port {port}.")
    except Exception as e:
        return format_error(ErrorType.EXECUTION, f"Error searching process: {e}")
