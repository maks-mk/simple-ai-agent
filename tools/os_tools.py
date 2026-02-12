import logging
import asyncio
import subprocess
import shlex
import os
import platform
import httpx
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from tools.system_tools import get_net_client  # Импортируем клиент из system_tools

# Пробуем импортировать psutil
try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

# ==================================================
# PROCESS MANAGEMENT
# ==================================================

@tool("run_background_process")
async def run_background_process(command: str) -> str:
    """
    Starts a shell command in the background without waiting for it to finish.
    Use this for servers, bots, or any long-running scripts.
    Returns the PID (Process ID).
    """
    try:
        if platform.system() == "Windows":
            process = subprocess.Popen(
                command, 
                shell=True,
                creationflags=0x08000000 | 0x00000008, # CREATE_NO_WINDOW | DETACHED_PROCESS
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        else:
            args = shlex.split(command)
            process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        return f"Process started. PID: {process.pid}. Command: {command}"
    except Exception as e:
        return f"Failed to start process: {e}"

@tool("stop_background_process")
async def stop_background_process(pid: int) -> str:
    """
    Kills a background process tree by its PID (including children).
    Recommended for servers/bots started via run_background_process.
    """
    try:
        if psutil:
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                
                parent.terminate()
                gone, alive = psutil.wait_procs(children + [parent], timeout=3)
                for p in alive:
                    p.kill()
                    
                return f"Process tree {pid} terminated ({len(children)} children)."
            except psutil.NoSuchProcess:
                return f"Info: Process with PID {pid} not found (already terminated)."
        
        # Fallback if no psutil
        if platform.system() == "Windows":
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                 return f"Process tree {pid} terminated via taskkill."
            if "not found" in result.stderr.lower() or "не найден" in result.stderr.lower():
                return f"Info: Process {pid} not found."
            return f"Failed to kill {pid}: {result.stderr}"
        else:
            subprocess.run(["pkill", "-P", str(pid)], capture_output=True)
            result = subprocess.run(["kill", str(pid)], capture_output=True, text=True)
            if result.returncode == 0:
                 return f"Process {pid} terminated via kill."
            if "no such process" in result.stderr.lower():
                 return f"Info: Process {pid} not found."
            return f"Failed to kill {pid}: {result.stderr}"

    except Exception as e:
        return f"Failed to stop process: {e}"

@tool("find_process_by_port")
async def find_process_by_port(port: int) -> str:
    """
    Finds the PID of the process listening on a specific TCP port.
    Very useful if a previous server didn't close properly or PID was lost.
    """
    try:
        if psutil:
            try:
                for conn in psutil.net_connections(kind='inet'):
                    if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                        if conn.pid:
                            proc = psutil.Process(conn.pid)
                            return f"Found process on port {port}: PID={conn.pid}, Name='{proc.name()}'"
            except psutil.AccessDenied:
                return f"Error: Access denied when scanning ports."
            except Exception as e:
                logger.debug(f"psutil port scan failed: {e}")

        if platform.system() == "Windows":
             cmd = f"netstat -ano | findstr LISTENING | findstr :{port}"
             result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
             if result.stdout:
                 for line in result.stdout.strip().split('\n'):
                     parts = line.split()
                     if parts:
                         return f"Found process on port {port}: PID={parts[-1]} (via netstat)"
        else:
             cmd = f"lsof -i :{port} -sTCP:LISTEN -t"
             result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
             if result.stdout:
                 pid = result.stdout.strip().split('\n')[0]
                 return f"Found process on port {port}: PID={pid} (via lsof)"

        return f"No process found listening on port {port}."
    except Exception as e:
        return f"Error finding process: {e}"

# ==================================================
# FILE TRANSFER
# ==================================================

@tool("download_file")
async def download_file(url: str, filename: Optional[str] = None) -> str:
    """
    Downloads a file from a URL to the current working directory.
    Uses httpx for downloading.
    """
    try:
        if not filename:
            filename = url.split("/")[-1] or "downloaded_file"
        
        if os.path.sep in filename or (os.path.altsep and os.path.altsep in filename):
             return f"Error: Invalid filename '{filename}'."
             
        destination = Path.cwd() / filename
        client = get_net_client() # Используем клиент из system_tools
        logger.info(f"⬇️ Downloading {url} to {destination}")
        
        try:
            async with client.client.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > 500 * 1024 * 1024:
                    return "Error: File too large (>500MB). Download aborted."

                with open(destination, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
            return f"Success: File downloaded to {destination}"
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} - {e.response.reason_phrase}"
        except httpx.RequestError as e:
            return f"Error: Network request failed: {e}"
    except Exception as e:
        return f"Error downloading file: {e}"