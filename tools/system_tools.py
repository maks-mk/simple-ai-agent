import logging
import httpx
import ipaddress
import platform
import shutil
import socket
import asyncio
import subprocess
import shlex
import os
from typing import Any, Dict, Optional, Literal
from dataclasses import dataclass
from langchain_core.tools import tool

# Пробуем импортировать psutil
try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

# ==================================================
# 1. CORE LOGIC (Network Client)
# ==================================================

@dataclass
class APIError:
    type: Literal["network", "timeout", "http_error", "invalid_input", "rate_limited", "unknown"]
    message: str
    retryable: bool
    hint: Optional[str] = None

@dataclass
class Result:
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[APIError] = None

class NetworkClient:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self.client.headers.update({
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)" 
        })

    async def my_ip(self) -> Result:
        # Пытаемся получить JSON с ip.sb
        result = await self._request("https://api.ip.sb/jsonip")
        if result.ok: return result
        # Fallback на ipify
        return await self._request("https://api.ipify.org?format=json")

    async def get_ip_info(self, ip: str) -> Result:
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            return Result(ok=False, error=APIError("invalid_input", f"Invalid IP: {ip}", False, "Provide valid IP"))
        return await self._request(f"https://api.ip.sb/geoip/{ip}")

    async def _request(self, url: str) -> Result:
        try:
            response = await self.client.get(url)
            if response.status_code == 429:
                return Result(ok=False, error=APIError("rate_limited", "Too many requests", True))
            response.raise_for_status()
            return Result(ok=True, data=response.json())
        except Exception as e:
            return Result(ok=False, error=APIError("network", str(e), True))
            
    async def close(self):
        await self.client.aclose()

# Глобальный клиент для переиспользования в CLI
_net_client: Optional[NetworkClient] = None

def get_net_client() -> NetworkClient:
    global _net_client
    if _net_client is None:
        _net_client = NetworkClient()
    return _net_client

def _format_result(result: Result) -> str:
    if result.ok:
        return "\n".join([f"- {k}: {v}" for k, v in result.data.items()])
    return f"Error: {result.error.message}"

# ==================================================
# 2. NETWORK TOOLS
# ==================================================

@tool("get_public_ip")
async def get_public_ip() -> str:
    """Gets public IPv4/IPv6 address and ISP info."""
    client = get_net_client()
    result = await client.my_ip()
    return _format_result(result)

@tool("lookup_ip_info")
async def lookup_ip_info(ip: str) -> str:
    """Looks up geolocation, ASN, and ISP details for a given IP address."""
    client = get_net_client()
    result = await client.get_ip_info(ip)
    return _format_result(result)

# ==================================================
# 3. HARDWARE & SYSTEM TOOLS
# ==================================================

def _get_system_info_sync() -> str:
    try:
        uname = platform.uname()
        total, used, free = shutil.disk_usage(".")
        
        info = [
            f"OS: {uname.system} {uname.release}",
            f"Machine: {uname.machine}",
            f"Disk: {free // (2**30)}GB free / {total // (2**30)}GB total"
        ]
        
        if psutil:
            ram = psutil.virtual_memory()
            cpu_load = psutil.cpu_percent(interval=0.1)
            info.append(f"RAM: {ram.percent}% used ({round(ram.total / 1024**3, 1)}GB total)")
            info.append(f"CPU: {cpu_load}% load ({psutil.cpu_count()} cores)")
        return "\n".join(info)
    except Exception as e:
        return f"Error: {e}"

@tool("get_system_info")
async def get_system_info() -> str:
    """Returns real-time OS, CPU, RAM and Disk usage statistics."""
    return await asyncio.to_thread(_get_system_info_sync)

@tool("get_local_network_info")
async def get_local_network_info() -> str:
    """Lists local IP addresses, hostnames, and active network interfaces."""
    def _sync():
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            res = [f"Hostname: {hostname}", f"Local IP: {local_ip}", "\nInterfaces:"]
            if psutil:
                for intf, addrs in psutil.net_if_addrs().items():
                    ips = [a.address for a in addrs if a.family == socket.AF_INET]
                    if ips: res.append(f" - {intf}: {', '.join(ips)}")
            return "\n".join(res)
        except Exception as e: return str(e)
    return await asyncio.to_thread(_sync)

# ==================================================
# 4. PROCESS MANAGEMENT
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
            # CREATE_NO_WINDOW = 0x08000000
            # DETACHED_PROCESS = 0x00000008
            process = subprocess.Popen(
                command, 
                shell=True,
                creationflags=0x08000000 | 0x00000008,
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
                # Kill children first (recursive)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                
                # Kill parent
                parent.terminate()
                gone, alive = psutil.wait_procs(children + [parent], timeout=3)
                
                # Force kill if still alive
                for p in alive:
                    p.kill()
                    
                return f"Process tree {pid} terminated ({len(children)} children)."
            except psutil.NoSuchProcess:
                return f"Info: Process with PID {pid} not found (already terminated)."
        
        # Fallback if no psutil
        if platform.system() == "Windows":
            # Используем taskkill /T (Tree)
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                 return f"Process tree {pid} terminated via taskkill."
            
            if "not found" in result.stderr.lower() or "не найден" in result.stderr.lower():
                return f"Info: Process {pid} not found (already terminated)."
            
            return f"Failed to kill {pid}: {result.stderr}"
        else:
             # Используем kill (no tree support in basic kill, fallback to pkill -P if needed, but for now simple kill)
             # Better: try pkill -P first
            subprocess.run(["pkill", "-P", str(pid)], capture_output=True)
            result = subprocess.run(
                ["kill", str(pid)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                 return f"Process {pid} terminated via kill."
            
            if "no such process" in result.stderr.lower():
                 return f"Info: Process {pid} not found (already terminated)."
                 
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
            # Проверяем все типы IP-соединений
            try:
                for conn in psutil.net_connections(kind='inet'):
                    if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                        if conn.pid:
                            proc = psutil.Process(conn.pid)
                            return f"Found process on port {port}: PID={conn.pid}, Name='{proc.name()}'"
            except psutil.AccessDenied:
                return f"Error: Access denied when scanning ports. (Hint: Try running with higher privileges)"
            except Exception as e:
                logger.debug(f"psutil port scan failed: {e}")

        # Fallback (Windows)
        if platform.system() == "Windows":
             # Находим строку с портом и статусом LISTENING, вырезаем PID (последняя колонка)
             cmd = f"netstat -ano | findstr LISTENING | findstr :{port}"
             result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
             if result.stdout:
                 for line in result.stdout.strip().split('\n'):
                     parts = line.split()
                     if parts:
                         return f"Found process on port {port}: PID={parts[-1]} (via netstat)"
             
        # Fallback (Linux/Mac)
        else:
             # lsof -t возвращает только чистый PID
             cmd = f"lsof -i :{port} -sTCP:LISTEN -t"
             result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
             if result.stdout:
                 pid = result.stdout.strip().split('\n')[0]
                 return f"Found process on port {port}: PID={pid} (via lsof)"

        return f"No process found listening on port {port}."
    except Exception as e:
        return f"Error finding process: {e}"