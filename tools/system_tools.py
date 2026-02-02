import logging
import httpx
import ipaddress
import platform
import shutil
import socket
import asyncio
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
# 1. CORE LOGIC (Network Client - Async)
# ==================================================

ErrorType = Literal["network", "timeout", "http_error", "invalid_input", "invalid_response", "rate_limited", "unknown"]

@dataclass
class APIError:
    type: ErrorType
    message: str
    retryable: bool
    hint: Optional[str] = None

@dataclass
class Result:
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[APIError] = None

class NetworkClient:
    """
    Robust network client with fallback providers using httpx (Async).
    """
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.client.headers.update({
            "Accept": "application/json",
            "User-Agent": "curl/7.68.0" 
        })

    async def my_ip(self) -> Result:
        result = await self._request("https://api.ip.sb/jsonip")
        if result.ok:
            return result
        logger.warning(f"Primary IP API failed ({result.error.message}), trying fallback...")
        return await self._request("https://api.ipify.org?format=json")

    async def get_ip_info(self, ip: str) -> Result:
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            return Result(ok=False, error=APIError("invalid_input", f"Invalid IP: {ip}", False, "Provide valid IPv4/IPv6"))
        return await self._request(f"https://api.ip.sb/geoip/{ip}")

    async def _request(self, url: str) -> Result:
        try:
            response = await self.client.get(url)
            if response.status_code == 429:
                return Result(ok=False, error=APIError("rate_limited", "Rate limit exceeded", True))
            if response.is_error:
                return Result(ok=False, error=APIError("http_error", f"HTTP {response.status_code}", True))
            return Result(ok=True, data=response.json())
        except httpx.TimeoutException:
            return Result(ok=False, error=APIError("timeout", f"Timeout connecting to {url}", True))
        except Exception as e:
            return Result(ok=False, error=APIError("network", str(e), True))
            
    async def close(self):
        await self.client.aclose()

_net_client: Optional[NetworkClient] = None

def get_net_client() -> NetworkClient:
    global _net_client
    if _net_client is None:
        _net_client = NetworkClient()
    return _net_client

def _format_result(result: Result) -> str:
    if result.ok:
        info = "\n".join([f"- {k}: {v}" for k, v in result.data.items()])
        return f"Success:\n{info}"
    else:
        err = result.error
        return f"Error ({err.type}): {err.message}. Hint: {err.hint}"

# ==================================================
# 2. EXPORTED TOOLS (Async Wrappers)
# ==================================================

@tool("get_public_ip")
async def get_public_ip() -> str:
    """
    Gets public IP address with provider fallback.
    """
    client = get_net_client()
    result = await client.my_ip()
    return _format_result(result)

@tool("lookup_ip_info")
async def lookup_ip_info(ip: str) -> str:
    """
    Gets geolocation/ISP info for an IP.
    """
    client = get_net_client()
    result = await client.get_ip_info(ip)
    return _format_result(result)

# ==================================================
# 3. EXPORTED TOOLS (System / Hardware)
# ==================================================

def _get_system_info_sync() -> str:
    """Внутренняя синхронная функция сбора данных."""
    try:
        uname = platform.uname()
        python_ver = platform.python_version()
        
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        total_gb = total // (2**30)
        
        info = [
            f"OS: {uname.system} {uname.release} ({uname.version})",
            f"Architecture: {uname.machine}",
            f"Python Version: {python_ver}",
            f"Disk Space (CWD): {free_gb} GB free / {total_gb} GB total"
        ]
        
        if psutil:
            ram = psutil.virtual_memory()
            total_ram_gb = round(ram.total / (1024**3), 1)
            info.append(f"RAM: {ram.percent}% used (Total: {total_ram_gb} GB)")
            
            cpu_freq = psutil.cpu_freq()
            freq_str = f" @ {cpu_freq.current:.0f}Mhz" if cpu_freq else ""
            # interval=0.2 блокирует поток на 0.2с, поэтому async здесь критичен
            cpu_load = psutil.cpu_percent(interval=0.2)
            cores = psutil.cpu_count(logical=False)
            
            info.append(f"CPU: {cpu_load}% load{freq_str} ({cores} phys cores)")
        else:
            info.append("Extended hardware info unavailable (install 'psutil')")

        return "\n".join(info)
    except Exception as e:
        return f"Error getting system info: {e}"

@tool("get_system_info")
async def get_system_info() -> str:
    """
    Gets OS, CPU, RAM, and Disk usage stats.
    """
    return await asyncio.to_thread(_get_system_info_sync)

def _get_local_network_sync() -> str:
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        lines = [
            f"Hostname: {hostname}",
            f"Primary Local IP: {local_ip}"
        ]

        if psutil:
            net_io = psutil.net_io_counters()
            sent_mb = net_io.bytes_sent // (1024**2)
            recv_mb = net_io.bytes_recv // (1024**2)
            lines.append(f"Total Traffic: Sent {sent_mb} MB, Received {recv_mb} MB")
            
            lines.append("\n--- Interfaces ---")
            stats = psutil.net_if_addrs()
            for int_name, addrs in stats.items():
                ip_info = [a.address for a in addrs if a.family == socket.AF_INET]
                if ip_info:
                    lines.append(f"{int_name}: {', '.join(ip_info)}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"Error getting local network info: {e}"

@tool("get_local_network_info")
async def get_local_network_info() -> str:
    """
    Gets local IPs, interfaces, and traffic stats.
    """
    return await asyncio.to_thread(_get_local_network_sync)