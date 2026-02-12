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

# Пробуем импортировать psutil для расширенной статистики
try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

# ==================================================
# 1. CORE LOGIC (Network Client)
# Эта часть оставлена здесь, так как она используется 
# и в system_tools, и в os_tools (через импорт)
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

# Глобальный клиент для переиспользования
_net_client: Optional[NetworkClient] = None

def get_net_client() -> NetworkClient:
    """Returns a singleton NetworkClient instance."""
    global _net_client
    if _net_client is None:
        _net_client = NetworkClient()
    return _net_client

def _format_result(result: Result) -> str:
    if result.ok:
        return "\n".join([f"- {k}: {v}" for k, v in result.data.items()])
    return f"Error: {result.error.message}"

# ==================================================
# 2. NETWORK TOOLS (Informational)
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
# 3. HARDWARE & SYSTEM TOOLS (Informational)
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
        else:
            info.append("(psutil not installed, detailed RAM/CPU info unavailable)")
            
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