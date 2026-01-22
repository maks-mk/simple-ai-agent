import logging
import requests
import ipaddress
import platform
import shutil
import socket
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
    Robust network client with fallback providers.
    """
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        # Притворяемся curl, чтобы API охотнее отдавали JSON
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "curl/7.68.0" 
        })

    def my_ip(self) -> Result:
        """
        Try primary provider (ip.sb), fallback to secondary (ipify).
        """
        # Попытка 1: ip.sb (дает много инфы)
        result = self._request("https://api.ip.sb/jsonip")
        if result.ok:
            return result
            
        logger.warning(f"Primary IP API failed ({result.error.message}), trying fallback...")
        
        # Попытка 2: ipify (очень надежный, но только IP)
        return self._request("https://api.ipify.org?format=json")

    def get_ip_info(self, ip: str) -> Result:
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            return Result(ok=False, error=APIError("invalid_input", f"Invalid IP: {ip}", False, "Provide valid IPv4/IPv6"))
        
        # Для геоинформации используем ip.sb (ipify не дает гео в бесплатной версии)
        # Можно добавить fallback на ip-api.com, но он http (не https) в бесплатной версии
        return self._request(f"https://api.ip.sb/geoip/{ip}")

    def _request(self, url: str) -> Result:
        try:
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 429:
                return Result(ok=False, error=APIError("rate_limited", "Rate limit exceeded", True))
            
            if not response.ok:
                return Result(ok=False, error=APIError("http_error", f"HTTP {response.status_code}", True))
                
            return Result(ok=True, data=response.json())
            
        except requests.exceptions.Timeout:
            return Result(ok=False, error=APIError("timeout", f"Timeout connecting to {url}", True))
        except Exception as e:
            return Result(ok=False, error=APIError("network", str(e), True))

_net_client = NetworkClient()

def _format_result(result: Result) -> str:
    if result.ok:
        info = "\n".join([f"- {k}: {v}" for k, v in result.data.items()])
        return f"Success:\n{info}"
    else:
        err = result.error
        return f"Error ({err.type}): {err.message}. Hint: {err.hint}"

# ==================================================
# 2. EXPORTED TOOLS (Network - Public)
# ==================================================

@tool("get_public_ip")
def get_public_ip() -> str:
    """
    Returns the agent's current public IP address.
    Uses reliable providers with fallback logic.
    """
    return _format_result(_net_client.my_ip())

@tool("lookup_ip_info")
def lookup_ip_info(ip: str) -> str:
    """
    Retrieves geolocation info for a specific IP address (Country, ISP, etc).
    """
    return _format_result(_net_client.get_ip_info(ip))

# ==================================================
# 3. EXPORTED TOOLS (System / Hardware / Local Net)
# ==================================================

@tool("get_system_info")
def get_system_info() -> str:
    """
    Returns detailed system information: OS, CPU, RAM, Disk usage.
    """
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
            cpu_load = psutil.cpu_percent(interval=0.2)
            cores = psutil.cpu_count(logical=False)
            
            info.append(f"CPU: {cpu_load}% load{freq_str} ({cores} phys cores)")
        else:
            info.append("Extended hardware info unavailable (install 'psutil')")

        return "\n".join(info)
    except Exception as e:
        return f"Error getting system info: {e}"

@tool("get_local_network_info")
def get_local_network_info() -> str:
    """
    Returns local network interfaces, local IPs, and traffic stats.
    """
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