import asyncio
import logging
import json
from pathlib import Path
from typing import List, Optional
from langchain_core.tools import BaseTool

from core.config import AgentConfig

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools: List[BaseTool] = []
        # Сохраняем ссылку на MCP клиент, чтобы соединения не разрывались GC
        self.mcp_client = None 

    async def load_all(self):
        """Загружает все инструменты в зависимости от конфигурации."""
        
        # 1. Локальные файловые инструменты (всегда включены, если нужны агенту)
        self._load_local_tools()
        
        # 2. Поисковые инструменты
        self._load_search_tools()
        
        # 3. Системные инструменты (информация: IP, RAM, CPU)
        if self.config.use_system_tools:
            self._load_system_tools()
            
        # 4. OS инструменты (активные действия: процессы, скачивание)
        if self.config.enable_os_tools:
            self._load_os_tools()
            
        # 5. Медиа инструменты (yt-dlp)
        if self.config.enable_media_tools:
            self._load_media_tools()

        # 6. MCP (Model Context Protocol) инструменты
        if self.config.mcp_config_path.exists():
            await self._load_mcp_tools()

        # logger.info(f"✔ Tools loaded: {[t.name for t in self.tools]}")

    def _load_local_tools(self):
        """Загрузка локальных файловых утилит."""
        try:
            # 1. Удаление
            from tools.delete_tools import safe_delete_file, safe_delete_directory
            
            # 2. Умное редактирование
            from tools.patch_tool import smart_replace

            local_tools = [
                safe_delete_file,
                safe_delete_directory,
                smart_replace,
            ]
            self.tools.extend(local_tools)
            
        except ImportError as e:
            logger.error(f"Failed to load local tools: {e}")

    def _load_search_tools(self):
        if not self.config.enable_search_tools:
            logger.info("Search tools are disabled via config.")
            return

        try:
            from tools.search_tools import (
                web_search,
                deep_search,
                fetch_content,
                batch_web_search,
                crawl_site,
            )

            if web_search and fetch_content:
                search_tools = [web_search, batch_web_search, fetch_content, crawl_site]
                self.tools.extend(search_tools)

            if self.config.enable_deep_search and deep_search:
                self.tools.append(deep_search)

        except ImportError:
            logger.warning("Search tools dependencies missing.")

    def _load_system_tools(self):
        """
        Загрузка информационных утилит (чтение состояния системы).
        Безопасны для использования.
        """
        try:
            from tools.system_tools import (
                get_public_ip, 
                lookup_ip_info,
                get_system_info, 
                get_local_network_info
            )
            
            system_tools = [
                get_public_ip, 
                lookup_ip_info,
                get_system_info, 
                get_local_network_info
            ]
            self.tools.extend(system_tools)
        except ImportError as e:
            logger.error(f"Failed to load system tools: {e}")

    def _load_os_tools(self):
        """
        Загрузка активных системных утилит (управление процессами, скачивание).
        Могут быть отключены через ENABLE_OS_TOOLS=false.
        """
        try:
            from tools.os_tools import (
                run_background_process,
                stop_background_process,
                find_process_by_port,
                download_file
            )
            
            os_tools = [
                run_background_process,
                stop_background_process,
                find_process_by_port,
                download_file
            ]
            self.tools.extend(os_tools)
        except ImportError as e:
            logger.error(f"Failed to load OS tools: {e}")

    def _load_media_tools(self):
        """Загрузка медиа инструментов (yt-dlp)."""
        try:
            from tools.media_tools import download_media
            self.tools.append(download_media)
        except ImportError as e:
            logger.error(f"Failed to load media tools: {e}")

    async def _load_mcp_tools(self):
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            
            # Чтение конфига
            raw_cfg = json.loads(self.config.mcp_config_path.read_text("utf-8"))
            mcp_cfg = {}
            
            for name, cfg in raw_cfg.items():
                if not cfg.get("enabled", True):
                    continue
                
                # Фильтрация аргументов для MultiServerMCPClient.
                valid_keys = {
                    "command", "args", "env", "cwd", "encoding", "encoding_error_handler", # stdio
                    "url", "headers", "timeout", "sse_read_timeout", "auth", # http/sse
                    "terminate_on_close", "httpx_client_factory", # streamable specific
                    "transport", "session_kwargs" # common
                }
                
                server_config = {k: v for k, v in cfg.items() if k in valid_keys}
                
                # Гарантируем наличие args для stdio
                if server_config.get("transport") == "stdio" and "args" not in server_config:
                    server_config["args"] = []
                
                # Алиас для удобства
                if server_config.get("transport") == "http":
                    server_config["transport"] = "streamable_http"
                    
                mcp_cfg[name] = server_config

            if mcp_cfg:
                # ВАЖНО: Сохраняем клиент в self, чтобы GC не убил соединения
                self.mcp_client = MultiServerMCPClient(mcp_cfg)
                
                # Получаем инструменты с таймаутом
                new_tools = await asyncio.wait_for(self.mcp_client.get_tools(), timeout=120)
                
                self.tools.extend(new_tools)
                logger.debug(f"🔌 MCP Adapter connected. Loaded {len(new_tools)} tools.")
                
        except Exception as e:
            logger.error(f"MCP Load Error: {e}")

    async def cleanup(self):
        """Закрывает MCP соединения при выходе."""
        if self.mcp_client:
            try:
                if hasattr(self.mcp_client, "aclose"):
                    await self.mcp_client.aclose()
                elif hasattr(self.mcp_client, "close"):
                    await self.mcp_client.close()
            except Exception as e:
                logger.warning(f"Error closing MCP client: {e}")