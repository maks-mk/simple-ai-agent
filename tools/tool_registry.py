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

    def _set_capability(self, tools: List[BaseTool], capability: str) -> None:
        for t in tools:
            try:
                meta = getattr(t, "metadata", None)
                if meta is None:
                    t.metadata = {"capability": capability}
                elif isinstance(meta, dict):
                    meta["capability"] = capability
                else:
                    t.metadata = {"capability": capability}
            except Exception as e:
                logger.warning(f"Failed to set capability for tool {t.name}: {e}")

    def get_tool_capability(self, tool: BaseTool) -> str:
        """
        Возвращает capability инструмента.
        Если не задано, пытается определить эвристически.
        """
        meta = getattr(tool, "metadata", None)
        if isinstance(meta, dict):
            cap = meta.get("capability")
            if cap: return cap

        # Heuristic Logic
        name = tool.name.lower()
        
        # 1. Явные маркеры записи (Write)
        write_keywords = [
            "write", "save", "edit", "delete", "move", "create", 
            "mkdir", "update", "replace", "append", "remove", 
            "upload", "post", "send", "patch", "put"
        ]
        if any(k in name for k in write_keywords):
            return "write"
            
        # 2. Явные маркеры чтения (Safe)
        safe_keywords = [
            "get", "read", "search", "list", "fetch", "check", 
            "status", "info", "lookup", "query", "load", "view",
            "describe", "scan"
        ]
        if any(k in name for k in safe_keywords):
            return "safe"
            
        # 3. Default Policy -> Write (Safety First)
        # Если мы не можем понять, что делает инструмент, считаем его опасным
        return "write"

    async def load_all(self):
        """Загружает все инструменты."""
        self._load_local_tools()
        self._load_search_tools()
        
        if self.config.use_long_term_memory:
            self._load_memory_tools()
            
        if self.config.use_system_tools:
            self._load_system_tools()
            
        if self.config.enable_media_tools:
            self._load_media_tools()

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
            self._set_capability(local_tools, "write")
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
                self._set_capability(search_tools, "safe")
                self.tools.extend(search_tools)

            if self.config.enable_deep_search and deep_search:
                self._set_capability([deep_search], "safe")
                self.tools.append(deep_search)

        except ImportError:
            logger.warning("Search tools dependencies missing.")

    def _load_memory_tools(self):
        try:
            from tools.memory_manager import MemoryManager, remember_fact, recall_facts, forget_fact
            # Инициализируем менеджер
            MemoryManager(db_path=self.config.memory_db_path)
            self._set_capability([remember_fact, recall_facts, forget_fact], "safe")
            self.tools.extend([remember_fact, recall_facts, forget_fact])
        except ImportError:
            logger.warning("MemoryManager not available (check dependencies).")
            
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
                # Оставляем только те ключи, которые понимает конструктор сессии и транспорта.
                valid_keys = {
                    "command", "args", "env", "cwd", "encoding", "encoding_error_handler", # stdio
                    "url", "headers", "timeout", "sse_read_timeout", "auth", # http/sse
                    "terminate_on_close", "httpx_client_factory", # streamable specific
                    "transport", "session_kwargs" # common
                }
                
                server_config = {k: v for k, v in cfg.items() if k in valid_keys}
                
                # Гарантируем наличие args для stdio (иначе падает)
                if server_config.get("transport") == "stdio" and "args" not in server_config:
                    server_config["args"] = []
                
                # Алиас для удобства: "http" -> "streamable_http"
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

    def _load_system_tools(self):
        """Загрузка системных утилит (сеть, ОС)."""
        try:
            from tools.system_tools import (
                get_public_ip, 
                lookup_ip_info, 
                get_system_info, 
                get_local_network_info,
                run_background_process,
                stop_background_process
            )

            system_tools = [
                get_public_ip,
                lookup_ip_info,
                get_system_info,
                get_local_network_info,
                run_background_process,
                stop_background_process
            ]
            # Mark run_background_process as safe or write? 
            # It changes system state (starts process), so 'write' might be safer, 
            # but usually we want to allow it in exploration if it's just a local server.
            # However, safety first: 'write' bucket.
            
            self._set_capability([t for t in system_tools if t.name not in ["run_background_process", "stop_background_process"]], "safe")
            self._set_capability([run_background_process, stop_background_process], "write")
            
            self.tools.extend(system_tools)
        except ImportError as e:
            logger.error(f"Failed to load system tools: {e}")

    def _load_media_tools(self):
        """Загрузка медиа инструментов (yt-dlp)."""
        try:
            from tools.media_tools import download_media
            # Скачивание файла - это операция записи
            self._set_capability([download_media], "write") 
            self.tools.append(download_media)
        except ImportError as e:
            logger.error(f"Failed to load media tools: {e}")

    async def cleanup(self):
        """Закрывает MCP соединения при выходе."""
        if self.mcp_client:
            try:
                # Пытаемся закрыть клиент, если библиотека предоставляет такой метод
                # В langchain-mcp-adapters v0.0.1+ может быть метод close() или aclose()
                if hasattr(self.mcp_client, "aclose"):
                    await self.mcp_client.aclose()
                elif hasattr(self.mcp_client, "close"):
                    await self.mcp_client.close()
                # Если метода нет, полагаемся на то, что Python закроет ресурсы при завершении процесса,
                # так как мы держали ссылку в self.mcp_client
            except Exception as e:
                logger.warning(f"Error closing MCP client: {e}")
