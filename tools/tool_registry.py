import asyncio
import logging
import json
from pathlib import Path
from typing import List
from langchain_core.tools import BaseTool

from core.config import AgentConfig

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools: List[BaseTool] = []

    async def load_all(self):
        """Загружает все инструменты."""
        self._load_local_tools()
        self._load_search_tools()
        self._load_system_tools()
        
        if self.config.use_long_term_memory:
            self._load_memory_tools()
            
        if self.config.mcp_config_path.exists():
            await self._load_mcp_tools()

        #logger.info(f"✅ Tools loaded: {[t.name for t in self.tools]}")

    def _load_local_tools(self):
        """Загрузка локальных файловых утилит."""
        try:
            # 1. Удаление (ИСПРАВЛЕНО: импортируем функции, а не классы)
            from tools.delete_tools import safe_delete_file, safe_delete_directory
            
            # 2. Умное редактирование (если вы создали файл core/patch_tool.py)
            # Если файла нет, закомментируйте следующие 2 строки
            from tools.patch_tool import smart_replace
            
            self.tools.extend([
                safe_delete_file, 
                safe_delete_directory,
                smart_replace
            ])
            
        except ImportError as e:
            logger.error(f"Failed to load local tools: {e}")

    def _load_search_tools(self):
        try:
            from tools.search_tools import web_search, deep_search, fetch_content, batch_web_search
            
            if web_search and fetch_content:
                self.tools.extend([web_search, batch_web_search, fetch_content])
            
            if self.config.enable_deep_search and deep_search:
                self.tools.append(deep_search)
        except ImportError:
            logger.warning("Search tools dependencies missing.")

    def _load_memory_tools(self):
        try:
            from tools.memory_manager import MemoryManager, remember_fact, recall_facts, forget_fact
            # Инициализируем менеджер
            MemoryManager(db_path=self.config.memory_db_path)
            self.tools.extend([remember_fact, recall_facts, forget_fact])
        except ImportError:
            logger.warning("MemoryManager not available (check dependencies).")
            
    async def _load_mcp_tools(self):
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            raw_cfg = json.loads(self.config.mcp_config_path.read_text("utf-8"))
            mcp_cfg = {
                name: {
                    **{k: v for k, v in cfg.items() if k != 'enabled'},
                    "args": cfg.get("args", [])
                }
                for name, cfg in raw_cfg.items() if cfg.get("enabled", True)
            }
            if mcp_cfg:
                client = MultiServerMCPClient(mcp_cfg)
                new_tools = await asyncio.wait_for(client.get_tools(), timeout=120)
                self.tools.extend(new_tools)
        except Exception as e:
            logger.error(f"MCP Load Error: {e}")

    def _load_system_tools(self):
        """Загрузка системных утилит (сеть, ОС)."""
        try:
            # Добавили get_local_network_info в импорт и в список
            from tools.system_tools import get_public_ip, lookup_ip_info, get_system_info, get_local_network_info
            
            self.tools.extend([
                get_public_ip, 
                lookup_ip_info, 
                get_system_info, 
                get_local_network_info
            ])
        except ImportError as e:
            logger.error(f"Failed to load system tools: {e}")
            