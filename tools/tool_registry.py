import asyncio
import logging
import json
import os
from typing import List, Any
from langchain_core.tools import BaseTool

from core.config import AgentConfig

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools: List[BaseTool] = []
        # Сохраняем список клиентов, чтобы соединения не разрывались GC
        self.mcp_clients = [] 

    async def load_all(self):
        """Загружает все инструменты в зависимости от конфигурации."""
        
        # 1. Локальные файловые инструменты (Filesystem v2)
        if self.config.enable_filesystem_tools:
            self._load_filesystem_tools()
        else:
            # Fallback для старых (если включено OS_TOOLS, но FS выключено)
             self._load_local_tools()
        
        # 2. Поисковые инструменты
        self._load_search_tools()
        
        # 3. Системные инструменты (информация: IP, RAM, CPU)
        if self.config.use_system_tools:
            self._load_system_tools()
            
        # 4. OS инструменты (активные действия: процессы)
        if self.config.enable_process_tools:
            self._load_process_tools()

        # 5. Shell (CLI) инструмент (НОВОЕ)
        # Требует добавления enable_shell_tool в Config
        if getattr(self.config, "enable_shell_tool", False):
            self._load_shell_tool()
            
        # 6. MCP (Model Context Protocol) инструменты
        if self.config.mcp_config_path.exists():
            await self._load_mcp_tools()

        # logger.info(f"✔ Tools loaded: {[t.name for t in self.tools]}")

    def _load_filesystem_tools(self):
        """Загрузка продвинутых инструментов файловой системы."""
        try:
            from tools.filesystem import (
                read_file_tool,
                write_file_tool,
                edit_file_tool,
                list_directory_tool,
                download_file,
                set_safety_policy
            )
            # Safe delete is still useful separately
            from tools.delete_tools import safe_delete_file, safe_delete_directory
            
            # Apply safety policy
            set_safety_policy(self.config.safety)

            fs_tools = [
                read_file_tool,
                write_file_tool,
                edit_file_tool,
                list_directory_tool,
                safe_delete_file,
                safe_delete_directory,
                download_file
            ]
            self.tools.extend(fs_tools)
        except ImportError as e:
            logger.error(f"Failed to load filesystem tools: {e}")

    def _load_process_tools(self):
        """Загрузка инструментов управления процессами."""
        try:
            # Обрати внимание: проверь, называется ли файл os_tools.py или process_tools.py
            # В твоем предыдущем коде это было os_tools.py, но в дампе реестра process_tools.
            # Здесь я использую process_tools, как в твоем последнем примере реестра.
            from tools.process_tools import (
                run_background_process,
                stop_background_process,
                find_process_by_port,
                set_safety_policy
            )
            
            set_safety_policy(self.config.safety)

            proc_tools = [
                run_background_process,
                stop_background_process,
                find_process_by_port
            ]
            self.tools.extend(proc_tools)
        except ImportError as e:
            logger.error(f"Failed to load process tools: {e}")

    def _load_shell_tool(self):
        """Загрузка инструмента командной строки (cli_exec)."""
        try:
            from tools.local_shell import cli_exec, set_safety_policy
            
            set_safety_policy(self.config.safety)
            
            self.tools.append(cli_exec)
            logger.debug("✔ Shell tool (cli_exec) loaded")
        except ImportError as e:
            logger.error(f"Failed to load shell tool: {e}")

    def _load_local_tools(self):
        """Загрузка базовых локальных утилит (Fallback)."""
        try:
            from tools.delete_tools import safe_delete_file, safe_delete_directory
            # from tools.patch_tool import smart_replace # Removed: file missing

            local_tools = [
                safe_delete_file,
                safe_delete_directory,
                # smart_replace,
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
                fetch_content,
                batch_web_search,
                set_safety_policy
            )
            # crawl_site might be missing or not updated, check import
            try:
                from tools.search_tools import crawl_site
                has_crawl = True
            except ImportError:
                has_crawl = False

            set_safety_policy(self.config.safety)

            search_tools = [web_search, batch_web_search, fetch_content]
            if has_crawl:
                search_tools.append(crawl_site)
                
            self.tools.extend(search_tools)

        except ImportError as e:
            logger.warning(f"Search tools dependencies missing: {e}")

    def _load_system_tools(self):
        """Загрузка информационных утилит."""
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

    async def _load_mcp_tools(self):
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            
            # Чтение конфига
            try:
                raw_cfg = json.loads(self.config.mcp_config_path.read_text("utf-8"))
            except json.JSONDecodeError:
                logger.error(f"❌ Invalid JSON in {self.config.mcp_config_path}")
                return

            if not isinstance(raw_cfg, dict):
                logger.error(f"❌ MCP Config must be a dictionary, got {type(raw_cfg).__name__}")
                return
            
            # Рекурсивно раскрываем переменные окружения в конфигурации
            raw_cfg = self._expand_env_vars(raw_cfg)

            for name, cfg in raw_cfg.items():
                try:
                    if not isinstance(cfg, dict):
                        logger.warning(f"⚠ Skipping invalid config entry '{name}': Expected dict, got {type(cfg).__name__}")
                        continue

                    if not cfg.get("enabled", True):
                        continue
                    
                    # Фильтрация аргументов для MultiServerMCPClient
                    valid_keys = {
                        "command", "args", "env", "cwd", "encoding", "encoding_error_handler", # stdio
                        "url", "headers", "timeout", "sse_read_timeout", "auth", # http/sse
                        "terminate_on_close", "httpx_client_factory", # streamable specific
                        "transport", "session_kwargs" # common
                    }
                    
                    server_config = {k: v for k, v in cfg.items() if k in valid_keys}
                    
                    # 🔹 IMPORTANT: Create client for this server
                    # MultiServerMCPClient expects a dict {server_name: config}
                    client = MultiServerMCPClient({name: server_config})
                    self.mcp_clients.append(client) # Prevent GC
                    
                    # 🔹 Load Tools (No explicit connect() needed for MultiServerMCPClient)
                    mcp_tools = await client.get_tools()
                    
                    if mcp_tools:
                        self.tools.extend(mcp_tools)
                        logger.info(f"✔ MCP Server '{name}': Loaded {len(mcp_tools)} tools")
                    else:
                        logger.warning(f"⚠ MCP Server '{name}': No tools found")

                except Exception as e:
                    logger.error(f"❌ MCP Server '{name}' Error: {e}")

        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")

    def _expand_env_vars(self, data: Any) -> Any:
        """
        Рекурсивно проходит по структуре (dict/list/str) и подставляет ENV vars.
        Поддерживает синтаксис ${VAR} или $VAR.
        """
        if isinstance(data, dict):
            return {k: self._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        elif isinstance(data, str):
            return os.path.expandvars(data)
        else:
            return data

    async def cleanup(self):
        """Cleanup resources (close MCP connections)."""
        # MultiServerMCPClient doesn't have close/cleanup methods yet in 0.2.1,
        # but we hold references to clients.
        # Ideally, we should close them if they support it.
        # Checking mcp source, MultiServerMCPClient is not a context manager in 0.2.1 and doesn't have close().
        # But individual sessions are context managers.
        # We don't have active sessions here, they are created per-call or handled internally.
        
        # However, we might want to clear the list.
        self.mcp_clients.clear()
        logger.info("ToolRegistry cleanup completed.")
