import asyncio
import importlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Union

from langchain_core.tools import BaseTool

from core.config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolLoaderSpec:
    name: str
    enabled: Callable[[AgentConfig], bool]
    module_name: str
    tool_names: Sequence[str]
    configure: Callable[[Any, AgentConfig], None] | None = None
    optional_tool_names: Sequence[str] = ()


class ToolRegistry:
    __slots__ = ("config", "tools", "mcp_clients")

    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools: List[BaseTool] = []
        self.mcp_clients = []

    async def load_all(self):
        for spec in self._loader_specs():
            if not spec.enabled(self.config):
                continue
            self._load_from_spec(spec)

        if self.config.mcp_config_path.exists():
            await self._load_mcp_tools()

    def _loader_specs(self) -> List[ToolLoaderSpec]:
        return [
            ToolLoaderSpec(
                name="filesystem",
                enabled=lambda config: config.enable_filesystem_tools,
                module_name="tools.filesystem",
                tool_names=(
                    "read_file_tool",
                    "write_file_tool",
                    "edit_file_tool",
                    "list_directory_tool",
                    "safe_delete_file",
                    "safe_delete_directory",
                    "download_file",
                    "search_in_file_tool",
                    "search_in_directory_tool",
                    "tail_file_tool",
                    "find_file_tool",
                ),
                configure=self._configure_safety,
            ),
            ToolLoaderSpec(
                name="local_delete_fallback",
                enabled=lambda config: not config.enable_filesystem_tools,
                module_name="tools.delete_tools",
                tool_names=("safe_delete_file", "safe_delete_directory"),
            ),
            ToolLoaderSpec(
                name="search",
                enabled=lambda config: config.enable_search_tools,
                module_name="tools.search_tools",
                tool_names=("web_search", "batch_web_search", "fetch_content"),
                optional_tool_names=("crawl_site",),
                configure=self._configure_search,
            ),
            ToolLoaderSpec(
                name="system",
                enabled=lambda config: config.use_system_tools,
                module_name="tools.system_tools",
                tool_names=("get_public_ip", "lookup_ip_info", "get_system_info", "get_local_network_info"),
            ),
            ToolLoaderSpec(
                name="process",
                enabled=lambda config: config.enable_process_tools,
                module_name="tools.process_tools",
                tool_names=("run_background_process", "stop_background_process", "find_process_by_port"),
                configure=self._configure_safety,
            ),
            ToolLoaderSpec(
                name="shell",
                enabled=lambda config: getattr(config, "enable_shell_tool", False),
                module_name="tools.local_shell",
                tool_names=("cli_exec",),
                configure=self._configure_shell,
            ),
        ]

    def _load_from_spec(self, spec: ToolLoaderSpec) -> None:
        try:
            module = importlib.import_module(spec.module_name)
            if spec.configure:
                spec.configure(module, self.config)

            names = list(spec.tool_names)
            names.extend(name for name in spec.optional_tool_names if hasattr(module, name))
            self.tools.extend(getattr(module, name) for name in names)
        except Exception as e:
            logger.exception("Failed to load %s tools: %s", spec.name, e)

    @staticmethod
    def _configure_safety(module: Any, config: AgentConfig) -> None:
        if hasattr(module, "set_safety_policy"):
            module.set_safety_policy(config.safety)
        if hasattr(module, "set_working_directory"):
            module.set_working_directory(str(Path.cwd()))

    @staticmethod
    def _configure_search(module: Any, config: AgentConfig) -> None:
        if hasattr(module, "set_safety_policy"):
            module.set_safety_policy(config.safety)
        if hasattr(module, "set_runtime_config"):
            module.set_runtime_config(config)

    @staticmethod
    def _configure_shell(module: Any, config: AgentConfig) -> None:
        ToolRegistry._configure_safety(module, config)

    async def _load_mcp_tools(self):
        try:
            raw_cfg = self._read_mcp_config()
            enabled_servers = [
                (name, cfg)
                for name, cfg in raw_cfg.items()
                if isinstance(cfg, dict) and cfg.get("enabled", True)
            ]
            for name, cfg in raw_cfg.items():
                if not isinstance(cfg, dict):
                    logger.warning(
                        "⚠ Skipping invalid config entry '%s': Expected dict, got %s",
                        name,
                        type(cfg).__name__,
                    )

            if not enabled_servers:
                logger.debug("No enabled MCP servers in config.")
                return

            from langchain_mcp_adapters.client import MultiServerMCPClient

            valid_keys = {
                "command",
                "args",
                "env",
                "cwd",
                "encoding",
                "encoding_error_handler",
                "url",
                "headers",
                "timeout",
                "sse_read_timeout",
                "auth",
                "terminate_on_close",
                "httpx_client_factory",
                "transport",
                "session_kwargs",
            }

            semaphore = asyncio.Semaphore(4)

            async def _load_one_server(name: str, cfg: Dict[str, Any]):
                async with semaphore:
                    try:
                        server_config = {key: value for key, value in cfg.items() if key in valid_keys}
                        client = MultiServerMCPClient({name: server_config})
                        return name, client, await client.get_tools(), None
                    except Exception as e:
                        return name, None, None, e

            results = await asyncio.gather(*(_load_one_server(name, cfg) for name, cfg in enabled_servers))
            for name, client, mcp_tools, err in results:
                if err is not None:
                    logger.error("❌ MCP Server '%s' Error: %s", name, err)
                    continue

                self.mcp_clients.append(client)
                if mcp_tools:
                    self.tools.extend(mcp_tools)
                    logger.info("✔ MCP Server '%s': Loaded %s tools", name, len(mcp_tools))
                else:
                    logger.warning("⚠ MCP Server '%s': No tools found", name)
        except Exception as e:
            logger.exception(f"Failed to load MCP tools: {e}")

    def _read_mcp_config(self) -> Dict[str, Any]:
        try:
            raw_cfg = json.loads(self.config.mcp_config_path.read_text("utf-8"))
        except json.JSONDecodeError:
            logger.error(f"❌ Invalid JSON in {self.config.mcp_config_path}")
            return {}

        if not isinstance(raw_cfg, dict):
            logger.error(f"❌ MCP Config must be a dictionary, got {type(raw_cfg).__name__}")
            return {}
        return self._expand_env_vars(raw_cfg)

    def _expand_env_vars(self, data: Union[Dict[str, Any], List[Any], str]) -> Any:
        if isinstance(data, dict):
            return {k: self._expand_env_vars(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        if isinstance(data, str):
            return os.path.expandvars(data)
        return data

    async def cleanup(self):
        for client in self.mcp_clients:
            try:
                close_method = getattr(client, "aclose", None) or getattr(client, "close", None)
                if callable(close_method):
                    if inspect.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()
                elif hasattr(client, "__aexit__"):
                    try:
                        await client.__aexit__(None, None, None)
                    except Exception as e:
                        if "MultiServerMCPClient cannot be used as a context manager" not in str(e):
                            raise
            except Exception as e:
                logger.error("Error closing MCP client: %s", e)

        self.mcp_clients.clear()
        logger.info("ToolRegistry cleanup completed.")
