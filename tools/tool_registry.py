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
from core.tool_policy import ToolMetadata, default_tool_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolLoaderSpec:
    name: str
    enabled: Callable[[AgentConfig], bool]
    module_name: str
    tool_names: Sequence[str]
    configure: Callable[[Any, AgentConfig], None] | None = None
    optional_tool_names: Sequence[str] = ()
    metadata: Dict[str, ToolMetadata] | None = None
    optional_metadata: Dict[str, ToolMetadata] | None = None


class ToolRegistry:
    __slots__ = (
        "config",
        "tools",
        "tool_metadata",
        "mcp_clients",
        "loader_status",
        "mcp_server_status",
        "checkpoint_info",
        "_cleanup_callbacks",
    )

    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools: List[BaseTool] = []
        self.tool_metadata: Dict[str, ToolMetadata] = {}
        self.mcp_clients = []
        self.loader_status: List[Dict[str, Any]] = []
        self.mcp_server_status: List[Dict[str, Any]] = []
        self.checkpoint_info: Dict[str, Any] = {}
        self._cleanup_callbacks: List[Callable[[], Any]] = []

    async def load_all(self):
        for spec in self._loader_specs():
            if not spec.enabled(self.config):
                continue
            self._load_local_spec(spec)

        if self.config.mcp_config_path.exists():
            await self._load_mcp_tools()

    def _loader_specs(self) -> List[ToolLoaderSpec]:
        return [
            ToolLoaderSpec(
                name="filesystem",
                enabled=lambda config: config.enable_filesystem_tools,
                module_name="tools.filesystem",
                tool_names=(
                    "file_info_tool",
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
                metadata={
                    "file_info_tool": ToolMetadata(name="file_info", read_only=True, impact_scope="files", ui_kind="read"),
                    "read_file_tool": ToolMetadata(name="read_file", read_only=True, impact_scope="files", ui_kind="read"),
                    "write_file_tool": ToolMetadata(
                        name="write_file", mutating=True, requires_approval=True, impact_scope="files", ui_kind="write"
                    ),
                    "edit_file_tool": ToolMetadata(
                        name="edit_file", mutating=True, requires_approval=True, impact_scope="files", ui_kind="edit"
                    ),
                    "list_directory_tool": ToolMetadata(name="list_directory", read_only=True, impact_scope="files", ui_kind="read"),
                    "safe_delete_file": ToolMetadata(
                        name="safe_delete_file",
                        mutating=True,
                        destructive=True,
                        requires_approval=True,
                        impact_scope="files",
                        ui_kind="delete",
                    ),
                    "safe_delete_directory": ToolMetadata(
                        name="safe_delete_directory",
                        mutating=True,
                        destructive=True,
                        requires_approval=True,
                        impact_scope="files",
                        ui_kind="delete",
                    ),
                    "download_file": ToolMetadata(
                        name="download_file",
                        mutating=True,
                        requires_approval=True,
                        networked=True,
                        impact_scope="files",
                        ui_kind="write",
                    ),
                    "search_in_file_tool": ToolMetadata(name="search_in_file", read_only=True, impact_scope="files", ui_kind="search"),
                    "search_in_directory_tool": ToolMetadata(
                        name="search_in_directory", read_only=True, impact_scope="files", ui_kind="search"
                    ),
                    "tail_file_tool": ToolMetadata(name="tail_file", read_only=True, impact_scope="files", ui_kind="read"),
                    "find_file_tool": ToolMetadata(name="find_file", read_only=True, impact_scope="files", ui_kind="search"),
                },
            ),
            ToolLoaderSpec(
                name="local_delete_fallback",
                enabled=lambda config: not config.enable_filesystem_tools,
                module_name="tools.delete_tools",
                tool_names=("safe_delete_file", "safe_delete_directory"),
                metadata={
                    "safe_delete_file": ToolMetadata(
                        name="safe_delete_file",
                        mutating=True,
                        destructive=True,
                        requires_approval=True,
                        impact_scope="files",
                        ui_kind="delete",
                    ),
                    "safe_delete_directory": ToolMetadata(
                        name="safe_delete_directory",
                        mutating=True,
                        destructive=True,
                        requires_approval=True,
                        impact_scope="files",
                        ui_kind="delete",
                    ),
                },
            ),
            ToolLoaderSpec(
                name="search",
                enabled=lambda config: config.enable_search_tools,
                module_name="tools.search_tools",
                tool_names=("web_search", "batch_web_search", "fetch_content"),
                optional_tool_names=("crawl_site",),
                configure=self._configure_search,
                metadata={
                    "web_search": ToolMetadata(name="web_search", read_only=True, networked=True, impact_scope="network", ui_kind="search"),
                    "batch_web_search": ToolMetadata(
                        name="batch_web_search", read_only=True, networked=True, impact_scope="network", ui_kind="search"
                    ),
                    "fetch_content": ToolMetadata(
                        name="fetch_content", read_only=True, networked=True, impact_scope="network", ui_kind="read"
                    ),
                },
                optional_metadata={
                    "crawl_site": ToolMetadata(name="crawl_site", read_only=True, networked=True, impact_scope="network", ui_kind="search")
                },
            ),
            ToolLoaderSpec(
                name="system",
                enabled=lambda config: config.use_system_tools,
                module_name="tools.system_tools",
                tool_names=("get_public_ip", "lookup_ip_info", "get_system_info", "get_local_network_info"),
                metadata={
                    "get_public_ip": ToolMetadata(
                        name="get_public_ip", read_only=True, networked=True, impact_scope="network", ui_kind="read"
                    ),
                    "lookup_ip_info": ToolMetadata(
                        name="lookup_ip_info", read_only=True, networked=True, impact_scope="network", ui_kind="read"
                    ),
                    "get_system_info": ToolMetadata(name="get_system_info", read_only=True, impact_scope="local_state", ui_kind="read"),
                    "get_local_network_info": ToolMetadata(
                        name="get_local_network_info", read_only=True, impact_scope="network", ui_kind="read"
                    ),
                },
            ),
            ToolLoaderSpec(
                name="process",
                enabled=lambda config: config.enable_process_tools,
                module_name="tools.process_tools",
                tool_names=("run_background_process", "stop_background_process", "find_process_by_port"),
                configure=self._configure_safety,
                metadata={
                    "run_background_process": ToolMetadata(
                        name="run_background_process",
                        mutating=True,
                        requires_approval=True,
                        impact_scope="processes",
                        ui_kind="process",
                    ),
                    "stop_background_process": ToolMetadata(
                        name="stop_background_process",
                        mutating=True,
                        destructive=True,
                        requires_approval=True,
                        impact_scope="processes",
                        ui_kind="process",
                    ),
                    "find_process_by_port": ToolMetadata(
                        name="find_process_by_port", read_only=True, impact_scope="processes", ui_kind="process"
                    ),
                },
            ),
            ToolLoaderSpec(
                name="shell",
                enabled=lambda config: getattr(config, "enable_shell_tool", False),
                module_name="tools.local_shell",
                tool_names=("cli_exec",),
                configure=self._configure_shell,
                metadata={
                    "cli_exec": ToolMetadata(
                        name="cli_exec",
                        mutating=True,
                        destructive=True,
                        requires_approval=True,
                        impact_scope="local_state",
                        ui_kind="shell",
                    )
                },
            ),
        ]

    def _iter_spec_tool_names(self, spec: ToolLoaderSpec, module: Any) -> List[str]:
        names = list(spec.tool_names)
        names.extend(name for name in spec.optional_tool_names if hasattr(module, name))
        return names

    @staticmethod
    def _metadata_for_spec_attr(spec: ToolLoaderSpec, attr_name: str, tool_name: str) -> ToolMetadata:
        metadata = (spec.metadata or {}).get(attr_name) or (spec.optional_metadata or {}).get(attr_name)
        if metadata:
            return metadata
        return default_tool_metadata(tool_name)

    def _record_loader_status(
        self,
        *,
        spec: ToolLoaderSpec,
        loaded_tools: List[BaseTool],
        error: str = "",
    ) -> None:
        self.loader_status.append(
            {
                "loader": spec.name,
                "module": spec.module_name,
                "loaded_tools": [tool.name for tool in loaded_tools],
                "error": error,
            }
        )

    def _load_local_spec(self, spec: ToolLoaderSpec) -> None:
        try:
            module = importlib.import_module(spec.module_name)
            if spec.configure:
                spec.configure(module, self.config)

            loaded_tools: List[BaseTool] = []
            for attr_name in self._iter_spec_tool_names(spec, module):
                tool = getattr(module, attr_name)
                loaded_tools.append(tool)
                metadata = self._metadata_for_spec_attr(spec, attr_name, tool.name)
                self.tool_metadata[tool.name] = metadata
            self.tools.extend(loaded_tools)
            self._record_loader_status(spec=spec, loaded_tools=loaded_tools)
        except Exception as e:
            self._record_loader_status(spec=spec, loaded_tools=[], error=str(e))
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

    @staticmethod
    def _infer_mcp_metadata(tool_name: str) -> ToolMetadata:
        base = default_tool_metadata(tool_name, source="mcp")
        lowered = tool_name.lower()
        if any(token in lowered for token in ("read", "search", "find")):
            ui_kind = "read" if "read" in lowered else "search"
            return ToolMetadata(
                name=tool_name,
                read_only=True,
                mutating=False,
                destructive=False,
                requires_approval=False,
                networked=True,
                source="mcp",
                impact_scope="network",
                ui_kind=ui_kind,
            )
        return base

    @staticmethod
    def _metadata_override_for_tool(server_name: str, cfg: Dict[str, Any], tool_name: str) -> ToolMetadata | None:
        overrides = cfg.get("tool_metadata")
        if not isinstance(overrides, dict):
            return None

        short_name = tool_name.split(":", 1)[-1]
        candidates = (
            tool_name,
            short_name,
            f"{server_name}:{short_name}",
        )
        for candidate in candidates:
            raw = overrides.get(candidate)
            if isinstance(raw, dict):
                return ToolMetadata.from_dict(tool_name, raw, source="mcp")
        return None

    async def _load_single_mcp_server(
        self,
        name: str,
        cfg: Dict[str, Any],
        valid_keys: set[str],
        semaphore: asyncio.Semaphore,
    ):
        async with semaphore:
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient

                server_config = {key: value for key, value in cfg.items() if key in valid_keys}
                client = MultiServerMCPClient({name: server_config})
                return name, client, await client.get_tools(), None
            except Exception as e:
                return name, None, None, e

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
            results = await asyncio.gather(
                *(
                    self._load_single_mcp_server(name, cfg, valid_keys, semaphore)
                    for name, cfg in enabled_servers
                )
            )
            for name, client, mcp_tools, err in results:
                if err is not None:
                    self.mcp_server_status.append({"server": name, "loaded_tools": [], "error": str(err)})
                    logger.error("❌ MCP Server '%s' Error: %s", name, err)
                    continue

                self.mcp_clients.append(client)
                if mcp_tools:
                    self.tools.extend(mcp_tools)
                    for tool in mcp_tools:
                        metadata = self._metadata_override_for_tool(name, cfg, tool.name) or self._infer_mcp_metadata(tool.name)
                        self.tool_metadata[tool.name] = metadata
                    self.mcp_server_status.append(
                        {
                            "server": name,
                            "loaded_tools": [tool.name for tool in mcp_tools],
                            "error": "",
                        }
                    )
                    logger.info("✔ MCP Server '%s': Loaded %s tools", name, len(mcp_tools))
                else:
                    self.mcp_server_status.append({"server": name, "loaded_tools": [], "error": "No tools found"})
                    logger.warning("⚠ MCP Server '%s': No tools found", name)
        except Exception as e:
            logger.exception(f"Failed to load MCP tools: {e}")

    def register_cleanup_callback(self, callback: Callable[[], Any]) -> None:
        self._cleanup_callbacks.append(callback)

    def get_runtime_status(self) -> Dict[str, Any]:
        return {
            "checkpoint": self.checkpoint_info,
            "loaders": list(self.loader_status),
            "mcp_servers": list(self.mcp_server_status),
        }

    def get_runtime_status_lines(self) -> List[str]:
        lines: List[str] = []
        checkpoint = self.checkpoint_info or {}
        if checkpoint:
            lines.append(
                f"Checkpoint: requested={checkpoint.get('backend')} active={checkpoint.get('resolved_backend')} target={checkpoint.get('target')}"
            )
            for warning in checkpoint.get("warnings", []):
                lines.append(f"Checkpoint warning: {warning}")
        for status in self.loader_status:
            if status["error"]:
                lines.append(f"Loader {status['loader']}: ERROR {status['error']}")
        for status in self.mcp_server_status:
            if status["error"]:
                lines.append(f"MCP {status['server']}: ERROR {status['error']}")
            else:
                lines.append(
                    f"MCP {status['server']}: loaded {len(status['loaded_tools'])} tool(s)"
                )
        return lines

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

        for callback in self._cleanup_callbacks:
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                logger.error("Error during runtime cleanup: %s", e)

        self.mcp_clients.clear()
        logger.info("ToolRegistry cleanup completed.")
