import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from core.config import AgentConfig
from core.destructive_guardrails import (
    deny_recursive_destructive_command,
    deny_recursive_destructive_path,
)
from core.validation import validate_tool_result
from tools import process_tools
from tools.tool_registry import ToolRegistry


class ToolingRefactorTests(unittest.IsolatedAsyncioTestCase):
    def _make_config(self, **overrides):
        defaults = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "PROMPT_PATH": Path(__file__).resolve().parents[1] / "prompt.txt",
            "MCP_CONFIG_PATH": Path(__file__).resolve().parents[1] / "tests" / "missing_mcp.json",
            "ENABLE_SEARCH_TOOLS": False,
            "ENABLE_SYSTEM_TOOLS": False,
            "ENABLE_PROCESS_TOOLS": False,
            "ENABLE_SHELL_TOOL": False,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def _workspace_tempdir(self) -> Path:
        path = Path.cwd() / ".tmp_tests" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    async def test_tool_registry_preserves_filesystem_delete_tools(self):
        registry = ToolRegistry(self._make_config())
        await registry.load_all()
        names = {tool.name for tool in registry.tools}
        self.assertIn("safe_delete_file", names)
        self.assertIn("safe_delete_directory", names)

    def test_unknown_mcp_metadata_defaults_conservatively(self):
        metadata = ToolRegistry._infer_mcp_metadata("unknown:custom-tool")
        self.assertFalse(metadata.read_only)
        self.assertTrue(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertTrue(metadata.requires_approval)
        self.assertTrue(metadata.networked)
        self.assertEqual(metadata.source, "mcp")
        self.assertEqual(metadata.impact_scope, "unknown")

    def test_mcp_read_search_find_heuristic_disables_approval(self):
        read_meta = ToolRegistry._infer_mcp_metadata("any:read_document")
        search_meta = ToolRegistry._infer_mcp_metadata("any:search_docs")
        find_meta = ToolRegistry._infer_mcp_metadata("any:find_item")

        self.assertTrue(read_meta.read_only)
        self.assertFalse(read_meta.requires_approval)
        self.assertEqual(read_meta.ui_kind, "read")

        self.assertTrue(search_meta.read_only)
        self.assertFalse(search_meta.requires_approval)
        self.assertEqual(search_meta.ui_kind, "search")

        self.assertTrue(find_meta.read_only)
        self.assertFalse(find_meta.requires_approval)
        self.assertEqual(find_meta.ui_kind, "search")

    def test_recursive_destructive_path_guard_blocks_drive_roots(self):
        denied = deny_recursive_destructive_path(Path("C:/"), recursive=True)
        self.assertIsNotNone(denied)
        self.assertIn("blocked", denied.lower())

    def test_recursive_destructive_command_guard_blocks_obvious_wipes(self):
        denied = deny_recursive_destructive_command("Remove-Item C:\\ -Recurse -Force")
        self.assertIsNotNone(denied)
        self.assertIn("blocked", denied.lower())

    def test_validation_supports_delete_argument_aliases(self):
        tmp = self._workspace_tempdir()
        file_path = tmp / "data.txt"
        file_path.write_text("demo", encoding="utf-8")
        error = validate_tool_result("safe_delete_file", {"file_path": str(file_path)}, "Success")
        self.assertIsNotNone(error)
        self.assertIn("still exists", error)

        dir_path = tmp / "folder"
        dir_path.mkdir()
        error = validate_tool_result("safe_delete_directory", {"dir_path": str(dir_path)}, "Success")
        self.assertIsNotNone(error)
        self.assertIn("still exists", error)

    def test_cli_utils_import_does_not_require_prompt_toolkit(self):
        import core.cli_utils as cli_utils

        with mock.patch.dict(sys.modules, {"prompt_toolkit": None, "prompt_toolkit.key_binding": None}):
            reloaded = __import__("importlib").reload(cli_utils)
            self.assertTrue(callable(reloaded.prepare_markdown_for_render))

    def test_run_background_process_rejects_shell_operators(self):
        result = process_tools.run_background_process.invoke({"command": "python -c \"print(1)\" && whoami"})
        self.assertIn("ERROR[VALIDATION]", result)
        self.assertIn("Shell operators are not allowed", result)

    def test_run_background_process_accepts_argument_list(self):
        tmp = self._workspace_tempdir()
        process_tools.set_working_directory(str(tmp))
        result = process_tools.run_background_process.invoke(
            {"command": [sys.executable, "-c", "import time; time.sleep(30)"], "cwd": "."}
        )
        self.assertIn("Success: Process started with PID", result)


if __name__ == "__main__":
    unittest.main()
