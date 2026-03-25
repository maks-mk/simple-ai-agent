import io
import asyncio
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import ToolMessage
from rich.console import Console

import agent_cli
from core.session_store import SessionSnapshot, SessionStore
from core.stream_processor import StreamProcessor
from core.tool_policy import ToolMetadata
from core.ui_theme import AGENT_THEME


class FakeTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class FakeToolRegistry:
    def __init__(self):
        self.tools = [
            FakeTool("read_file", "Read a file"),
            FakeTool("edit_file", "Edit a file in place"),
            FakeTool("context7:resolve-library-id", "Resolve a Context7 library id"),
        ]
        self.tool_metadata = {
            "read_file": ToolMetadata(name="read_file", read_only=True),
            "edit_file": ToolMetadata(name="edit_file", mutating=True, requires_approval=True),
            "context7:resolve-library-id": ToolMetadata(
                name="context7:resolve-library-id",
                read_only=True,
                networked=True,
                source="mcp",
            ),
        }
        self.checkpoint_info = {
            "backend": "sqlite",
            "resolved_backend": "sqlite",
            "target": ".agent_state/checkpoints.sqlite",
            "warnings": [],
        }
        self.mcp_server_status = [
            {"server": "context7", "loaded_tools": ["resolve-library-id"], "error": ""},
            {"server": "pdf-tools", "loaded_tools": ["read_pdf"], "error": ""},
        ]
        self.loader_status = []

    def get_runtime_status_lines(self):
        return [
            "Checkpoint: requested=sqlite active=sqlite target=.agent_state/checkpoints.sqlite",
            "MCP context7: loaded 1 tool(s)",
            "MCP pdf-tools: loaded 1 tool(s)",
        ]


class FakeLive:
    def __init__(self):
        self.renderable = None

    def update(self, renderable):
        self.renderable = renderable


class CliUxTests(unittest.TestCase):
    def _workspace_tempdir(self) -> Path:
        path = Path.cwd() / ".tmp_tests" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def _console(self) -> Console:
        return Console(
            record=True,
            file=io.StringIO(),
            width=140,
            force_terminal=False,
            theme=AGENT_THEME,
        )

    def _snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            session_id="session-1234567890abcdef",
            thread_id="thread-abcdef1234567890",
            checkpoint_backend="sqlite",
            checkpoint_target=".agent_state/checkpoints.sqlite",
            created_at="2026-03-20T12:00:00+00:00",
            updated_at="2026-03-20T12:00:00+00:00",
            approval_mode="prompt",
        )

    def _config(self):
        return SimpleNamespace(
            provider="openai",
            openai_model="gpt-4o",
            gemini_model="gemini-1.5-flash",
            checkpoint_backend="sqlite",
            enable_approvals=True,
            debug=False,
        )

    def test_render_overview_includes_runtime_summary(self):
        out = self._console()
        agent_cli.render_overview(
            self._config(),
            FakeToolRegistry(),
            self._snapshot(),
            show_cheatsheet=True,
            out=out,
        )

        output = out.export_text()
        self.assertIn("AI Agent", output)
        self.assertIn("OpenAI", output)
        self.assertIn("gpt-4o", output)
        self.assertIn("sqlite", output)
        self.assertIn("context7, pdf-tools", output)
        self.assertIn("/new", output)

    def test_render_overview_shows_session_scoped_always_status(self):
        out = self._console()
        snapshot = self._snapshot()
        snapshot.approval_mode = "always"

        agent_cli.render_overview(self._config(), FakeToolRegistry(), snapshot, out=out)

        output = out.export_text()
        self.assertIn("always for this session", output)

    def test_try_handle_command_supports_new_alias(self):
        calls = {"reset": 0}

        def reset_session():
            calls["reset"] += 1

        handled = agent_cli.try_handle_command("/new", FakeToolRegistry(), reset_session, lambda: None)

        self.assertTrue(handled)
        self.assertEqual(calls["reset"], 1)

    def test_try_handle_command_can_reset_session_approval_mode(self):
        snapshot = self._snapshot()
        snapshot.approval_mode = "always"

        def reset_session():
            nonlocal snapshot
            snapshot = SessionSnapshot(
                session_id="session-new",
                thread_id="thread-new",
                checkpoint_backend="sqlite",
                checkpoint_target=".agent_state/checkpoints.sqlite",
                created_at="2026-03-21T12:00:00+00:00",
                updated_at="2026-03-21T12:00:00+00:00",
                approval_mode="prompt",
            )

        handled = agent_cli.try_handle_command("/new", FakeToolRegistry(), reset_session, lambda: None)

        self.assertTrue(handled)
        self.assertEqual(snapshot.approval_mode, "prompt")

    def test_render_help_is_task_oriented(self):
        out = self._console()
        agent_cli.render_help(out=out)

        output = out.export_text()
        self.assertIn("Start work", output)
        self.assertIn("Inspect tools", output)
        self.assertIn("/session", output)
        self.assertIn("Approvals", output)
        self.assertIn("Yes/No/Always", output)

    def test_render_tools_groups_tools_and_badges(self):
        out = self._console()
        agent_cli.render_tools(FakeToolRegistry(), out=out)

        output = out.export_text()
        self.assertIn("Read-only", output)
        self.assertIn("Protected", output)
        self.assertIn("MCP", output)
        self.assertIn("read-only", output)
        self.assertIn("approval", output)
        self.assertIn("network", output)

    def test_approval_summary_defaults_to_deny_for_destructive_batch(self):
        summary = agent_cli._summarize_approval_request(
            [{"name": "safe_delete_file", "policy": {"destructive": True, "mutating": True}}]
        )

        self.assertFalse(summary.default_approve)
        self.assertEqual(summary.risk_level, "high")

    def test_approval_summary_defaults_to_approve_for_non_destructive_batch(self):
        summary = agent_cli._summarize_approval_request(
            [{"name": "edit_file", "policy": {"mutating": True}}]
        )

        self.assertTrue(summary.default_approve)
        self.assertEqual(summary.risk_level, "medium")

    def test_approval_summary_defaults_to_deny_for_mixed_batch(self):
        summary = agent_cli._summarize_approval_request(
            [{"name": "download_file", "policy": {"mutating": True, "networked": True}}]
        )

        self.assertFalse(summary.default_approve)
        self.assertEqual(summary.risk_level, "high")
        self.assertIn("files", summary.impacts)
        self.assertIn("network", summary.impacts)

    def test_stream_processor_status_labels_are_normalized(self):
        processor = StreamProcessor(self._console())

        processor.active_node = "agent"
        self.assertEqual(processor._status_label(), "Thinking")
        processor.active_node = "tools"
        self.assertEqual(processor._status_label(), "Running tools")
        processor.active_node = "critic"
        self.assertEqual(processor._status_label(), "Reviewing")
        processor.active_node = "approval"
        self.assertEqual(processor._status_label(), "Waiting for approval")
        processor.active_node = "summarize"
        self.assertEqual(processor._status_label(), "Compressing context")

    def test_stream_processor_live_layout_keeps_pending_markdown_above_spinner(self):
        processor = StreamProcessor(self._console())
        processor._append_text("Partial answer")
        live = FakeLive()

        processor._update_live_display(live)

        self.assertIsNotNone(live.renderable)
        self.assertEqual(len(live.renderable.renderables), 2)
        self.assertEqual(type(live.renderable.renderables[0]).__name__, "Padding")
        self.assertEqual(type(live.renderable.renderables[1]).__name__, "Spinner")

    def test_stream_processor_tool_lines_use_turn_local_prefix(self):
        out = self._console()
        processor = StreamProcessor(out)
        processor.tool_buffer["call-1"] = {"name": "edit_file", "args": {"path": "demo.txt"}}

        processor._render_tool_call({"id": "call-1", "name": "edit_file", "args": {"path": "demo.txt"}})
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-1",
                name="edit_file",
                content="Success: File edited.\n\nDiff:\n```diff\n-foo\n+bar\n```",
            )
        )

        output = out.export_text()
        self.assertIn("tool", output)
        self.assertIn("edit_file", output)
        self.assertIn("foo", output)

    def test_prompt_for_interrupt_yes_approves(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        snapshot = self._snapshot()
        out = self._console()

        async def selector(summary):
            self.assertTrue(summary.default_approve)
            return "yes"

        result = asyncio.run(
            agent_cli.prompt_for_interrupt(
                {
                    "kind": "tool_approval",
                    "tools": [{"name": "edit_file", "args": {"path": "demo.txt"}, "policy": {"mutating": True}}],
                },
                snapshot,
                store,
                out=out,
                selector=selector,
            )
        )

        self.assertEqual(result, {"approved": True})
        self.assertEqual(snapshot.approval_mode, "prompt")
        self.assertIn("approved", out.export_text())

    def test_prompt_for_interrupt_no_denies(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        snapshot = self._snapshot()
        out = self._console()

        async def selector(summary):
            self.assertFalse(summary.default_approve)
            return "no"

        result = asyncio.run(
            agent_cli.prompt_for_interrupt(
                {
                    "kind": "tool_approval",
                    "tools": [{"name": "safe_delete_file", "args": {}, "policy": {"destructive": True, "mutating": True}}],
                },
                snapshot,
                store,
                out=out,
                selector=selector,
            )
        )

        self.assertEqual(result, {"approved": False})
        self.assertEqual(snapshot.approval_mode, "prompt")

    def test_prompt_for_interrupt_always_persists_session_mode(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        snapshot = self._snapshot()
        out = self._console()

        async def selector(summary):
            return "always"

        result = asyncio.run(
            agent_cli.prompt_for_interrupt(
                {
                    "kind": "tool_approval",
                    "tools": [{"name": "edit_file", "args": {"path": "demo.txt"}, "policy": {"mutating": True}}],
                },
                snapshot,
                store,
                out=out,
                selector=selector,
            )
        )

        loaded = store.load_active_session()
        self.assertEqual(result, {"approved": True})
        self.assertEqual(snapshot.approval_mode, "always")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.approval_mode, "always")
        self.assertIn("Future protected actions will be auto-approved", out.export_text())

    def test_prompt_for_interrupt_bypasses_prompt_when_session_is_always(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        snapshot = self._snapshot()
        snapshot.approval_mode = "always"
        store.save_active_session(snapshot)
        out = self._console()

        async def selector(_summary):
            raise AssertionError("selector must not run when approval mode is always")

        result = asyncio.run(
            agent_cli.prompt_for_interrupt(
                {
                    "kind": "tool_approval",
                    "tools": [{"name": "edit_file", "args": {"path": "demo.txt"}, "policy": {"mutating": True}}],
                },
                snapshot,
                store,
                out=out,
                selector=selector,
            )
        )

        self.assertEqual(result, {"approved": True})
        self.assertIn("auto-approved", out.export_text())

    def test_prompt_for_interrupt_cancel_denies(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        snapshot = self._snapshot()
        out = self._console()

        async def selector(summary):
            return None

        result = asyncio.run(
            agent_cli.prompt_for_interrupt(
                {
                    "kind": "tool_approval",
                    "tools": [{"name": "edit_file", "args": {"path": "demo.txt"}, "policy": {"mutating": True}}],
                },
                snapshot,
                store,
                out=out,
                selector=selector,
            )
        )

        self.assertEqual(result, {"approved": False})
        self.assertEqual(snapshot.approval_mode, "prompt")


if __name__ == "__main__":
    unittest.main()
