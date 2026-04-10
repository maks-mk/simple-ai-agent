import asyncio
import io
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
            FakeTool("edit_file", "Edit a file"),
            FakeTool("context7:resolve-library-id", "Resolve a docs library id"),
        ]
        self.tool_metadata = {
            "read_file": ToolMetadata(
                name="read_file",
                read_only=True,
                impact_scope="files",
                ui_kind="read",
            ),
            "edit_file": ToolMetadata(
                name="edit_file",
                mutating=True,
                requires_approval=True,
                impact_scope="files",
                ui_kind="edit",
            ),
            "context7:resolve-library-id": ToolMetadata(
                name="context7:resolve-library-id",
                read_only=True,
                networked=True,
                source="mcp",
                impact_scope="network",
                ui_kind="search",
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
        ]

    def get_runtime_status_lines(self):
        return ["MCP context7: loaded 1 tool(s)"]


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
        self.assertIn("context7", output)
        self.assertIn("tools 3", output)

    def test_render_turn_header_shows_turn_model_thread_and_session(self):
        out = self._console()
        agent_cli.render_turn_header(3, "gpt-4o", self._snapshot(), out=out)
        output = out.export_text()
        self.assertIn("Turn", output)
        self.assertIn("3", output)
        self.assertIn("gpt-4o", output)
        self.assertIn("thread", output)
        self.assertIn("session", output)

    def test_build_initial_state_uses_deterministic_retry_fields(self):
        state = agent_cli.build_initial_state("hello", session_id="session-1")
        self.assertEqual(state["retry_count"], 0)
        self.assertEqual(state["retry_reason"], "")
        self.assertEqual(state["turn_outcome"], "")
        self.assertEqual(state["final_issue"], "")
        self.assertNotIn("critic_status", state)

    def test_approval_summary_uses_explicit_impact_scope(self):
        summary = agent_cli._summarize_approval_request(
            [
                {
                    "name": "totally_custom",
                    "policy": {"mutating": True, "impact_scope": "processes", "ui_kind": "process"},
                },
                {
                    "name": "docs_lookup",
                    "policy": {"networked": True, "impact_scope": "network", "ui_kind": "search"},
                },
            ]
        )

        self.assertFalse(summary.default_approve)
        self.assertEqual(summary.risk_level, "high")
        self.assertIn("processes", summary.impacts)
        self.assertIn("network", summary.impacts)

    def test_approval_panel_body_renders_ui_kind_impact_and_target(self):
        summary = agent_cli._summarize_approval_request(
            [
                {
                    "name": "edit_file",
                    "args": {"path": "demo.txt"},
                    "policy": {"mutating": True, "impact_scope": "files", "ui_kind": "edit"},
                }
            ]
        )
        body = agent_cli._approval_panel_body(
            [
                {
                    "name": "edit_file",
                    "args": {"path": "demo.txt"},
                    "policy": {"mutating": True, "impact_scope": "files", "ui_kind": "edit"},
                }
            ],
            summary,
        )

        console = self._console()
        console.print(body)
        output = console.export_text()
        self.assertIn("edit_file", output)
        self.assertIn("Action", output)
        self.assertIn("Target", output)
        self.assertIn("demo.txt", output)

    def test_stream_processor_status_labels_follow_graph_phases(self):
        processor = StreamProcessor(self._console())

        processor.active_node = "agent"
        self.assertEqual(processor._status_label(), "Thinking")
        processor.active_node = "tools"
        self.assertEqual(processor._status_label(), "Running tools")
        processor.active_node = "approval"
        self.assertEqual(processor._status_label(), "Awaiting approval")
        processor.active_node = "prepare_retry"
        self.assertEqual(processor._status_label(), "Recovering")
        processor.active_node = "finalize_blocked"
        self.assertEqual(processor._status_label(), "Finishing")

    def test_stream_processor_live_layout_keeps_pending_markdown_and_spinner(self):
        processor = StreamProcessor(self._console())
        processor._append_text("Partial answer")
        live = FakeLive()

        processor._update_live_display(live)

        self.assertIsNotNone(live.renderable)
        self.assertEqual(len(live.renderable.renderables), 2)

    def test_stream_processor_formats_tool_result_from_ui_kind_metadata(self):
        out = self._console()
        processor = StreamProcessor(
            out,
            tool_metadata={
                "context7:query-docs": ToolMetadata(
                    name="context7:query-docs",
                    read_only=True,
                    networked=True,
                    impact_scope="network",
                    ui_kind="read",
                )
            },
        )

        processor._remember_tool_call({"id": "tool-1", "name": "context7:query-docs", "args": {"query": "graph"}})
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="tool-1",
                name="context7:query-docs",
                content="line one\nline two\nline three",
            )
        )

        output = out.export_text()
        self.assertIn("Read 3 lines", output)

    def test_stream_processor_stats_include_approvals_and_retries_without_tools_counter(self):
        processor = StreamProcessor(self._console())
        processor.completed_tool_count = 2
        processor.approval_count = 1
        processor.retry_count = 1
        stats = processor.render_stats()
        self.assertNotIn("tools 2", stats)
        self.assertIn("approvals 1", stats)
        self.assertIn("retries 1", stats)

    def test_prompt_for_interrupt_always_persists_session_mode(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        snapshot = self._snapshot()
        out = self._console()

        async def selector(_summary):
            return "always"

        result = asyncio.run(
            agent_cli.prompt_for_interrupt(
                {
                    "kind": "tool_approval",
                    "tools": [
                        {
                            "name": "edit_file",
                            "args": {"path": "demo.txt"},
                            "policy": {"mutating": True, "impact_scope": "files", "ui_kind": "edit"},
                        }
                    ],
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


if __name__ == "__main__":
    unittest.main()
