import json
import os
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent import create_agent_workflow
from core.checkpointing import create_checkpoint_runtime
from core.config import AgentConfig
from core.nodes import AgentNodes
from core.run_logger import JsonlRunLogger
from core.session_store import SessionStore
from core.stream_processor import StreamProcessor
from core.tool_policy import ToolMetadata
from tools import process_tools


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    async def ainvoke(self, context):
        self.invocations.append(context)
        if not self.responses:
            return AIMessage(content="STATUS: FINISHED\nREASON: fallback\nNEXT_STEP: NONE")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.description = f"Fake tool {name}"
        self.result = result
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        if callable(self.result):
            return self.result(args)
        return self.result


class RuntimeRefactorTests(unittest.IsolatedAsyncioTestCase):
    def _workspace_tempdir(self) -> Path:
        path = Path.cwd() / ".tmp_tests" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

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

    def _initial_state(self, task="Проверь задачу", session_id="session-test", run_id="run-test"):
        return {
            "messages": [("user", task)],
            "steps": 0,
            "token_usage": {},
            "current_task": task,
            "critic_status": "",
            "critic_source": "",
            "critic_feedback": "",
            "session_id": session_id,
            "run_id": run_id,
            "pending_approval": None,
            "last_tool_error": "",
            "last_tool_result": "",
            "safety_mode": "default",
        }

    async def test_create_checkpoint_runtime_uses_sqlite_backend_when_available(self):
        tmp = self._workspace_tempdir()
        db_path = tmp / "checkpoints.sqlite"
        runtime = await create_checkpoint_runtime(
            self._make_config(CHECKPOINT_BACKEND="sqlite", CHECKPOINT_SQLITE_PATH=db_path)
        )
        try:
            self.assertEqual(runtime.resolved_backend, "sqlite")
            self.assertEqual(Path(runtime.target), db_path.resolve())
            self.assertTrue(db_path.exists())
        finally:
            await runtime.aclose()

    async def test_sqlite_checkpointer_persists_state_across_recompiled_app(self):
        tmp = self._workspace_tempdir()
        db_path = tmp / "persist.sqlite"
        config = self._make_config(CHECKPOINT_BACKEND="sqlite", CHECKPOINT_SQLITE_PATH=db_path)
        thread_config = {"configurable": {"thread_id": "persist-thread"}, "recursion_limit": 24}

        runtime1 = await create_checkpoint_runtime(config)
        app1 = create_agent_workflow(
            AgentNodes(
                config=config,
                llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: ok\nNEXT_STEP: NONE")]),
                tools=[],
                llm_with_tools=FakeLLM([AIMessage(content="Первый ответ.")]),
            ),
            config,
            tools_enabled=False,
        ).compile(checkpointer=runtime1.checkpointer)
        await app1.ainvoke(self._initial_state("Первая задача"), config=thread_config)
        await runtime1.aclose()

        runtime2 = await create_checkpoint_runtime(config)
        try:
            app2 = create_agent_workflow(
                AgentNodes(
                    config=config,
                    llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: ok\nNEXT_STEP: NONE")]),
                    tools=[],
                    llm_with_tools=FakeLLM([AIMessage(content="Второй ответ.")]),
                ),
                config,
                tools_enabled=False,
            ).compile(checkpointer=runtime2.checkpointer)
            saved_state = await app2.aget_state({"configurable": {"thread_id": "persist-thread"}})
            saved_messages = saved_state.values["messages"]
            self.assertTrue(any(isinstance(msg, HumanMessage) and msg.content == "Первая задача" for msg in saved_messages))
            self.assertTrue(any(isinstance(msg, AIMessage) and msg.content == "Первый ответ." for msg in saved_messages))
        finally:
            await runtime2.aclose()

    async def test_approval_interrupt_requires_resume_before_mutating_tool_runs(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-1"}]),
                    AIMessage(content="Готово."),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])

        resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)
        self.assertEqual(tool.calls, [{"action": "apply"}])
        self.assertEqual(resumed["messages"][-1].content, "Готово.")

    async def test_approval_rejection_returns_access_denied_without_tool_execution(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: blocker reported\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-2"}]),
                    AIMessage(content="Не удалось выполнить действие без подтверждения."),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread-2"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        resumed = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(tool.calls, [])
        tool_messages = [msg for msg in resumed["messages"] if isinstance(msg, ToolMessage)]
        self.assertTrue(tool_messages)
        self.assertIn("ACCESS_DENIED", str(tool_messages[-1].content))

    async def test_run_logger_writes_structured_tool_failure_event(self):
        tmp = self._workspace_tempdir()
        logger = JsonlRunLogger(tmp)
        failing_tool = FakeTool("danger_tool", "ERROR[EXECUTION]: boom")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[failing_tool],
            llm_with_tools=FakeLLM([]),
            run_logger=logger,
        )

        await nodes.tools_node(
            {
                **self._initial_state("Почини ошибку", session_id="session-log", run_id="run-log"),
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "danger_tool", "args": {"action": "x"}, "id": "tc-log"}],
                    )
                ],
            }
        )

        log_path = logger.file_path_for("session-log")
        records = [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]
        end_records = [record for record in records if record["event"] == "tool_call_end"]
        self.assertTrue(end_records)
        self.assertEqual(end_records[-1]["result"]["ok"], False)
        self.assertEqual(end_records[-1]["result"]["error_type"], "EXECUTION")

    def test_session_store_round_trip(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        snapshot = store.new_session(checkpoint_backend="sqlite", checkpoint_target="demo.sqlite")
        store.save_active_session(snapshot)
        loaded = store.load_active_session()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.session_id, snapshot.session_id)
        self.assertEqual(loaded.thread_id, snapshot.thread_id)

    def test_stop_background_process_denies_external_pid_by_default(self):
        result = process_tools.stop_background_process.invoke({"pid": os.getpid()})
        self.assertIn("ACCESS_DENIED", result)

    def test_stream_processor_ignores_tool_call_without_id(self):
        import io
        from rich.console import Console

        capture = io.StringIO()
        processor = StreamProcessor(Console(record=True, file=capture, width=120, force_terminal=False))
        processor._remember_tool_call({"name": "broken_tool", "args": {"x": 1}})
        self.assertEqual(processor.tool_buffer, {})


if __name__ == "__main__":
    unittest.main()
