import io
import os
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from rich.console import Console

from agent import create_agent_workflow
from core.config import AgentConfig
from core.nodes import AgentNodes
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
            return AIMessage(content="")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ProviderSafeFakeLLM(FakeLLM):
    async def ainvoke(self, context):
        last_visible = next((message for message in reversed(context) if message.type != "system"), None)
        if isinstance(last_visible, AIMessage):
            raise AssertionError("provider-unsafe assistant-last context")
        return await super().ainvoke(context)


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
    def _make_config(self, **overrides):
        defaults = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "PROMPT_PATH": os.path.join(os.path.dirname(__file__), "..", "prompt.txt"),
            "MCP_CONFIG_PATH": os.path.join(os.path.dirname(__file__), "missing_mcp.json"),
            "ENABLE_SEARCH_TOOLS": False,
            "ENABLE_SYSTEM_TOOLS": False,
            "ENABLE_PROCESS_TOOLS": False,
            "ENABLE_SHELL_TOOL": False,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def _initial_state(self, task="Проверь задачу", session_id="session-test", run_id="run-test"):
        return {
            "messages": [HumanMessage(content=task)],
            "steps": 0,
            "token_usage": {},
            "current_task": task,
            "retry_count": 0,
            "retry_reason": "",
            "turn_outcome": "",
            "final_issue": "",
            "session_id": session_id,
            "run_id": run_id,
            "turn_id": 1,
            "pending_approval": None,
            "open_tool_issue": None,
            "last_tool_error": "",
            "last_tool_result": "",
            "safety_mode": "default",
        }

    def _graph_config(self, thread_id="thread-test", recursion_limit=24):
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}

    async def test_final_answer_finishes_turn_without_reviewer_node(self):
        config = self._make_config()
        app = create_agent_workflow(
            AgentNodes(
                config=config,
                llm=FakeLLM([]),
                tools=[],
                llm_with_tools=FakeLLM([AIMessage(content="Задача выполнена.")]),
            ),
            config,
            tools_enabled=False,
        ).compile(checkpointer=MemorySaver())

        result = await app.ainvoke(self._initial_state("Скажи готово"), config=self._graph_config("thread-final", 24))

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(result["messages"][-1].content, "Задача выполнена.")
        self.assertNotIn("critic_status", result)

    async def test_empty_response_retries_until_recovery_budget_exhausted(self):
        config = self._make_config(MAX_RECOVERY_ATTEMPTS=2)
        agent_llm = FakeLLM([AIMessage(content=""), AIMessage(content="")])
        app = create_agent_workflow(
            AgentNodes(config=config, llm=FakeLLM([]), tools=[], llm_with_tools=agent_llm),
            config,
            tools_enabled=False,
        ).compile(checkpointer=MemorySaver())

        result = await app.ainvoke(self._initial_state("Сделай задачу"), config=self._graph_config("thread-empty", 24))

        self.assertEqual(len(agent_llm.invocations), 3)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIn("Не удалось завершить задачу", str(result["messages"][-1].content))
        self.assertEqual(result["retry_count"], 2)

    async def test_valid_tool_call_routes_to_tool_execution_then_finishes(self):
        config = self._make_config()
        tool = FakeTool("demo_tool", "Готово.")
        agent_llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "run"}, "id": "tc-1"}]),
                AIMessage(content="Инструмент выполнил задачу."),
            ]
        )
        app = create_agent_workflow(
            AgentNodes(config=config, llm=FakeLLM([]), tools=[tool], llm_with_tools=agent_llm),
            config,
            tools_enabled=True,
        ).compile(checkpointer=MemorySaver())

        result = await app.ainvoke(self._initial_state("Сделай дело"), config=self._graph_config("thread-tool", 36))

        self.assertEqual(tool.calls, [{"action": "run"}])
        self.assertEqual(result["messages"][-1].content, "Инструмент выполнил задачу.")
        self.assertEqual(result["turn_outcome"], "finish_turn")

    async def test_approval_interrupt_requires_resume_before_mutating_tool_runs(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-2"}]),
                    AIMessage(content="Готово."),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                    impact_scope="local_state",
                    ui_kind="process",
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

    async def test_approval_denial_blocks_followup_tool_calls_in_same_turn(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        denied_tool = FakeTool("danger_tool", "Изменение применено.")
        fallback_tool = FakeTool("write_file", "Файл записан.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[denied_tool, fallback_tool],
            llm_with_tools=ProviderSafeFakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-3"}]),
                    AIMessage(
                        content="Сохраню результат в другой файл.",
                        tool_calls=[{"name": "write_file", "args": {"path": "alt.md", "content": "x"}, "id": "tc-4"}],
                    ),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(name="danger_tool", mutating=True, destructive=True, requires_approval=True),
                "write_file": ToolMetadata(name="write_file", mutating=True, requires_approval=True),
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread-2"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        resumed = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(denied_tool.calls, [])
        self.assertEqual(fallback_tool.calls, [])
        self.assertIn("you chose No", str(resumed["messages"][-1].content))
        self.assertIsNone(resumed["open_tool_issue"])
        self.assertEqual(resumed["turn_outcome"], "finish_turn")

    async def test_tool_failure_produces_bounded_recovery_then_blocker(self):
        config = self._make_config(MAX_RECOVERY_ATTEMPTS=2)
        failing_tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        agent_llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-5"}]),
                AIMessage(content=""),
            ]
        )
        app = create_agent_workflow(
            AgentNodes(config=config, llm=FakeLLM([]), tools=[failing_tool], llm_with_tools=agent_llm),
            config,
            tools_enabled=True,
        ).compile(checkpointer=MemorySaver())

        result = await app.ainvoke(self._initial_state("Обработай файл"), config=self._graph_config("thread-fail", 36))

        self.assertEqual(len(agent_llm.invocations), 3)
        self.assertIn("Не удалось завершить задачу", str(result["messages"][-1].content))
        self.assertEqual(result["retry_count"], 2)

    async def test_successful_tool_progress_resets_recovery_budget_for_later_failures(self):
        config = self._make_config(MAX_RECOVERY_ATTEMPTS=1)

        def tool_result(args):
            action = args.get("action")
            if action in {"fail-1", "fail-2"}:
                return "ERROR[EXECUTION]: transient failure"
            return "Success: progress made"

        tool = FakeTool("demo_tool", tool_result)
        agent_llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "fail-1"}, "id": "tc-r1"}]),
                AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "success"}, "id": "tc-r2"}]),
                AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "fail-2"}, "id": "tc-r3"}]),
                AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "success-2"}, "id": "tc-r4"}]),
                AIMessage(content="Удалось продолжить после второй ошибки."),
            ]
        )
        app = create_agent_workflow(
            AgentNodes(config=config, llm=FakeLLM([]), tools=[tool], llm_with_tools=agent_llm),
            config,
            tools_enabled=True,
        ).compile(checkpointer=MemorySaver())

        result = await app.ainvoke(self._initial_state("Продолжай несмотря на ошибки"), config=self._graph_config("thread-reset", 48))

        self.assertEqual(
            tool.calls,
            [
                {"action": "fail-1"},
                {"action": "success"},
                {"action": "fail-2"},
                {"action": "success-2"},
            ],
        )
        self.assertEqual(result["messages"][-1].content, "Удалось продолжить после второй ошибки.")
        self.assertEqual(result["turn_outcome"], "finish_turn")

    async def test_new_user_turn_ignores_old_open_tool_issue(self):
        agent_llm = FakeLLM([AIMessage(content="Короткая сводка на экране.")])
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[],
            llm_with_tools=agent_llm,
        )
        state = {
            **self._initial_state("Покажи коротко эту инфу на экран"),
            "messages": [
                HumanMessage(content="Сохрани в файл"),
                AIMessage(content="Не сделал, так как вы выбрали Нет. Ожидаю дальнейших инструкций."),
                HumanMessage(content="Покажи коротко эту инфу на экран"),
            ],
            "turn_id": 1,
            "open_tool_issue": {
                "turn_id": 1,
                "kind": "approval_denied",
                "summary": "Execution of 'write_file' was cancelled by approval policy.",
                "tool_names": ["write_file"],
                "source": "approval",
            },
            "current_task": "Покажи коротко эту инфу на экран",
        }

        result = await nodes.agent_node(state)

        self.assertEqual(result["turn_id"], 2)
        self.assertIsNone(result["open_tool_issue"])

    def test_stream_processor_ignores_tool_call_without_id(self):
        processor = StreamProcessor(Console(record=True, file=io.StringIO(), width=120, force_terminal=False))
        processor._remember_tool_call({"name": "broken_tool", "args": {"x": 1}})
        self.assertEqual(processor.tool_buffer, {})

    def test_stop_background_process_denies_unknown_pid(self):
        result = process_tools.stop_background_process.invoke({"pid": 999999})
        self.assertTrue("ACCESS_DENIED" in result or "NOT_FOUND" in result)


if __name__ == "__main__":
    unittest.main()
