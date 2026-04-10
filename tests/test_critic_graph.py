import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent import create_agent_workflow
from core.config import AgentConfig
from core.nodes import AgentNodes
from core.tool_policy import ToolMetadata


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    async def ainvoke(self, context):
        self.invocations.append(context)
        if not self.responses:
            return AIMessage(content="")
        return self.responses.pop(0)


class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.description = f"Fake tool {name}"
        self.result = result
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return self.result


class DeterministicGraphTests(unittest.IsolatedAsyncioTestCase):
    def _make_config(self, **overrides):
        defaults = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "PROMPT_PATH": "prompt.txt",
            "MCP_CONFIG_PATH": "tests/missing_mcp.json",
            "ENABLE_SEARCH_TOOLS": False,
            "ENABLE_SYSTEM_TOOLS": False,
            "ENABLE_PROCESS_TOOLS": False,
            "ENABLE_SHELL_TOOL": False,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def _initial_state(self, task="Проверь задачу"):
        return {
            "messages": [HumanMessage(content=task)],
            "steps": 0,
            "token_usage": {},
            "current_task": task,
            "retry_count": 0,
            "retry_reason": "",
            "turn_outcome": "",
            "final_issue": "",
            "session_id": "session-test",
            "run_id": "run-test",
            "turn_id": 1,
            "pending_approval": None,
            "open_tool_issue": None,
            "last_tool_error": "",
            "last_tool_result": "",
            "safety_mode": "default",
        }

    def _graph_config(self, thread_id="thread-test", recursion_limit=24):
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}

    async def test_final_answer_ends_without_reviewer_node(self):
        config = self._make_config()
        agent_llm = FakeLLM([AIMessage(content="Готово.")])
        app = create_agent_workflow(
            AgentNodes(config=config, llm=FakeLLM([]), tools=[], llm_with_tools=agent_llm),
            config,
            tools_enabled=False,
        ).compile(checkpointer=MemorySaver())

        result = await app.ainvoke(self._initial_state("Скажи готово"), config=self._graph_config("critic-final", 24))

        self.assertEqual(result["messages"][-1].content, "Готово.")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 1)

    async def test_valid_tool_call_routes_to_tools_then_final_answer(self):
        config = self._make_config()
        tool = FakeTool("demo_tool", "ok")
        agent_llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"x": 1}, "id": "tc-1"}]),
                AIMessage(content="Инструмент сработал."),
            ]
        )
        app = create_agent_workflow(
            AgentNodes(config=config, llm=FakeLLM([]), tools=[tool], llm_with_tools=agent_llm),
            config,
            tools_enabled=True,
        ).compile(checkpointer=MemorySaver())

        result = await app.ainvoke(self._initial_state("Сделай"), config=self._graph_config("critic-tool", 36))

        self.assertEqual(tool.calls, [{"x": 1}])
        self.assertEqual(result["messages"][-1].content, "Инструмент сработал.")
        self.assertEqual(result["turn_outcome"], "finish_turn")

    async def test_empty_response_triggers_one_retry_then_finalize_blocked(self):
        config = self._make_config()
        agent_llm = FakeLLM([AIMessage(content=""), AIMessage(content="")])
        app = create_agent_workflow(
            AgentNodes(config=config, llm=FakeLLM([]), tools=[], llm_with_tools=agent_llm),
            config,
            tools_enabled=False,
        ).compile(checkpointer=MemorySaver())

        result = await app.ainvoke(self._initial_state("Сделай"), config=self._graph_config("critic-empty", 24))

        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertIn("Не удалось завершить задачу", str(result["messages"][-1].content))
        self.assertEqual(result["retry_count"], 1)

    async def test_malformed_tool_call_retries_once_then_stops_deterministically(self):
        config = self._make_config(model_supports_tools=True)
        agent_llm = FakeLLM(
            [
                AIMessage(content="", invalid_tool_calls=[{"name": "demo_tool", "args": "{", "id": "bad", "error": "json"}]),
                AIMessage(content=""),
            ]
        )
        nodes = AgentNodes(config=config, llm=FakeLLM([]), tools=[FakeTool("demo_tool", "ok")], llm_with_tools=agent_llm)
        app = create_agent_workflow(nodes, config, tools_enabled=False).compile(checkpointer=MemorySaver())

        result = await app.ainvoke(self._initial_state("Сделай вызов"), config=self._graph_config("critic-protocol", 24))

        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertIn("Не удалось завершить задачу", str(result["messages"][-1].content))

    async def test_tool_failure_then_finalize_blocked(self):
        config = self._make_config()
        tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        agent_llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"x": 1}, "id": "tc-2"}]),
                AIMessage(content=""),
            ]
        )
        app = create_agent_workflow(
            AgentNodes(config=config, llm=FakeLLM([]), tools=[tool], llm_with_tools=agent_llm),
            config,
            tools_enabled=True,
        ).compile(checkpointer=MemorySaver())

        result = await app.ainvoke(self._initial_state("Сделай"), config=self._graph_config("critic-fail", 36))

        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertIn("Не удалось завершить задачу", str(result["messages"][-1].content))

    async def test_approval_denial_never_runs_tools(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "ok")
        agent_llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"x": 1}, "id": "tc-3"}]),
                AIMessage(content="Я не выполнил действие."),
            ]
        )
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "danger_tool": ToolMetadata(name="danger_tool", mutating=True, destructive=True, requires_approval=True)
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-denied"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)

        result = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(tool.calls, [])
        self.assertEqual(result["turn_outcome"], "finish_turn")


if __name__ == "__main__":
    unittest.main()
