import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.messages import HumanMessage

from agent import create_agent_workflow
from core.config import AgentConfig
from core.nodes import AgentNodes


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


class CriticGraphTests(unittest.IsolatedAsyncioTestCase):
    def _make_config(self, *, model_supports_tools=True, max_loops=6, max_retries=3, retry_delay=0):
        return AgentConfig(
            provider="openai",
            openai_api_key="test-key",
            model_supports_tools=model_supports_tools,
            max_loops=max_loops,
            max_retries=max_retries,
            retry_delay=retry_delay,
            prompt_path=Path(__file__).resolve().parents[1] / "prompt.txt",
        )

    def _build_app(
        self,
        *,
        agent_responses,
        critic_responses,
        tools=None,
        model_supports_tools=True,
        max_loops=6,
        max_retries=3,
        retry_delay=0,
    ):
        config = self._make_config(
            model_supports_tools=model_supports_tools,
            max_loops=max_loops,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        agent_llm = FakeLLM(agent_responses)
        critic_llm = FakeLLM(critic_responses)
        nodes = AgentNodes(
            config=config,
            llm=critic_llm,
            tools=tools or [],
            llm_with_tools=agent_llm,
        )
        workflow = create_agent_workflow(
            nodes,
            config,
            tools_enabled=bool(tools) and model_supports_tools,
        )
        app = workflow.compile()
        return app, agent_llm, critic_llm

    def _initial_state(self, task="Проверь задачу"):
        return {
            "messages": [("user", task)],
            "steps": 0,
            "token_usage": {},
            "current_task": task,
            "critic_status": "",
            "critic_source": "",
            "critic_feedback": "",
            "turn_id": 1,
            "open_tool_issue": None,
        }

    def _assert_feedback_hidden(self, messages):
        joined = "\n".join(str(message.content) for message in messages)
        self.assertNotIn("Task incomplete.", joined)
        self.assertNotIn("Task appears complete based on the tool results.", joined)
        self.assertNotIn("Task may still be incomplete.", joined)

    async def test_agent_final_answer_requires_critic_finish(self):
        app, agent_llm, critic_llm = self._build_app(
            agent_responses=[AIMessage(content="Задача выполнена.")],
            critic_responses=[AIMessage(content="STATUS: FINISHED\nREASON: Goal achieved.\nNEXT_STEP: NONE")],
            tools=[],
            model_supports_tools=False,
        )

        result = await app.ainvoke(self._initial_state("Скажи готово"), config={"recursion_limit": 24})

        self.assertEqual(result["critic_status"], "FINISHED")
        self.assertEqual(result["critic_source"], "agent")
        self.assertEqual(result["messages"][-1].content, "Задача выполнена.")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 1)

    async def test_incomplete_final_answer_after_tools_loops_back_with_critic_feedback(self):
        tool = FakeTool("demo_tool", "Apache installed, but service is stopped.")
        app, agent_llm, critic_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "demo_tool", "args": {"action": "install"}, "id": "tc-1"}],
                ),
                AIMessage(content="Apache установлен."),
                AIMessage(content="Сервис запущен и проверен."),
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: The final answer does not confirm the service is running.\n"
                        "NEXT_STEP: start the service and verify port 80"
                    )
                ),
                AIMessage(content="STATUS: FINISHED\nREASON: Verification passed.\nNEXT_STEP: NONE"),
            ],
            tools=[tool],
        )

        result = await app.ainvoke(self._initial_state("Установи Apache"), config={"recursion_limit": 36})

        self.assertEqual(tool.calls, [{"action": "install"}])
        self.assertEqual(result["messages"][-1].content, "Сервис запущен и проверен.")
        self.assertEqual(len(agent_llm.invocations), 3)
        self.assertEqual(len(critic_llm.invocations), 2)
        critic_hint = [
            message.content
            for message in agent_llm.invocations[2]
            if isinstance(message, SystemMessage) and "INTERNAL CRITIC FEEDBACK" in str(message.content)
        ]
        self.assertTrue(critic_hint)
        self.assertIn("Task incomplete.", critic_hint[0])
        retry_context = [
            message.content
            for message in agent_llm.invocations[2]
            if isinstance(message, SystemMessage) and "RETRY CONTEXT:" in str(message.content)
        ]
        self.assertTrue(retry_context)
        self._assert_feedback_hidden(result["messages"])

    async def test_tools_flow_reaches_final_answer_before_running_critic(self):
        tool = FakeTool("demo_tool", "Deployment complete and service responds with 200 OK.")
        app, agent_llm, critic_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "demo_tool", "args": {"action": "deploy"}, "id": "tc-2"}],
                ),
                AIMessage(content="Деплой завершён, сервис отвечает 200 OK."),
            ],
            critic_responses=[
                AIMessage(content="STATUS: FINISHED\nREASON: Deployment is complete.\nNEXT_STEP: NONE"),
                AIMessage(content="STATUS: FINISHED\nREASON: Final answer delivered.\nNEXT_STEP: NONE"),
            ],
            tools=[tool],
        )

        result = await app.ainvoke(self._initial_state("Задеплой сервис"), config={"recursion_limit": 36})

        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertEqual(len(critic_llm.invocations), 1)
        self.assertEqual(result["messages"][-1].content, "Деплой завершён, сервис отвечает 200 OK.")
        critic_hint = [
            message.content
            for message in agent_llm.invocations[1]
            if isinstance(message, SystemMessage) and "INTERNAL CRITIC FEEDBACK" in str(message.content)
        ]
        self.assertFalse(critic_hint)
        self._assert_feedback_hidden(result["messages"])

    async def test_chat_only_mode_still_runs_critic(self):
        app, agent_llm, critic_llm = self._build_app(
            agent_responses=[AIMessage(content="Готово без инструментов.")],
            critic_responses=[AIMessage(content="STATUS: FINISHED\nREASON: Answer is sufficient.\nNEXT_STEP: NONE")],
            tools=[],
            model_supports_tools=False,
        )

        result = await app.ainvoke(self._initial_state("Ответь без tools"), config={"recursion_limit": 24})

        self.assertEqual(result["messages"][-1].content, "Готово без инструментов.")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 1)

    async def test_malformed_critic_verdict_uses_soft_incomplete_fallback_for_ambiguous_answer(self):
        app, agent_llm, critic_llm = self._build_app(
            agent_responses=[
                AIMessage(content="Похоже, всё готово."),
                AIMessage(content="Я перепроверил результат и теперь всё подтверждено."),
            ],
            critic_responses=[
                AIMessage(content="Looks good to me."),
                AIMessage(content="STATUS: FINISHED\nREASON: Explicit verification completed.\nNEXT_STEP: NONE"),
            ],
            tools=[],
            model_supports_tools=False,
        )

        result = await app.ainvoke(self._initial_state("Сделай и проверь"), config={"recursion_limit": 36})

        self.assertEqual(result["messages"][-1].content, "Я перепроверил результат и теперь всё подтверждено.")
        self.assertEqual(len(agent_llm.invocations), 2)
        feedback_messages = [
            message.content
            for message in agent_llm.invocations[1]
            if isinstance(message, SystemMessage) and "INTERNAL CRITIC FEEDBACK" in str(message.content)
        ]
        self.assertTrue(feedback_messages)
        self.assertIn("Task incomplete.", feedback_messages[0])
        self.assertIn("latest assistant message suggests work is still pending", feedback_messages[0])
        self._assert_feedback_hidden(result["messages"])

    async def test_malformed_critic_verdict_finishes_for_clear_final_answer(self):
        app, agent_llm, critic_llm = self._build_app(
            agent_responses=[AIMessage(content="PDF-файл успешно создан и сохранён как report.pdf.")],
            critic_responses=[AIMessage(content="Looks good to me.")],
            tools=[],
            model_supports_tools=False,
        )

        result = await app.ainvoke(self._initial_state("Собери данные и сохрани в PDF"), config={"recursion_limit": 24})

        self.assertEqual(result["messages"][-1].content, "PDF-файл успешно создан и сохранён как report.pdf.")
        self.assertEqual(result["critic_status"], "FINISHED")
        self.assertEqual(result["critic_source"], "agent")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 1)
        self._assert_feedback_hidden(result["messages"])

    async def test_multiple_critic_rounds_fit_into_recursion_budget(self):
        app, agent_llm, critic_llm = self._build_app(
            agent_responses=[
                AIMessage(content="Черновой результат."),
                AIMessage(content="Ещё одна проверка выполнена."),
                AIMessage(content="Итог подтверждён."),
            ],
            critic_responses=[
                AIMessage(content="STATUS: INCOMPLETE\nREASON: Need explicit verification.\nNEXT_STEP: verify output"),
                AIMessage(content="STATUS: INCOMPLETE\nREASON: Verification still weak.\nNEXT_STEP: confirm final state"),
                AIMessage(content="STATUS: FINISHED\nREASON: Final confirmation present.\nNEXT_STEP: NONE"),
            ],
            tools=[],
            model_supports_tools=False,
            max_loops=5,
        )

        result = await app.ainvoke(self._initial_state("Сделай несколько проверок"), config={"recursion_limit": 30})

        self.assertEqual(result["messages"][-1].content, "Итог подтверждён.")
        self.assertEqual(len(agent_llm.invocations), 3)
        self.assertEqual(len(critic_llm.invocations), 3)

    async def test_fatal_llm_error_stops_graph_without_reentering_critic(self):
        fatal_error = Exception(
            "Error code: 402 - {'error': {'code': '402', 'message': 'Insufficient account balance', 'type': 'insufficient_balance'}}"
        )
        app, agent_llm, critic_llm = self._build_app(
            agent_responses=[fatal_error, fatal_error, fatal_error],
            critic_responses=[],
            tools=[],
            model_supports_tools=False,
            max_retries=3,
            retry_delay=0,
        )

        with self.assertRaises(Exception) as ctx:
            await app.ainvoke(self._initial_state("Собери информацию"), config={"recursion_limit": 24})

        self.assertIn("402", str(ctx.exception))
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 0)

    async def test_invalid_tool_calls_become_protocol_error_not_unknown_tool(self):
        config = self._make_config(model_supports_tools=True)
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[FakeTool("demo_tool", "ok")],
            llm_with_tools=FakeLLM([
                AIMessage(
                    content="",
                    invalid_tool_calls=[{"name": "demo_tool", "args": "{", "id": "broken-1", "error": "bad json"}],
                )
            ]),
        )

        result = await nodes.agent_node(self._initial_state("Сделай вызов инструмента"))
        response = result["messages"][0]

        self.assertIsInstance(response, AIMessage)
        self.assertFalse(response.tool_calls)
        self.assertNotIn("unknown_tool", str(response.content))
        self.assertIn("INTERNAL TOOL PROTOCOL ERROR", str(response.content))
        self.assertEqual(result["critic_source"], "agent")

    async def test_tool_errors_no_longer_store_fake_critic_feedback(self):
        config = self._make_config(model_supports_tools=True)
        failing_tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[failing_tool],
            llm_with_tools=FakeLLM([]),
        )

        result = await nodes.tools_node(
            {
                **self._initial_state("Почини ошибку"),
                "messages": [
                    AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-9"}])
                ],
            }
        )

        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].name, "demo_tool")
        self.assertEqual(result["critic_feedback"], "")

    async def test_agent_context_includes_unresolved_tool_failure_system_message(self):
        config = self._make_config(model_supports_tools=True)
        failing_tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        agent_llm = FakeLLM([AIMessage(content="Не удалось завершить задачу из-за ошибки инструмента.")])
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[failing_tool],
            llm_with_tools=agent_llm,
        )

        await nodes.agent_node(
            {
                **self._initial_state("Почини ошибку"),
                "messages": [
                    HumanMessage(content="Почини ошибку"),
                    AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-10"}]),
                    ToolMessage(content="ERROR[EXECUTION]: boom", tool_call_id="tc-10", name="demo_tool"),
                ],
                "open_tool_issue": {
                    "turn_id": 1,
                    "kind": "tool_error",
                    "summary": "boom",
                    "tool_names": ["demo_tool"],
                    "source": "tools",
                },
            }
        )

        unresolved_messages = [
            str(message.content)
            for message in agent_llm.invocations[0]
            if isinstance(message, SystemMessage) and "UNRESOLVED TOOL FAILURE" in str(message.content)
        ]
        self.assertTrue(unresolved_messages)

    async def test_unresolved_tool_error_blocks_fake_success_until_agent_reports_blocker(self):
        failing_tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        app, agent_llm, critic_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-11"}],
                ),
                AIMessage(content="Файл успешно обработан и всё готово."),
                AIMessage(content="Не удалось завершить задачу: инструмент завершился с ошибкой."),
            ],
            critic_responses=[
                AIMessage(content="STATUS: FINISHED\nREASON: Looks complete.\nNEXT_STEP: NONE"),
                AIMessage(content="STATUS: FINISHED\nREASON: Honest blocker report delivered.\nNEXT_STEP: NONE"),
            ],
            tools=[failing_tool],
        )

        result = await app.ainvoke(self._initial_state("Обработай файл"), config={"recursion_limit": 36})

        self.assertEqual(result["messages"][-1].content, "Не удалось завершить задачу: инструмент завершился с ошибкой.")
        self.assertEqual(result["critic_status"], "FINISHED")
        self.assertEqual(len(agent_llm.invocations), 3)
        self.assertEqual(len(critic_llm.invocations), 2)
        final_hint_messages = [
            str(message.content)
            for message in agent_llm.invocations[2]
            if isinstance(message, SystemMessage) and "UNRESOLVED TOOL FAILURE" in str(message.content)
        ]
        self.assertTrue(final_hint_messages)

    async def test_tool_failure_sets_open_tool_issue_for_current_turn(self):
        config = self._make_config(model_supports_tools=True)
        failing_tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[failing_tool],
            llm_with_tools=FakeLLM([]),
        )

        result = await nodes.tools_node(
            {
                **self._initial_state("Почини ошибку"),
                "messages": [
                    AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-12"}])
                ],
            }
        )

        self.assertEqual(result["turn_id"], 1)
        self.assertEqual(result["open_tool_issue"]["kind"], "tool_error")
        self.assertEqual(result["open_tool_issue"]["tool_names"], ["demo_tool"])


if __name__ == "__main__":
    unittest.main()
