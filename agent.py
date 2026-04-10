import asyncio
import logging
from typing import Any, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from core.checkpointing import create_checkpoint_runtime
from core.config import AgentConfig
from core.logging_config import setup_logging
from core.nodes import AgentNodes
from core.run_logger import JsonlRunLogger
from core.state import AgentState
from tools.tool_registry import ToolRegistry

# Setup logging
try:
    logger = setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("agent")


# --- Factories ---

def create_llm(config: AgentConfig) -> BaseChatModel:
    """Initializes LLM based on configuration."""
    if config.provider == "gemini":
        # Lazy import to avoid loading both providers on startup.
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Безопасное извлечение ключа (защита от краша, если ключ None)
        api_key = config.gemini_api_key.get_secret_value() if config.gemini_api_key else None
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            temperature=config.temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )
    if config.provider == "openai":
        # Lazy import to avoid loading both providers on startup.
        from langchain_openai import ChatOpenAI

        api_key = config.openai_api_key.get_secret_value() if config.openai_api_key else None
        return ChatOpenAI(
            model=config.openai_model,
            temperature=config.temperature,
            api_key=api_key,
            base_url=config.openai_base_url,
            stream_usage=True,
        )
    raise ValueError(f"Unknown provider: {config.provider}")


# --- Builder ---

def create_agent_workflow(
    nodes: AgentNodes,
    config: AgentConfig,
    tools_enabled: Optional[bool] = None,
) -> StateGraph:
    """Builds the deterministic LangGraph workflow for a single user turn."""
    tools_enabled = bool(nodes.tools) and config.model_supports_tools if tools_enabled is None else tools_enabled

    workflow = StateGraph(AgentState)

    workflow.add_node("summarize", nodes.summarize_node)
    workflow.add_node("agent", nodes.agent_node)
    workflow.add_node("prepare_retry", nodes.prepare_retry_node)
    workflow.add_node("finalize_blocked", nodes.finalize_blocked_node)
    workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})

    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "update_step")
    workflow.add_edge("update_step", "agent")

    if tools_enabled:
        workflow.add_node("approval", nodes.approval_node)
        workflow.add_node("tools", nodes.tools_node)

    def route_after_agent(state: AgentState):
        steps = state.get("steps", 0)

        if steps >= config.max_loops:
            logger.debug("🛑 Loop Guard: %s steps reached.", steps)
            return "finalize_blocked"

        messages = state.get("messages") or []
        if not messages:
            logger.warning("Agent node returned no messages; routing to deterministic fallback.")
            return "prepare_retry" if nodes._has_recovery_budget(int(state.get("retry_count", 0) or 0)) else "finalize_blocked"

        last_msg = messages[-1]
        open_tool_issue = state.get("open_tool_issue")
        retry_count = int(state.get("retry_count", 0) or 0)

        if tools_enabled and isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            if nodes.tool_calls_require_approval(last_msg.tool_calls):
                return "approval"
            return "tools"

        if state.get("turn_outcome") == "finish_turn":
            return END
        if state.get("turn_outcome") == "retry_agent":
            return "prepare_retry"
        if state.get("turn_outcome") == "finalize_blocked":
            return "finalize_blocked"

        if isinstance(last_msg, AIMessage) and str(last_msg.content).strip() and not open_tool_issue:
            return END
        return "prepare_retry" if nodes._has_recovery_budget(retry_count) else "finalize_blocked"

    def route_after_tools(state: AgentState):
        issue = state.get("open_tool_issue")
        if not isinstance(issue, dict):
            return "update_step"
        if issue.get("kind") == "approval_denied":
            return "update_step"
        return "prepare_retry" if nodes._has_recovery_budget(int(state.get("retry_count", 0) or 0)) else "finalize_blocked"

    if tools_enabled:
        workflow.add_conditional_edges("agent", route_after_agent, ["approval", "tools", "prepare_retry", "finalize_blocked", END])
        workflow.add_edge("approval", "tools")
        workflow.add_conditional_edges("tools", route_after_tools, ["update_step", "prepare_retry", "finalize_blocked"])
    else:
        workflow.add_conditional_edges("agent", route_after_agent, ["prepare_retry", "finalize_blocked", END])

    workflow.add_edge("prepare_retry", "update_step")
    workflow.add_edge("finalize_blocked", END)

    return workflow

async def build_agent_app(config: Optional[AgentConfig] = None) -> Tuple[Any, ToolRegistry]:
    """
    Builds the LangGraph application and returns it along with the tool registry.
    """
    # Pydantic AgentConfig автоматически загружает .env.
    config = config or AgentConfig()

    logger.info(f"Initializing agent: [bold cyan]{config.provider}[/]", extra={"markup": True})

    # 1. Initialize Resources
    llm = create_llm(config)
    tool_registry = ToolRegistry(config)
    await tool_registry.load_all()
    checkpoint_runtime = await create_checkpoint_runtime(config)
    run_logger = JsonlRunLogger(config.run_log_dir)
    tool_registry.checkpoint_info = checkpoint_runtime.to_dict()
    tool_registry.register_cleanup_callback(checkpoint_runtime.aclose)

    # 2. Bind Tools
    tools = tool_registry.tools
    can_use_tools = config.model_supports_tools

    llm_with_tools = llm
    if tools and can_use_tools:
        try:
            llm_with_tools = llm.bind_tools(tools)
            logger.info("🛠️ Tools bound to LLM successfully.")
        except Exception as e:
            logger.error(f"Failed to bind tools: {e}")
    else:
        if not can_use_tools:
            logger.debug("⚠️ Tools disabled: Model does not support tool calling.")

    # 3. Create Nodes
    nodes = AgentNodes(
        config=config,
        llm=llm,
        tools=tools,
        llm_with_tools=llm_with_tools,
        tool_metadata=tool_registry.tool_metadata,
        run_logger=run_logger,
    )
    workflow = create_agent_workflow(nodes, config, tools_enabled=bool(tools) and can_use_tools)
    return workflow.compile(checkpointer=checkpoint_runtime.checkpointer), tool_registry


if __name__ == "__main__":

    async def main():
        app, registry = await build_agent_app()
        print(f"✔ Agent Ready. Tools: {len(registry.tools)}")
        await registry.cleanup()

    asyncio.run(main())
