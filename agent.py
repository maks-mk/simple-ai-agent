import asyncio
import logging
from typing import List, Optional, Tuple, Any
from dotenv import load_dotenv

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from core.constants import BASE_DIR
from core.config import AgentConfig
from core.state import AgentState
from core.logging_config import setup_logging
from core.nodes import AgentNodes
from tools.tool_registry import ToolRegistry

# LLM Imports (Moved from config.py)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

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
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            temperature=config.temperature,
            google_api_key=config.gemini_api_key.get_secret_value(),
            convert_system_message_to_human=True
        )
    elif config.provider == "openai":
        return ChatOpenAI(
            model=config.openai_model,
            temperature=config.temperature,
            api_key=config.openai_api_key.get_secret_value(),
            base_url=config.openai_base_url,
            stream_usage=True
        )
    raise ValueError(f"Unknown provider: {config.provider}")

def check_tool_support(config: AgentConfig) -> bool:
    """Determines if the current model supports tool calling."""
    # Полностью полагаемся на явную настройку в .env (MODEL_SUPPORTS_TOOLS)
    return config.model_supports_tools

# --- Builder ---

async def build_agent_app() -> Tuple[Any, ToolRegistry]:
    """
    Builds the LangGraph application and returns it along with the tool registry.
    """
    load_dotenv(BASE_DIR / '.env')
    config = AgentConfig()
    
    logger.info(f"Initializing agent: [bold cyan]{config.provider}[/]", extra={"markup": True})
    logger.debug(f"Prompt Path: {config.prompt_path.absolute()}")

    # 1. Initialize Resources
    llm = create_llm(config)
    tool_registry = ToolRegistry(config)
    await tool_registry.load_all()
    
    # 2. Bind Tools
    tools = tool_registry.tools
    can_use_tools = check_tool_support(config)
    
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
        llm_with_tools=llm_with_tools
    )

    # 4. Build Graph
    workflow = StateGraph(AgentState)

    workflow.add_node("summarize", nodes.summarize_node)
    workflow.add_node("agent", nodes.agent_node)
    workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})
    workflow.add_node("tools", nodes.tools_node)

    # Simple Linear Flow: Start -> Summarize -> Update Step -> Agent -> [Tools -> Agent] or End
    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "update_step")
    workflow.add_edge("update_step", "agent")

    def should_continue(state: AgentState):
        steps = state.get("steps", 0)
        messages = state.get("messages", [])

        if steps >= config.max_loops:
            logger.debug(f"🛑 Loop Guard: {steps} steps.")
            return END

        if not messages: return "agent"
        last_msg = messages[-1]

        # Only go to tools if tools are actually enabled/bound
        if tools and can_use_tools and isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"
        
        return END

    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    
    if tools and can_use_tools:
        workflow.add_edge("tools", "update_step")

    return workflow.compile(checkpointer=MemorySaver()), tool_registry

if __name__ == "__main__":
    async def main():
        app, registry = await build_agent_app()
        print(f"✔ Agent Ready. Tools: {len(registry.tools)}")
        await registry.cleanup()

    asyncio.run(main())
