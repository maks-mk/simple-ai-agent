import asyncio
import logging
from typing import List, Optional
from dotenv import load_dotenv

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    AIMessage, ToolMessage, SystemMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from core.constants import BASE_DIR
from core.config import AgentConfig
from core.state import AgentState
from core.utils import AgentUtils
from core.logging_config import setup_logging
from core.nodes import AgentNodes
from tools.tool_registry import ToolRegistry

# Setup logging
try:
    logger = setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("agent")

class AgentWorkflow:
    def __init__(self):
        load_dotenv(BASE_DIR / '.env')
        self.config = AgentConfig()
        self.utils = AgentUtils()
        self.tool_registry = ToolRegistry(self.config)
        
        self.llm: Optional[BaseChatModel] = None
        self.llm_with_tools: Optional[BaseChatModel] = None
        
        self.nodes: Optional[AgentNodes] = None

    async def initialize_resources(self):
        """Initializes LLM, Tools and Nodes."""
        logger.info(f"Initializing agent: [bold cyan]{self.config.provider}[/]", extra={"markup": True})
        
        self.llm = self.config.get_llm()
        await self.tool_registry.load_all()
        
        # Bind tools if supported
        can_use_tools = self.config.check_tool_support()
        
        if self.tool_registry.tools and can_use_tools:
            try:
                self.llm_with_tools = self.llm.bind_tools(self.tool_registry.tools)
                logger.info("🛠️ Tools bound to LLM successfully.")
            except Exception as e:
                logger.error(f"Failed to bind tools: {e}")
                self.llm_with_tools = self.llm
        else:
            if not can_use_tools:
                logger.debug("⚠️ Tools disabled: Model does not support tool calling.")
            self.llm_with_tools = self.llm

        # Initialize Nodes
        self.nodes = AgentNodes(
            config=self.config,
            llm=self.llm,
            tools=self.tools,
            llm_with_tools=self.llm_with_tools
        )

    @property
    def tools(self) -> List[BaseTool]:
        return self.tool_registry.tools

    def build_graph(self):
        if not self.nodes:
            raise RuntimeError("Resources not initialized. Call initialize_resources() first.")

        workflow = StateGraph(AgentState)

        workflow.add_node("summarize", self.nodes.summarize_node)
        workflow.add_node("agent", self.nodes.agent_node)
        workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})
        workflow.add_node("tools", self.nodes.tools_node)

        tools_enabled = bool(self.tools) and self.config.check_tool_support()

        # Simple Linear Flow: Start -> Summarize -> Update Step -> Agent -> [Tools -> Agent] or End
        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "update_step")
        workflow.add_edge("update_step", "agent")

        def should_continue(state: AgentState):
            steps = state.get("steps", 0)
            messages = state.get("messages", [])

            if steps >= self.config.max_loops:
                logger.debug(f"🛑 Loop Guard: {steps} steps.")
                return END

            if not messages: return "agent"
            last_msg = messages[-1]

            if tools_enabled and isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                return "tools"
            
            # If agent returns content without tool calls, we stop (or wait for user input in CLI loop)
            return END

        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        
        if tools_enabled:
            workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    async def main():
        wf = AgentWorkflow()
        await wf.initialize_resources()
        print(f"✔ Agent Ready. Tools: {len(wf.tools)}")

    asyncio.run(main())
