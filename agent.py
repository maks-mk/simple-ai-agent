import asyncio
import logging
from typing import List, Optional, Dict
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
        
        # Caches
        self.tool_buckets = {"safe": [], "write": []}
        self.nodes: Optional[AgentNodes] = None

    async def initialize_resources(self):
        """Initializes LLM, Tools and Nodes."""
        logger.info(f"Initializing agent: [bold cyan]{self.config.provider}[/]", extra={"markup": True})
        
        self.llm = self.config.get_llm()
        await self.tool_registry.load_all()
        
        # 1. Classify tools
        self.tool_buckets = self._classify_tools()
        logger.info(f"🧠 Tool Capabilities: {len(self.tool_buckets['safe'])} safe, {len(self.tool_buckets['write'])} write.")
        
        # 2. Bind tools if supported
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

        # 3. Initialize Nodes
        self.nodes = AgentNodes(
            config=self.config,
            llm=self.llm,
            tools=self.tools,
            tool_buckets=self.tool_buckets,
            llm_with_tools=self.llm_with_tools
        )

    def _classify_tools(self) -> Dict[str, List[str]]:
        """Separates tools into safe (read-only) and write (action) buckets."""
        buckets = {"safe": [], "write": []}
        for t in self.tools:
            capability = self.tool_registry.get_tool_capability(t)
            if capability == "write":
                buckets["write"].append(t.name)
            else:
                buckets["safe"].append(t.name)
        return buckets

    @property
    def tools(self) -> List[BaseTool]:
        return self.tool_registry.tools

    def build_graph(self):
        if not self.nodes:
            raise RuntimeError("Resources not initialized. Call initialize_resources() first.")

        workflow = StateGraph(AgentState)

        workflow.add_node("summarize", self.nodes.summarize_node)
        workflow.add_node("tool_filter", self.nodes.tool_filter_node)
        workflow.add_node("agent", self.nodes.agent_node)
        workflow.add_node("loop_guard", self.nodes.loop_guard_node)
        workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})
        workflow.add_node("tools", self.nodes.tools_and_validate_node)
        workflow.add_node("reflection", self.nodes.reflection_node)
        workflow.add_node("token_budget_guard", self.nodes.token_budget_guard_node)

        tools_enabled = bool(self.tools) and self.config.check_tool_support()

        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "update_step")
        
        workflow.add_edge("update_step", "token_budget_guard")
        workflow.add_edge("token_budget_guard", "tool_filter")
        workflow.add_edge("tool_filter", "agent")

        def should_continue(state: AgentState):
            steps = state.get("steps", 0)
            messages = state.get("messages", [])

            if steps >= self.config.max_loops:
                logger.debug(f"🛑 Loop Guard: {steps} steps.")
                return "loop_guard"

            if not messages: return "agent"
            last_msg = messages[-1]

            # Redirect system alerts back to agent
            if isinstance(last_msg, SystemMessage):
                content = str(last_msg.content)
                if any(tag in content for tag in ("SYSTEM ALERT", "REFLECTION", "TOKEN BUDGET ALERT")):
                    return "agent"
                return END

            if tools_enabled and isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                return "tools"
            if isinstance(last_msg, ToolMessage): return "agent"
            return END

        destinations = ["tools", "loop_guard", "agent", END] if tools_enabled else ["loop_guard", END]
        workflow.add_conditional_edges("agent", should_continue, destinations)

        if tools_enabled:
            workflow.add_edge("tools", "reflection")
            workflow.add_edge("reflection", "tool_filter")

        workflow.add_edge("loop_guard", END)

        return workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    async def main():
        wf = AgentWorkflow()
        await wf.initialize_resources()
        print(f"✔ Agent Ready. Tools: {len(wf.tools)}")

    asyncio.run(main())
