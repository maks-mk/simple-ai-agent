import json
import logging
import asyncio
from typing import List, Optional, Dict, Any

from langchain_core.messages import (
    BaseMessage, SystemMessage, RemoveMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from core.state import AgentState
from core.config import AgentConfig
from core import constants

logger = logging.getLogger("agent")

class AgentNodes:
    def __init__(self, config: AgentConfig, llm: BaseChatModel, tools: List[BaseTool], llm_with_tools: Optional[BaseChatModel] = None):
        self.config = config
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = llm_with_tools or llm

    # --- NODE: SUMMARIZE ---
    
    async def summarize_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")

        if len(messages) <= self.config.summary_threshold:
            return {}

        # Determine cut-off point
        idx = len(messages) - self.config.summary_keep_last
        if idx < 0: idx = 0

        # Try to find a clean break at a HumanMessage
        scan_idx = idx
        while scan_idx < len(messages):
            if isinstance(messages[scan_idx], HumanMessage):
                idx = scan_idx
                break
            scan_idx += 1
        
        to_summarize = messages[:idx]
        if not to_summarize: return {}

        # Format history for LLM
        history_text = self._format_history_for_summary(to_summarize)
        
        prompt = constants.SUMMARY_PROMPT_TEMPLATE.format(
            summary=summary,
            history_text=history_text
        )
        
        try:
            res = await self.llm.ainvoke(prompt)
            # Ensure metadata exists for summary too
            self._ensure_usage_metadata(res, [HumanMessage(content=prompt)])
            
            delete_msgs = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
            logger.info(f"🧹 Summary: Removed {len(delete_msgs)} messages.")
            return {"summary": res.content, "messages": delete_msgs}
        except Exception as e:
            logger.error(f"Summarization Error: {e}")
            return {}

    def _format_history_for_summary(self, messages: List[BaseMessage]) -> str:
        parts = []
        for m in messages:
            content = str(m.content)
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            parts.append(f"{m.type}: {content}")
        return "\n".join(parts)

    # --- NODE: AGENT ---

    async def agent_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")
        
        # Build System Prompt
        sys_msg = self._build_system_message(summary, tools_available=bool(self.tools))
        
        # Prepare context
        full_context = [sys_msg] + messages
        
        # Invoke LLM
        response = await self._invoke_llm_with_retry(self.llm_with_tools, full_context)
        
        # Ensure metadata is present (inject estimate if missing)
        self._ensure_usage_metadata(response, full_context)

        # Token Updates
        token_usage_update = {}
        if isinstance(response, AIMessage):
            # No need to call _update_token_usage separately if we rely on metadata now
            # But let's keep it consistent with existing state logic
            if response.usage_metadata:
                 token_usage_update = {"token_usage": response.usage_metadata}

        return {
            "messages": [response],
            **token_usage_update
        }

    # --- NODE: TOOLS ---

    async def tools_node(self, state: AgentState):
        messages = state["messages"]
        last_msg = messages[-1]
        
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {}
            
        final_messages = []
        has_error = False

        for tool_call in last_msg.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call["id"]
            
            # Execute
            content = await self._execute_tool(t_name, t_args)
            
            # Check for error signature
            if content.startswith("Error:") or content.startswith("Ошибка:"):
                has_error = True

            # Truncate output to avoid context window explosion
            # 4000 chars ~ 1000 tokens. This is a safe limit for tool outputs.
            MAX_TOOL_OUTPUT = 4000
            if len(content) > MAX_TOOL_OUTPUT:
                content = content[:MAX_TOOL_OUTPUT] + f"\n... [Output truncated. Total length: {len(content)} chars]"

            tool_msg = ToolMessage(content=content, tool_call_id=t_id, name=t_name)
            
            final_messages.append(tool_msg)

        if has_error:
            # Inject reflection prompt to guide the agent
            reflection_msg = HumanMessage(content=constants.REFLECTION_PROMPT)
            final_messages.append(reflection_msg)

        return {
            "messages": final_messages,
        }

    async def _execute_tool(self, name: str, args: dict) -> str:
        tool = next((t for t in self.tools if t.name == name), None)
        if not tool:
            return f"Error: Tool '{name}' not found."
        try:
            raw_result = await tool.ainvoke(args)
            content = str(raw_result)
            if not content.strip():
                return "Error: Tool returned empty response."
            return content
        except Exception as e:
            return f"Error: {str(e)}"

    # --- HELPERS ---

    def _build_system_message(self, summary: str, tools_available: bool = True) -> SystemMessage:
        # 1. Try configured path
        if self.config.prompt_path.exists():
            prompt_file = self.config.prompt_path
        elif (constants.BASE_DIR / "prompt.txt").exists():
            prompt_file = constants.BASE_DIR / "prompt.txt"
        else:
            prompt_file = None

        if prompt_file:
            raw_prompt = prompt_file.read_text("utf-8")
        else:
            raw_prompt = (
                "You are an autonomous AI agent.\n"
                "Reason in English, Reply in Russian.\n"
                "Date: {{current_date}}"
            )
        
        from datetime import datetime
        from pathlib import Path
        
        prompt = raw_prompt.replace("{{current_date}}", datetime.now().strftime("%Y-%m-%d"))
        prompt = prompt.replace("{{cwd}}", str(Path.cwd()))
        
        if not tools_available:
            prompt += "\nNOTE: You are in CHAT-ONLY mode. Tools are disabled."
             
        if summary:
            prompt += f"\n\n<memory>\n{summary}\n</memory>"
            
        return SystemMessage(content=prompt)

    async def _invoke_llm_with_retry(self, llm, context: List[BaseMessage]) -> AIMessage:
        for attempt in range(3):
            try:
                response = await llm.ainvoke(context)
                if not response.content and not response.tool_calls:
                    raise ValueError("Empty response from LLM")
                return response
            except Exception as e:
                logger.warning(f"LLM Error (Attempt {attempt+1}): {e}")
                if attempt == 2:
                    return AIMessage(content=f"System Error: API request failed after 3 retries. ({e})")
                await asyncio.sleep(1)
        return AIMessage(content="System Error: Unknown failure.")

    def _ensure_usage_metadata(self, response: AIMessage, context: List[BaseMessage]):
        """Injects estimated usage metadata if missing."""
        if not response.usage_metadata:
            # Estimate Input
            # Rough estimate: 4 chars per token
            in_chars = sum(len(str(m.content)) for m in context)
            in_tokens = in_chars // 4
            
            # Estimate Output
            out_chars = len(str(response.content))
            out_tokens = out_chars // 4
            
            response.usage_metadata = {
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
                "token_source": "Est"
            }
