import json
import asyncio
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field 

# --- LANGCHAIN & LANGGRAPH ---
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    BaseMessage, SystemMessage, RemoveMessage, HumanMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- CORE MODULES ---
from core.config import AgentConfig
from core.state import AgentState
from core.utils import AgentUtils
from tools.tool_registry import ToolRegistry
from core.tool_validator import validate_tool_execution 

# --- OPTIONAL CORE MODULES ---
from dotenv import load_dotenv

try:
    from core.tool_sanitizer import ToolSanitizer
except ImportError:
    class ToolSanitizer:
        @staticmethod
        def sanitize_tool_calls(tc): pass
        
try:
    from core.safety_guard import SafetyGuard
except ImportError:
    class SafetyGuard:
        ENABLED = False
        @classmethod
        def is_unsafe_write(cls, *args): return False

try:
    from core.logging_config import setup_logging
    logger = setup_logging() 
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("agent")

# ==========================================
# ГРАФ АГЕНТА (WORKFLOW)
# ==========================================

class IntentClassification(BaseModel):
    intent: Literal["read_only", "write_action"] = Field(
        description="User's intent: 'read_only' (search, query, view) or 'write_action' (create, edit, delete, patch, fix, modify)."
    )
    reasoning: str = Field(description="Short explanation of why this intent was chosen.")

class AgentWorkflow:
    def __init__(self):
        load_dotenv()
        self.config = AgentConfig()
        self.utils = AgentUtils()
        self.tool_registry = ToolRegistry(self.config)
        
        self.llm: Optional[BaseChatModel] = None
        self.llm_with_tools: Optional[BaseChatModel] = None
        
        # Кэш классифицированных инструментов
        self.tool_buckets = {}

    async def initialize_resources(self):
        logger.info(f"Initializing agent: [bold cyan]{self.config.provider}[/]", extra={"markup": True})
        
        self.llm = self.config.get_llm()
        await self.tool_registry.load_all()
        
        # 1. Классифицируем инструменты (Safe vs Write)
        self.tool_buckets = self._classify_tools()
        logger.info(f"🧠 Tool Capabilities: {len(self.tool_buckets['safe'])} safe, {len(self.tool_buckets['write'])} write.")
        
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

    def _classify_tools(self):
        """
        Детерминированное разделение инструментов.
        Использует ToolRegistry для определения возможностей.
        """
        buckets = {
            "safe": [],
            "write": []
        }
        
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

    # --- NODES ---

    async def _summarize_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")

        if len(messages) <= self.config.summary_threshold:
            return {}

        idx = len(messages) - self.config.summary_keep_last
        if idx < 0: idx = 0

        # Попытка найти HumanMessage в хвосте, чтобы сделать красивый разрез
        scan_idx = idx
        found_human = False
        while scan_idx < len(messages):
            if isinstance(messages[scan_idx], HumanMessage):
                idx = scan_idx
                found_human = True
                break
            scan_idx += 1
        
        # Если HumanMessage не найден в хвосте, используем жесткий idx (keep_last)
        
        to_summarize = messages[:idx]
        if not to_summarize: return {}

        # Формируем текст истории с ограничением длины контента для экономии токенов
        history_parts = []
        for m in to_summarize:
            content = str(m.content)
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            history_parts.append(f"{m.type}: {content}")
        
        history_text = "\n".join(history_parts)
        
        prompt = (
            f"Current memory context:\n<previous_context>\n{summary}\n</previous_context>\n\n"
            f"New events:\n{history_text}\n\n"
            "Update <previous_context>. Keep only key facts, decisions, and results. "
            "Remove chit-chat. Return only the updated context text."
        )
        
        try:
            res = await self.llm.ainvoke(prompt)
            delete_msgs = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
            logger.info(f"🧹 Summary: Removed {len(delete_msgs)} messages.")
            return {"summary": res.content, "messages": delete_msgs}
        except Exception as e:
            logger.error(f"Summarization Error: {e}")
            return {}

    async def _tool_filter_node(self, state: AgentState):
        """
        Smart Tool Filtering (v6.2).
        Combines deterministic checks (Recovery, Active Tool) with LLM-based Intent Classification.
        """
        # 0. Global Bypass via Config
        if not self.config.enable_tool_filtering:
            logger.debug("🔓 Filter: DISABLED by config (All tools allowed)")
            return {"allowed_tools": None}

        messages = state["messages"]
        phase = "exploration"
        last_msg = messages[-1] if messages else None

        # 1. RECOVERY PHASE (Priority #1)
        if isinstance(last_msg, SystemMessage):
            text = str(last_msg.content).upper()
            if "SYSTEM ALERT" in text or "DO NOT RETRY" in text:
                phase = "recovery"
                logger.debug("🛡 Filter: RECOVERY phase (System alert detected)")
                return {"allowed_tools": self.tool_buckets["safe"]}

        # 2. ACTION PHASE (Priority #2)
        # Если мы уже в цикле использования инструментов (предыдущее сообщение - ToolMessage),
        # продолжаем разрешать доступ, чтобы агент мог завершить начатое.
        if phase != "recovery":
            for m in reversed(messages):
                if isinstance(m, ToolMessage):
                    content = str(m.content)
                    if content and not content.startswith("Error"):
                        phase = "action"
                        break
                if isinstance(m, HumanMessage):
                    break # Stop looking back at previous turn

        # 3. LLM INTENT CLASSIFICATION (Priority #3)
        # Если фаза все еще "exploration" и последнее сообщение от человека,
        # используем LLM для определения намерения.
        if phase == "exploration" and isinstance(last_msg, HumanMessage):
            # Fast Path: Check for direct tool mentions first (optimization)
            text = last_msg.content.lower()
            write_tool_names = [name.lower() for name in self.tool_buckets["write"]]
            
            if any(t_name in text for t_name in write_tool_names):
                phase = "intent_action"
                logger.debug("⚡ Filter: Fast Path (Tool mentioned) -> Write Allowed")
            else:
                # LLM Path
                try:
                    classifier = self.llm.with_structured_output(IntentClassification)
                    
                    # SYSTEM PROMPT для классификатора
                    system_prompt = SystemMessage(content=(
                        "Analyze the conversation and determine the user's intent. "
                        "Return a JSON object with 'intent' and 'reasoning'. "
                        "If the user wants to create, edit, save, delete, or modify files/system -> 'write_action'. "
                        "If the user just wants to search, read, or ask questions -> 'read_only'."
                    ))
                    
                    # Берем последние 3 сообщения для контекста
                    context_msgs = [system_prompt] + messages[-3:]
                    
                    classification = await classifier.ainvoke(context_msgs)
                    
                    if classification.intent == "write_action":
                        phase = "intent_action"
                        logger.info(f"🧠 Intent: WRITE detected. Reason: {classification.reasoning}")
                    else:
                        logger.info(f"🧠 Intent: READ detected. Reason: {classification.reasoning}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Intent Classifier Failed: {e}. Falling back to keyword search.")
                    # Fallback to keywords (Safety Net)
                    intent_hints = (
                        "создай", "создать", "запиши", "записать", "сохрани",
                        "сделай", "сделать", "напиши", "написать", "измени",
                        "добавь", "добавить", "обнови", "обновить", "исправь", "почини",
                        "create", "write", "save", "generate", "edit", "update", "delete",
                        "add", "insert", "modify", "fix", "replace", "patch"
                    )
                    if any(hint in text for hint in intent_hints):
                        phase = "intent_action"

        # 4. TOOL GATING
        if phase in ["action", "intent_action"]:
            allowed = None # All tools allowed
            logger.debug(f"🔓 Filter: {phase.upper()} (All tools allowed)")
        else:
            allowed = self.tool_buckets["safe"]
            logger.debug(f"🔒 Filter: {phase.upper()} ({len(allowed)} safe tools allowed)")

        return {"allowed_tools": allowed}
        
    async def _tools_and_validate_node(self, state: AgentState):
        messages = state["messages"]
        last_msg = messages[-1]
        
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {}
            
        final_messages = []
        validation_errors = []
        should_force_retry = False
        tool_retries = state.get("tool_retries", {}).copy()

        for tool_call in last_msg.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call["id"]
            
            # 1. Выполнение инструмента
            tool = next((t for t in self.tools if t.name == t_name), None)
            content = ""
            
            if not tool:
                content = f"Error: Tool '{t_name}' not found."
            else:
                try:
                    raw_result = await tool.ainvoke(t_args)
                    content = str(raw_result)
                except Exception as e:
                    content = f"Error: {str(e)}"

            if not content.strip():
                content = "Error: Tool returned empty response."

            # Создаем сообщение (пока предварительно)
            tool_msg = ToolMessage(content=content, tool_call_id=t_id, name=t_name)

            # 2. Валидация
            result = validate_tool_execution(tool_msg, t_args, t_name)
            
            if not result["is_valid"]:
                logger.debug(f"Tool Error ({t_name}): {result['error_message']}")
                
                # --- [FIX START] Умная обработка ретраев ---
                retry_count = tool_retries.get(t_name, 0)
                
                if result["retry_needed"]:
                    if retry_count < 3:
                        # Разрешаем повтор (мягкая ошибка)
                        tool_retries[t_name] = retry_count + 1
                        should_force_retry = True
                        validation_errors.append(f"- Tool '{t_name}' failed (Attempt {retry_count+1}/3): {result['error_message']}")
                    else:
                        # Лимит исчерпан! ЖЕСТКИЙ БЛОК.
                        # Мы подменяем контент сообщения, чтобы LLM увидела тупик.
                        error_text = f"SYSTEM BLOCK: Too many consecutive errors for '{t_name}'. The tool is failing repeatedly with: {content[:200]}..."
                        tool_msg.content = error_text 
                        validation_errors.append(f"- STOP: {t_name} blocked due to repeated failures.")
                        # Сбрасываем счетчик, так как мы уже наказали агента
                        if t_name in tool_retries: del tool_retries[t_name]
                else:
                    # Если retry_needed=False (например, файла нет), просто сообщаем
                    validation_errors.append(f"- Tool '{t_name}' returned: {result['error_message']}")
                
            else:
                # Успех - сбрасываем счетчик ошибок для этого инструмента
                if t_name in tool_retries: del tool_retries[t_name]

            final_messages.append(tool_msg)

        # 3. Формирование системного совета
        if validation_errors:
            if should_force_retry:
                advice = "INSTRUCTION: Arguments invalid or tool failed. Review the error and TRY AGAIN with corrected parameters."
            else:
                # Если мы здесь, значит либо ретраи кончились, либо ошибка фатальна
                advice = (
                    "INSTRUCTION: Action failed repeatedly or is impossible. "
                    "DO NOT RETRY the same tool with the same arguments. "
                    "Stop and analyze the error. Try a different approach (e.g., check file existence first)."
                )
            
            err_text = "\n".join(validation_errors)
            final_messages.append(SystemMessage(content=f"SYSTEM ALERT:\n{err_text}\n{advice}"))

        return {
            "messages": final_messages,
            "tool_retries": tool_retries
        }
        
    async def _agent_node(self, state: AgentState):
        messages = state["messages"]
        
        # --- DYNAMIC BINDING (Фильтрация инструментов) ---
        allowed = state.get("allowed_tools")
        if allowed is not None:
            # Фильтруем список
            selected_tools = [t for t in self.tools if t.name in allowed]
            current_llm = self.llm.bind_tools(selected_tools)
        else:
            # Полный доступ
            current_llm = self.llm_with_tools
        # -------------------------------------------------

        tools_available = (current_llm != self.llm)
        
        sys_msg = self._build_system_message(state.get("summary", ""), tools_available)
        full_context = [sys_msg] + messages
        
        response = await self._invoke_llm_with_retry(current_llm, full_context)
        
        last_tool_call = None
        
        # SafetyGuard (дополнительный слой)
        if SafetyGuard.is_unsafe_write(response, full_context):
            response = SystemMessage(
                content="STOP. You are trying to write a file without valid data from search/fetch. "
                        "Perform a search first to get actual content."
            )
            
        elif isinstance(response, AIMessage) and response.tool_calls:
            ToolSanitizer.sanitize_tool_calls(response.tool_calls)
            last_tool_call = response.tool_calls[0]
            
            for tc in response.tool_calls:
                if tc['name'] in ['write_file', 'save_file']:
                    raw_path = str(tc['args'].get('path', '')).strip()
                    if len(raw_path) < 2 or re.match(r'^[\.,\-_:;\'" ]+$', raw_path):
                        logger.debug(f"🛡️ Quality Gate: Rejecting garbage filename '{raw_path}'")
                        response = SystemMessage(
                            content=f"SYSTEM ERROR: The filename '{raw_path}' is invalid. "
                                    "Please RETRY with a meaningful filename."
                        )
                        last_tool_call = None
                        break 
                        
        if isinstance(response, AIMessage):
            self._patch_token_usage(response, full_context)
        
        if isinstance(response, AIMessage) and response.tool_calls:
            last_msg = messages[-1] if messages else None
            if isinstance(last_msg, ToolMessage):
                current_tool = response.tool_calls[0]['name']
                last_tool = last_msg.name
                if current_tool == "write_file" and last_tool == "write_file":
                    logger.warning("🛑 Loop Guard: Blocked repetitive write_file.")
                    response = AIMessage(
                        content="System: File already written. Stop overwriting."
                    )

        return {
            "messages": [response],
            "last_tool_call": last_tool_call
        }
        
    async def _loop_guard_node(self, state: AgentState):
        return {"messages": [AIMessage(content="🛑 **Auto-Stop**: Max steps limit reached.")]}

    # --- HELPERS ---

    def _build_system_message(self, summary: str, tools_available: bool = True) -> SystemMessage:
        if self.config.prompt_path.exists():
            raw_prompt = self.config.prompt_path.read_text("utf-8")
            logger.info(f"✅ System prompt loaded from: {self.config.prompt_path} ({len(raw_prompt)} chars)")
        else:
            logger.warning(f"⚠️ System prompt not found at: {self.config.prompt_path}. Using fallback.")
            raw_prompt = (
                "You are an autonomous AI agent.\n"
                "Reason in English, Reply in Russian.\n"
                "Date: {{current_date}}\nCWD: {{cwd}}"
            )
        
        prompt = raw_prompt.replace("{{current_date}}", datetime.now().strftime("%Y-%m-%d"))
        prompt = prompt.replace("{{cwd}}", str(Path.cwd()))
        
        if not tools_available:
            prompt += "\nNOTE: You are in CHAT-ONLY mode. Tools are disabled for this session."
        elif self.config.use_long_term_memory:
             prompt += "\nUse memory tools (recall_facts/remember_fact) when necessary."
             
        if summary:
            prompt += f"\n\n<memory>\n{summary}\n</memory>"
            
        return SystemMessage(content=prompt)

    async def _invoke_llm_with_retry(self, llm, context: List[BaseMessage]) -> AIMessage:
        FATAL_ERRORS = ["401", "unauthorized", "quota", "billing", "context_length_exceeded"]

        for attempt in range(3):
            try:
                response = await llm.ainvoke(context)
                if not response.content and not response.tool_calls:
                    raise ValueError("Empty response from LLM")
                return response

            except Exception as e:
                error_str = str(e).lower()
                if any(err in error_str for err in FATAL_ERRORS):
                    logger.error(f"🛑 Fatal LLM Error: {e}")
                    return AIMessage(content=f"System Error: API refused request ({e})")

                if attempt < 2:
                    logger.debug(f"⚠️ LLM Crash (Attempt {attempt+1}): {e}. Retrying...")
                    await asyncio.sleep(1)
                    continue
                
                logger.error(f"💀 All retries failed: {e}")
            
        return AIMessage(content=f"**System Failure**: Multiple API crashes.")
      
    def _patch_token_usage(self, response: AIMessage, context: List[BaseMessage]):
        usage = response.usage_metadata or {}
        if (isinstance(usage, dict) and usage.get("input_tokens", 0) > 0):
            return

        meta = response.response_metadata or {}
        add_kwargs = response.additional_kwargs or {}

        candidates = [
            meta.get("usage"),
            meta.get("token_usage"),
            add_kwargs.get("usage"),
            add_kwargs.get("token_usage"),
            meta.get("body", {}).get("usage") if isinstance(meta.get("body"), dict) else None,
        ]

        raw_usage = None
        for c in candidates:
            if isinstance(c, dict) and ("prompt_tokens" in c or "completion_tokens" in c):
                raw_usage = c
                break

        if raw_usage:
            input_tokens = raw_usage.get("prompt_tokens", 0)
            output_tokens = raw_usage.get("completion_tokens", 0)
            total_tokens = raw_usage.get("total_tokens", input_tokens + output_tokens)

            if input_tokens > 0 or output_tokens > 0:
                response.usage_metadata = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "token_source": "Provider"
                }
                return

        input_tokens = self.utils.estimate_payload_tokens(context, self.tools)
        output_content = response.content
        if isinstance(output_content, list):
            output_content = " ".join(str(x) for x in output_content)

        output_tokens = self.utils.count_tokens(str(output_content))
        if response.tool_calls:
            output_tokens += self.utils.count_tokens(json.dumps(response.tool_calls, default=str))

        response.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "token_source": "Manual"
        }
    
    # --- GRAPH BUILDER ---

    def build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("summarize", self._summarize_node)
        
        # Добавляем наш детерминированный фильтр
        workflow.add_node("tool_filter", self._tool_filter_node)
        
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("loop_guard", self._loop_guard_node)
        workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})
        workflow.add_node("tools", self._tools_and_validate_node)
        
        tools_enabled = bool(self.tools) and self.config.check_tool_support()

        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "update_step")
        
        # Жесткий маршрут: update -> filter -> agent
        workflow.add_edge("update_step", "tool_filter")
        workflow.add_edge("tool_filter", "agent")

        def should_continue(state: AgentState):
            steps = state.get("steps", 0)
            messages = state.get("messages", [])

            if steps >= self.config.max_loops:
                logger.debug(f"🛑 Loop Guard: {steps} steps.")
                return "loop_guard"

            if not messages: return "agent"
            last_msg = messages[-1]

            if isinstance(last_msg, SystemMessage): return "agent"
            if tools_enabled and isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                return "tools"
            if isinstance(last_msg, ToolMessage): return "agent"
            return END

        destinations = ["tools", "loop_guard", "agent", END] if tools_enabled else ["loop_guard", END]
        workflow.add_conditional_edges("agent", should_continue, destinations)

        if tools_enabled:
            # После инструментов возвращаемся в фильтр, 
            # чтобы он увидел успешный ToolMessage и открыл доступ к write
            workflow.add_edge("tools", "tool_filter")

        workflow.add_edge("loop_guard", END)

        return workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    async def main():
        wf = AgentWorkflow()
        await wf.initialize_resources()
        print(f"✅ Agent Ready. Tools: {len(wf.tools)}")

    asyncio.run(main())
