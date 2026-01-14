import json
import asyncio
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Literal, TypedDict, Annotated, Optional, Any

# --- LANGCHAIN & LANGGRAPH ---
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool, BaseTool
# ДОБАВЛЕН ToolMessage В ИМПОРТЫ
from langchain_core.messages import BaseMessage, SystemMessage, RemoveMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# --- PROVIDERS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# --- CONFIG & UTILS ---
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# --- LOCAL MODULES ---
try:
    from logging_config import setup_logging
    logger = setup_logging() 
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("agent")

# Инструменты
try:
    from delete_tools import SafeDeleteFileTool, SafeDeleteDirectoryTool
except ImportError:
    SafeDeleteFileTool = SafeDeleteDirectoryTool = None
    
try:
    from search_tools import web_search, fetch_url
except ImportError:
    web_search = fetch_url = None
    logger.warning("Search tools not found or dependencies missing (httpx, bs4).")

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None
    
# Токенизатор
try:
    import tiktoken
except ImportError:
    tiktoken = None


# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================

class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    provider: Literal["gemini", "openai"] = "gemini"
    
    # Keys
    gemini_api_key: Optional[SecretStr] = None
    gemini_model: str = "gemini-1.5-flash"
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4o"
    openai_base_url: Optional[str] = None

    temperature: float = 0.2
    
    # Logic
    use_long_term_memory: bool = Field(default=False, alias="LONG_TERM_MEMORY")
    max_loops: int = Field(default=15, description="Максимальное количество шагов агента за один запрос")
    
    # Summarization
    summary_threshold: int = Field(default=20, alias="SESSION_SIZE")
    summary_keep_last: int = Field(default=4, alias="SUMMARY_KEEP_LAST")
    
    # Paths
    prompt_path: Path = Field(default=Path("prompt.txt"), alias="PROMPT_PATH")
    mcp_config_path: Path = Path("mcp.json")
    memory_db_path: str = "./memory_db"

    @model_validator(mode='after')
    def check_api_keys(self) -> 'AgentConfig':
        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for gemini provider.")
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for openai provider.")
        return self

    def get_llm(self) -> BaseChatModel:
        if self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.gemini_model,
                temperature=self.temperature,
                google_api_key=self.gemini_api_key.get_secret_value(),
                convert_system_message_to_human=True
            )
        elif self.provider == "openai":
            return ChatOpenAI(
                model=self.openai_model,
                temperature=self.temperature,
                api_key=self.openai_api_key.get_secret_value(),
                base_url=self.openai_base_url,
                model_kwargs={"stream_options": {"include_usage": True}}
            )
        raise ValueError(f"Unknown provider: {self.provider}")


# ==========================================
# 2. СОСТОЯНИЕ ГРАФА
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    steps: int


# ==========================================
# 3. WORKFLOW
# ==========================================

class AgentWorkflow:
    def __init__(self):
        load_dotenv()
        self.config = AgentConfig()
        self.tools: List[BaseTool] = []
        self.llm: Optional[BaseChatModel] = None
        self.llm_with_tools: Optional[BaseChatModel] = None
        self._encoder = None

    async def initialize_resources(self):
        logger.info(f"Initializing agent: [bold cyan]{self.config.provider}[/]", extra={"markup": True})
        self.llm = self.config.get_llm()

        if SafeDeleteFileTool and SafeDeleteDirectoryTool:
            cwd = Path.cwd()
            self.tools.extend([
                SafeDeleteFileTool(root_dir=cwd),
                SafeDeleteDirectoryTool(root_dir=cwd)
            ])
            
        # --- ДОБАВЛЯЕМ ИНСТРУМЕНТЫ ПОИСКА ---
        if web_search and fetch_url:
            self.tools.extend([web_search, fetch_url])
            logger.info("✅ Search tools loaded.")

        if self.config.use_long_term_memory:
            self._init_memory_tools()

        if MultiServerMCPClient and self.config.mcp_config_path.exists():
            await self._init_mcp_tools()

        # Привязка инструментов к LLM
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm
        
        # Инициализация энкодера для подсчета токенов
        if tiktoken:
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass

    def _init_memory_tools(self):
        try:
            from memory_manager import MemoryManager
            memory = MemoryManager(db_path=self.config.memory_db_path)
            
            @tool
            async def remember_fact(text: str, category: str = "general") -> str:
                """Saves an important fact about the user, project, or preferences."""
                return await memory.aremember(text, {"type": category})
            
            @tool
            async def recall_facts(query: str) -> str:
                """Searches for information in long-term memory."""
                facts = await memory.arecall(query)
                return "\n".join(f"- {f}" for f in facts) if facts else "No facts found."
            
            @tool
            async def forget_fact(query: str) -> str:
                """Removes facts from memory."""
                return f"Forgotten: {await memory.adelete_fact_by_query(query)}"

            self.tools.extend([remember_fact, recall_facts, forget_fact])
        except ImportError:
            logger.warning("MemoryManager not found.")

    async def _init_mcp_tools(self):
        if not self.config.mcp_config_path.exists(): return
        try:
            raw_cfg = json.loads(self.config.mcp_config_path.read_text("utf-8"))
            mcp_cfg = {
                name: {
                    **{k: v for k, v in cfg.items() if k != 'enabled'},
                    "args": [a.replace("{filesystem_path}", str(Path.cwd())) for a in cfg.get("args", [])]
                }
                for name, cfg in raw_cfg.items() if cfg.get("enabled", True)
            }
            if mcp_cfg:
                client = MultiServerMCPClient(mcp_cfg)
                new_tools = await asyncio.wait_for(client.get_tools(), timeout=120)
                self.tools.extend(new_tools)
                logger.info(f"Loaded MCP tools: {list(mcp_cfg.keys())}")
        except Exception as e:
            logger.error(f"MCP Error: {e}")

    def _get_base_prompt(self) -> str:
        if self.config.prompt_path.exists():
            raw_prompt = self.config.prompt_path.read_text("utf-8")
        else:
            raw_prompt = (
                "You are an autonomous AI agent with access to tools.\n"
                "Always fulfill the user's request.\n"
                "Your internal reasoning and tool usage must be in English.\n"
                "HOWEVER, your final response to the user must be in Russian.\n"
                "Current date: {{current_date}}\n"
                "CWD: {{cwd}}"
            )

        prompt = raw_prompt.replace("{{current_date}}", datetime.now().strftime("%Y-%m-%d"))
        prompt = prompt.replace("{{cwd}}", str(Path.cwd()))
        
        if self.config.use_long_term_memory:
             prompt += "\nUse memory tools (recall_facts/remember_fact) when necessary."
        
        return prompt

    def _count_tokens(self, text: str) -> int:
        if not text: return 0
        if self._encoder:
            return len(self._encoder.encode(text))
        return len(text) // 3  # Fallback heuristic

    def _estimate_payload_tokens(self, messages: List[BaseMessage], tools: List[BaseTool]) -> int:
        """
        Считает токены для полного контекста: сообщения + схемы инструментов.
        """
        total = 0
        # 1. Считаем сообщения
        for m in messages:
            content = m.content if isinstance(m.content, str) else ""
            if isinstance(m.content, list):
                content = " ".join([str(x) for x in m.content])
            total += self._count_tokens(content)
        
        # 2. Считаем инструменты (JSON schema)
        if tools:
            try:
                tool_schemas = [convert_to_openai_tool(t) for t in tools]
                tools_json = json.dumps(tool_schemas, ensure_ascii=False)
                total += self._count_tokens(tools_json)
            except Exception:
                simple_desc = "\n".join([f"{t.name}: {t.description}" for t in tools])
                total += self._count_tokens(simple_desc)
        
        return total

    # ==========================================
    # 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (SANITIZER)
    # ==========================================
    
    def _sanitize_path(self, path: str) -> str:
        """
        Жестко чистит путь от двоеточий, кавычек и языковых префиксов.
        Пример: ":ru:file.txt" -> "file.txt"
        """
        original = path
        path = re.sub(r'^:[a-z]{2,3}:', '', path) # :ru:
        path = re.sub(r'[:"<>|?*]+', '', path)    # Win chars
        path = path.strip()
        
        if path != original:
            # logger.warning(f"🛡️ Path sanitized: '{original}' -> '{path}'")
            # Можно использовать debug, чтобы не шуметь в консоль
            logger.debug(f"🛡️ Path sanitized: '{original}' -> '{path}'")
        
        return path

    def _fix_tool_calls(self, tool_calls: List[dict]):
        """
        Универсальный санитайзер аргументов инструментов.
        1. Чистит пути к файлам (File Tools).
        2. Чистит и валидирует URL (Web Tools), предотвращая loop с '}'.
        """
        path_keys = {"path", "file_path", "dir_path", "destination", "source", "filename"}
        url_keys = {"url", "link", "target_url", "query"} 

        for tc in tool_calls:
            args = tc.get("args")
            name = tc.get("name")

            # Внутренняя функция очистки значения
            def clean_val(k, v):
                if not isinstance(v, str): return v
                
                # A. Логика для путей
                if k in path_keys:
                    return self._sanitize_path(v)
                
                # B. Логика для URL (защита от мусора)
                if name == "fetch_url" and (k in url_keys or k == "url"):
                    # Удаляем кавычки, скобки, фигурные скобки И ДВОЕТОЧИЯ
                    clean = v.strip().strip("'").strip('"').strip("{}").strip(":")
                    
                    # Если осталась пустая строка или мусор, но внутри есть http - спасаем
                    if "http" in v and not clean.startswith("http"):
                        match = re.search(r'(https?://[^\s\'"<>{}]+)', v)
                        if match:
                            clean = match.group(1)
                            logger.debug(f"🛡️ URL extracted: '{v}' -> '{clean}'")
                    
                    return clean
                
                return v

            # 1. Именованные аргументы (Dict)
            if isinstance(args, dict):
                for key, value in args.items():
                    args[key] = clean_val(key, value)

            # 2. Позиционные аргументы (List)
            elif isinstance(args, list) and len(args) > 0:
                if isinstance(args[0], str):
                    # Эвристика: если fetch_url, то первый аргумент - url, иначе - путь
                    fake_key = "url" if name == "fetch_url" else "path"
                    args[0] = clean_val(fake_key, args[0])

    # ==========================================
    # 5. УЗЛЫ ГРАФА
    # ==========================================

    async def _summarize_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")

        if len(messages) <= self.config.summary_threshold:
            return {}

        keep_last = self.config.summary_keep_last
        idx = len(messages) - keep_last
        while idx < len(messages):
            if isinstance(messages[idx], HumanMessage):
                break
            idx += 1
        
        to_summarize = messages[:idx]
        if not to_summarize:
            return {}

        history_text = "\n".join([f"{m.type}: {m.content}" for m in to_summarize])

        prompt = (
            f"Current memory context:\n<previous_context>\n{summary}\n</previous_context>\n\n"
            f"New events:\n{history_text}\n\n"
            "Update <previous_context>. Keep only key facts, decisions, and results. "
            "Remove chit-chat. Return only the updated context text."
        )
        
        try:
            res = await self.llm.ainvoke(prompt)
            delete_msgs = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
            logger.info(f"🧹 Сжатие контекста: удалено {len(delete_msgs)} сообщений.")
            return {"summary": res.content, "messages": delete_msgs}
        except Exception as e:
            logger.error(f"Summarization Error: {e}")
            return {}

    async def _agent_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")
        
        # 1. Формируем системный промпт и контекст
        sys_text = self._get_base_prompt()
        if summary:
            sys_text += f"\n\n<previous_context role='memory' priority='low'>\n{summary}\n</previous_context>"
        
        sys_msg = SystemMessage(content=sys_text)
        history = [m for m in messages if not isinstance(m, SystemMessage)]
        full_context = [sys_msg] + history
        
        # 2. [FALLBACK] Предварительно считаем Input токены
        estimated_input = self._estimate_payload_tokens(full_context, self.tools)
        response = None
        
        # 3. RETRY LOGIC: Цикл попыток вызова LLM
        for attempt in range(3):
            try:
                response = await self.llm_with_tools.ainvoke(full_context)
                
                # Если ответ пустой
                if not response.content and not response.tool_calls:
                    raise ValueError("Empty response received from LLM")
                
                break # Успех
                
            except Exception as e:
                # --- SMART FIX ДЛЯ write_file ---
                # Если ошибка возникла, но последнее сообщение в истории - это успешная запись файла,
                # значит модель просто "поленилась" ответить. Генерируем ответ за неё.
                last_msg = full_context[-1] if full_context else None
                is_write_success = (
                    isinstance(last_msg, ToolMessage) 
                    and "Successfully wrote" in str(last_msg.content)
                )
                
                if is_write_success:
                    logger.info("🛡️ Auto-completing after write_file crash.")
                    response = AIMessage(
                        content="Файл успешно записан. Задача выполнена. (Авто-завершение)"
                    )
                    break
                # --------------------------------
                
                logger.warning(f"⚠️ LLM Error (Attempt {attempt+1}/3): {e}")
                if attempt == 2:
                    response = AIMessage(
                        content=f"System Error: The model produced invalid output after 3 attempts. Error: {e}"
                    )
                else:
                    await asyncio.sleep(1)
                    
        # 4. --- ИНТЕРСЕПТОР: ЧИСТКА АРГУМЕНТОВ ---
        if response.tool_calls:
            self._fix_tool_calls(response.tool_calls)

        # 5. --- QUALITY GATE: Защита от записи пустых файлов ---
        if response.tool_calls and any(tc['name'] == 'write_file' for tc in response.tool_calls):
            has_valid_data = False
            for m in history:
                # Используем ToolMessage из импортов!
                if isinstance(m, ToolMessage) and m.name in ["fetch_url", "web_search"]:
                    content_str = str(m.content)
                    if "Error" not in content_str and "Ошибка" not in content_str and len(content_str) > 100:
                        has_valid_data = True
                        break
            
            if not has_valid_data:
                logger.warning("🛡️ Quality Gate: Blocked write_file due to lack of valid sources.")
                response = AIMessage(
                    content="STOP. You are trying to write a file, but ALL your previous search/fetch attempts failed or returned errors. "
                            "You have NO valid data to write. You MUST try searching again with different keywords or fetch different URLs first."
                )

        # 6. [FALLBACK] Патч метаданных токенов
        usage = response.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        
        if input_tokens == 0:
            output_content = response.content
            if isinstance(output_content, list):
                output_content = " ".join([str(x) for x in output_content])
            
            estimated_output = self._count_tokens(str(output_content))
            
            if response.tool_calls:
                tools_str = json.dumps([tc for tc in response.tool_calls], default=str)
                estimated_output += self._count_tokens(tools_str)
            
            new_meta = {
                "input_tokens": estimated_input,
                "output_tokens": estimated_output,
                "total_tokens": estimated_input + estimated_output
            }
            
            try:
                response.usage_metadata = new_meta
            except Exception:
                response = AIMessage(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    id=response.id,
                    response_metadata=response.response_metadata,
                    usage_metadata=new_meta
                )

        return {"messages": [response]}
        
    async def _loop_guard_node(self, state: AgentState):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "🛑 **Автоматическая остановка**\n\n"
                        "Агент превысил лимит шагов."
                    )
                )
            ]
        }

    def build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("loop_guard", self._loop_guard_node)
        workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})
        
        if self.tools:
            workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "update_step") 
        workflow.add_edge("update_step", "agent")

        def should_continue(state):
            steps = state.get("steps", 0)
            if steps >= self.config.max_loops:
                logger.warning(f"🛑 Loop Guard triggered: {steps} steps.")
                return "loop_guard" 

            last_msg = state["messages"][-1]
            return "tools" if getattr(last_msg, "tool_calls", None) else END

        destinations = ["tools", "loop_guard", END] if self.tools else ["loop_guard", END]
        workflow.add_conditional_edges("agent", should_continue, destinations)

        if self.tools:
            workflow.add_edge("tools", "update_step")

        workflow.add_edge("loop_guard", END)

        return workflow.compile(checkpointer=MemorySaver())
        
if __name__ == "__main__":
    async def main():
        wf = AgentWorkflow()
        await wf.initialize_resources()
        wf.build_graph()
        print(f"✅ Agent Ready. Tools: {len(wf.tools)}")

    asyncio.run(main())