import json
import asyncio
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Literal, TypedDict, Annotated, Optional, Any, Union

# --- LANGCHAIN & LANGGRAPH ---
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import (
    BaseMessage, SystemMessage, RemoveMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# --- PROVIDERS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# --- CONFIG ---
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

# --- OPTIONAL DEPENDENCIES ---
try:
    import tiktoken
except ImportError:
    tiktoken = None

# ==========================================
# 1. КОНФИГУРАЦИЯ (CONFIG)
# ==========================================

class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    provider: Literal["gemini", "openai"] = "gemini"
    
    # API Keys & Models
    gemini_api_key: Optional[SecretStr] = None
    gemini_model: str = "gemini-1.5-flash"
    
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4o"
    openai_base_url: Optional[str] = None

    temperature: float = 0.2
    
    # Настройка Deep Search (true/false)
    enable_deep_search: bool = Field(default=False, alias="DEEP_SEARCH")
    
    # Ручное управление поддержкой инструментов (по умолчанию True)
    model_supports_tools: bool = Field(default=True, alias="MODEL_SUPPORTS_TOOLS")

    # Logic Settings
    use_long_term_memory: bool = Field(default=False, alias="LONG_TERM_MEMORY")
    max_loops: int = Field(default=15, description="Limit steps per request")
    
    # Summarization Settings
    summary_threshold: int = Field(default=20, alias="SESSION_SIZE")
    summary_keep_last: int = Field(default=4, alias="SUMMARY_KEEP_LAST")
    
    # Paths
    prompt_path: Path = Field(default=Path("prompt.txt"), alias="PROMPT_PATH")
    mcp_config_path: Path = Path("mcp.json")
    memory_db_path: str = "./memory_db"

    @model_validator(mode='after')
    def validate_provider_keys(self) -> 'AgentConfig':
        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for gemini provider.")
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for openai provider.")
        return self

    def check_tool_support(self) -> bool:
        """
        Определяет, поддерживает ли текущая модель вызов инструментов.
        1. Смотрит на явный флаг MODEL_SUPPORTS_TOOLS в .env
        2. Если флаг True, проверяет черный список моделей (эвристика).
        """
        if not self.model_supports_tools:
            return False
            
        # Эвристика для OpenRouter и других провайдеров
        if self.provider == "openai":
            model_name = self.openai_model.lower()
            # Список моделей, которые точно не умеют в Tools или работают плохо
            no_tool_prefixes = (
                "tngtech/", 
                "huggingface/",
                "grey-wing/",
                "sao10k/" 
            )
            if model_name.startswith(no_tool_prefixes):
                return False
                
        return True

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
                model_kwargs={
                    "stream_options": {"include_usage": True},
                    "frequency_penalty": 0.6,
                    "presence_penalty": 0.3,
                }            
            )
        raise ValueError(f"Unknown provider: {self.provider}")


# ==========================================
# 2. УТИЛИТЫ (UTILS)
# ==========================================

class AgentUtils:
    """Вспомогательные функции для токенов, путей и санитайзинга."""
    
    def __init__(self):
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base") if tiktoken else None
        except Exception:
            self._encoder = None

    def count_tokens(self, text: str) -> int:
        if not text: return 0
        if self._encoder:
            return len(self._encoder.encode(text))
        return len(text) // 3

    def estimate_payload_tokens(self, messages: List[BaseMessage], tools: List[BaseTool]) -> int:
        """Считает токены сообщений + схемы инструментов."""
        total = 0
        for m in messages:
            content = m.content if isinstance(m.content, str) else ""
            if isinstance(m.content, list):
                content = " ".join([str(x) for x in m.content])
            total += self.count_tokens(content)
        
        if tools:
            try:
                tool_schemas = [convert_to_openai_tool(t) for t in tools]
                tools_json = json.dumps(tool_schemas, ensure_ascii=False)
                total += self.count_tokens(tools_json)
            except Exception:
                pass
        return total

    @staticmethod
    def sanitize_path(path: str) -> str:
        """Чистит путь от мусора (:ru:, win-chars)."""
        path = re.sub(r'^:[a-z]{2,3}:', '', path)
        path = re.sub(r'[<>|?*]+', '', path)
        return path.strip()

    @staticmethod
    def fix_tool_calls(tool_calls: List[dict]):
        """Чистит аргументы (пути, url) в вызовах инструментов."""
        import time 
        from pathlib import Path

        path_keys = {"path", "file_path", "dir_path", "destination", "source", "filename"}
        url_keys = {"url", "link", "target_url", "query", "urls"} 

        for tc in tool_calls:
            args = tc.get("args")
            name = tc.get("name")
            
            def clean_val(k, v):
                if not isinstance(v, str): return v
                
                # --- ЛОГИКА ОБРАБОТКИ ПУТЕЙ ---
                if k in path_keys:
                    if not v or v.strip() == ".":
                        return f"doc_{int(time.time())}.txt"

                    path_obj = Path(v)
                    # Если путь абсолютный - ДОВЕРЯЕМ ЕМУ (MCP/Tools сами проверят безопасность)
                    if path_obj.is_absolute():
                        return str(path_obj)
                    
                    # Если путь относительный - чистим от ".." (Path Traversal)
                    clean_parts = [
                        AgentUtils.sanitize_path(p) 
                        for p in path_obj.parts 
                        if p not in [".", "..", "\\", "/"]
                    ]
                    
                    if not clean_parts:
                         return f"doc_{int(time.time())}.txt"
                    
                    return str(Path(*clean_parts))

                # --- ОБРАБОТКА URL ---
                if name == "fetch_content" and (k in url_keys or k == "url" or k == "urls"):
                    if isinstance(v, list):
                         return [clean_val("url", item) for item in v]
                    
                    clean = v.strip().strip("'").strip('"').strip("{}").strip(":")
                    if "http" in v and not clean.startswith("http"):
                        match = re.search(r'(https?://[^\s\'"<>{}]+)', v)
                        if match: clean = match.group(1)
                    return clean
                return v

            if isinstance(args, dict):
                for key, value in args.items():
                    args[key] = clean_val(key, value)
            elif isinstance(args, list) and len(args) > 0:
                if isinstance(args[0], (str, list)):
                    fake_key = "urls" if name == "fetch_content" else "path"
                    args[0] = clean_val(fake_key, args[0])


# ==========================================
# 3. РЕЕСТР ИНСТРУМЕНТОВ (TOOLS REGISTRY)
# ==========================================

class ToolRegistry:
    """Управляет загрузкой и инициализацией инструментов."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools: List[BaseTool] = []

    async def load_all(self):
        """Загружает все доступные группы инструментов."""
        self._load_file_tools()
        self._load_search_tools()
        
        if self.config.use_long_term_memory:
            self._load_memory_tools()
            
        if self.config.mcp_config_path.exists():
            await self._load_mcp_tools()

        logger.info(f"✅ Tools loaded: {[t.name for t in self.tools]}")

    def _load_file_tools(self):
        try:
            from delete_tools import SafeDeleteFileTool, SafeDeleteDirectoryTool
            cwd = Path.cwd()
            self.tools.extend([
                SafeDeleteFileTool(root_dir=cwd),
                SafeDeleteDirectoryTool(root_dir=cwd)
            ])
        except ImportError:
            pass

    def _load_search_tools(self):
        try:
            from search_tools import web_search, deep_search, fetch_content
            
            if web_search and fetch_content:
                self.tools.extend([web_search, fetch_content])
            
            if self.config.enable_deep_search and deep_search:
                logger.info("🔹 Deep Search tool is ENABLED")
                self.tools.append(deep_search)
        except ImportError:
            logger.warning("Search tools dependencies missing.")

    def _load_memory_tools(self):
        try:
            from memory_manager import MemoryManager
            memory = MemoryManager(db_path=self.config.memory_db_path)
            
            @tool
            async def remember_fact(text: str, category: str = "general") -> str:
                return await memory.aremember(text, {"type": category})
            
            @tool
            async def recall_facts(query: str) -> str:
                facts = await memory.arecall(query)
                return "\n".join(f"- {f}" for f in facts) if facts else "No facts found."
            
            @tool
            async def forget_fact(query: str) -> str:
                return f"Forgotten: {await memory.adelete_fact_by_query(query)}"

            self.tools.extend([remember_fact, recall_facts, forget_fact])
        except ImportError:
            logger.warning("MemoryManager not available.")

    async def _load_mcp_tools(self):
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            raw_cfg = json.loads(self.config.mcp_config_path.read_text("utf-8"))
            mcp_cfg = {
                name: {
                    **{k: v for k, v in cfg.items() if k != 'enabled'},
                    "args": cfg.get("args", [])
                }
                for name, cfg in raw_cfg.items() if cfg.get("enabled", True)
            }
            if mcp_cfg:
                client = MultiServerMCPClient(mcp_cfg)
                new_tools = await asyncio.wait_for(client.get_tools(), timeout=120)
                self.tools.extend(new_tools)
        except Exception as e:
            logger.error(f"MCP Load Error: {e}")


# ==========================================
# 4. СОСТОЯНИЕ (STATE)
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    steps: int


# ==========================================
# 5. ГРАФ АГЕНТА (WORKFLOW)
# ==========================================

class AgentWorkflow:
    def __init__(self):
        load_dotenv()
        self.config = AgentConfig()
        self.utils = AgentUtils()
        self.tool_registry = ToolRegistry(self.config)
        
        self.llm: Optional[BaseChatModel] = None
        self.llm_with_tools: Optional[BaseChatModel] = None

    async def initialize_resources(self):
        logger.info(f"Initializing agent: [bold cyan]{self.config.provider}[/]", extra={"markup": True})
        
        # 1. LLM
        self.llm = self.config.get_llm()
        
        # 2. Tools
        await self.tool_registry.load_all()
        
        # 3. Binding с проверкой поддержки инструментов!
        can_use_tools = self.config.check_tool_support()
        
        if self.tool_registry.tools and can_use_tools:
            try:
                self.llm_with_tools = self.llm.bind_tools(self.tool_registry.tools)
                logger.info("🛠️ Tools bound to LLM successfully.")
            except Exception as e:
                logger.error(f"Failed to bind tools (LLM might not support them): {e}")
                self.llm_with_tools = self.llm
        else:
            if not can_use_tools:
                logger.warning("⚠️ Tools disabled: Model does not support tool calling (or disabled in config).")
            self.llm_with_tools = self.llm

    @property
    def tools(self) -> List[BaseTool]:
        return self.tool_registry.tools

    # --- NODES ---

    async def _summarize_node(self, state: AgentState):
        """Узел сжатия истории сообщений."""
        messages = state["messages"]
        summary = state.get("summary", "")

        if len(messages) <= self.config.summary_threshold:
            return {}

        # Определяем точку среза (оставляем последние N)
        idx = len(messages) - self.config.summary_keep_last
        while idx < len(messages) and idx > 0:
            if isinstance(messages[idx], HumanMessage):
                break
            idx += 1
        
        to_summarize = messages[:idx]
        if not to_summarize: return {}

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
            logger.info(f"🧹 Summary: Removed {len(delete_msgs)} messages.")
            return {"summary": res.content, "messages": delete_msgs}
        except Exception as e:
            logger.error(f"Summarization Error: {e}")
            return {}

    async def _agent_node(self, state: AgentState):
        """Основной узел принятия решений."""
        # 1. Подготовка контекста
        messages = state["messages"]
        
        # Определяем, доступны ли инструменты для текущего запуска
        tools_available = (self.llm_with_tools != self.llm)
        
        sys_msg = self._build_system_message(state.get("summary", ""), tools_available)
        full_context = [sys_msg] + [m for m in messages if not isinstance(m, SystemMessage)]
        
        # 2. Вызов LLM с Retry Logic
        response = await self._invoke_llm_with_retry(full_context)
        
        # 3. Пост-обработка (Sanitizing)
        if response.tool_calls:
            self.utils.fix_tool_calls(response.tool_calls)
            
        # 4. Проверка безопасности (Quality Gate)
        if self._is_unsafe_write(response, full_context):
            response = AIMessage(
                content="STOP. You are trying to write a file without valid data from search/fetch. "
                        "Perform a search first to get actual content."
            )

        # 5. Патч токенов
        self._patch_token_usage(response, full_context)

        return {"messages": [response]}

    async def _loop_guard_node(self, state: AgentState):
        return {"messages": [AIMessage(content="🛑 **Auto-Stop**: Max steps limit reached.")]}

    # --- HELPERS ---

    def _build_system_message(self, summary: str, tools_available: bool = True) -> SystemMessage:
        if self.config.prompt_path.exists():
            raw_prompt = self.config.prompt_path.read_text("utf-8")
        else:
            raw_prompt = (
                "You are an autonomous AI agent.\n"
                "Reason in English, Reply in Russian.\n"
                "Date: {{current_date}}\nCWD: {{cwd}}"
            )
        
        prompt = raw_prompt.replace("{{current_date}}", datetime.now().strftime("%Y-%m-%d"))
        prompt = prompt.replace("{{cwd}}", str(Path.cwd()))
        
        if not tools_available:
            prompt += "\nNOTE: You are in CHAT-ONLY mode. Tools are disabled for this session. Do not try to use tools."
        elif self.config.use_long_term_memory:
             prompt += "\nUse memory tools (recall_facts/remember_fact) when necessary."
             
        if summary:
            prompt += f"\n\n<memory>\n{summary}\n</memory>"
            
        return SystemMessage(content=prompt)

    async def _invoke_llm_with_retry(self, context: List[BaseMessage]) -> AIMessage:
        """Попытка вызова LLM с обработкой ошибок и 'ленивых' ответов."""
        for attempt in range(3):
            try:
                response = await self.llm_with_tools.ainvoke(context)
                if not response.content and not response.tool_calls:
                    raise ValueError("Empty response")
                return response
            except Exception as e:
                # Авто-комплит при успешной записи файла (частый баг)
                last_msg = context[-1] if context else None
                if isinstance(last_msg, ToolMessage) and "Successfully wrote" in str(last_msg.content):
                    logger.info("🛡️ Auto-completing after write_file crash.")
                    return AIMessage(content="Файл записан. (Авто-завершение)")
                
                logger.debug(f"⚠️ LLM Retry {attempt+1}/3: {e}")
                if attempt == 2:
                    return AIMessage(content=f"System Error: {e}")
                await asyncio.sleep(1)
        return AIMessage(content="System Error: Unknown")

    def _is_unsafe_write(self, response: AIMessage, history: List[BaseMessage]) -> bool:
        """Блокирует запись файла, если в истории нет успешных чтений/поиска."""
        if not response.tool_calls: return False
        if not any(tc['name'] == 'write_file' for tc in response.tool_calls): return False
        
        # Проверяем, были ли успешные получения данных
        has_data = False
        valid_sources = [
            "fetch_content",    # Универсальный инструмент
            "web_search", 
            "deep_search",      
            "read_text_file"
        ]
        
        for m in history:
            if isinstance(m, ToolMessage) and m.name in valid_sources:
                content = str(m.content)
                is_system_error = content.startswith("System:") or content.startswith("Error:")
                
                if not is_system_error and len(content) > 50:
                    has_data = True
                    break
                    
        if not has_data:
            logger.warning("🛡️ Quality Gate: Blocked write_file (no data source).")
            return True
        return False

    def _patch_token_usage(self, response: AIMessage, context: List[BaseMessage]):
        """Добавляет метаданные о токенах, если их нет (для OpenAI/Compatible)."""
        usage = response.usage_metadata or {}
        if usage.get("input_tokens", 0) == 0:
            input_tokens = self.utils.estimate_payload_tokens(context, self.tools)
            output_content = response.content
            if isinstance(output_content, list):
                output_content = " ".join([str(x) for x in output_content])
            
            output_tokens = self.utils.count_tokens(str(output_content))
            if response.tool_calls:
                output_tokens += self.utils.count_tokens(json.dumps(response.tool_calls, default=str))

            response.usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }

    # --- GRAPH BUILDER ---

    def build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("loop_guard", self._loop_guard_node)
        workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})
        
        # 🔥 ИСПРАВЛЕНИЕ: Добавляем узел tools ТОЛЬКО если они реально доступны
        # Это предотвращает маршрутизацию в несуществующий узел для простых моделей
        tools_enabled = bool(self.tools) and self.config.check_tool_support()
        
        if tools_enabled:
            workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "update_step") 
        workflow.add_edge("update_step", "agent")

        def should_continue(state):
            steps = state.get("steps", 0)
            if steps >= self.config.max_loops:
                logger.warning(f"🛑 Loop Guard: {steps} steps.")
                return "loop_guard" 
            
            # 🔥 Если тулы отключены - ВСЕГДА выходим, даже если модель бредит
            if not tools_enabled:
                return END
                
            last_msg = state["messages"][-1]
            return "tools" if getattr(last_msg, "tool_calls", None) else END

        # 🔥 Динамические переходы
        destinations = ["tools", "loop_guard", END] if tools_enabled else ["loop_guard", END]
        workflow.add_conditional_edges("agent", should_continue, destinations)

        if tools_enabled:
            workflow.add_edge("tools", "update_step")

        workflow.add_edge("loop_guard", END)

        return workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    async def main():
        wf = AgentWorkflow()
        await wf.initialize_resources()
        print(f"✅ Agent Ready. Tools: {len(wf.tools)}")
        # wf.build_graph() # Test graph build

    asyncio.run(main())