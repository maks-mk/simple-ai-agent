import os
import json
import logging
import warnings
import asyncio
from pathlib import Path
from typing import List, Any, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache

# LangChain / LangGraph
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool # Добавить этот импорт

# Локальные импорты
from logging_config import setup_logging
# Пробуем импортировать инструменты удаления, если они есть
try:
    from delete_tools import SafeDeleteFileTool, SafeDeleteDirectoryTool
except ImportError:
    SafeDeleteFileTool = None
    SafeDeleteDirectoryTool = None

# Настройка логгера
logger = setup_logging()
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*create_react_agent has been moved.*")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

load_dotenv()

# === КОНФИГУРАЦИЯ ===

@dataclass
class AgentConfig:
    """Конфигурация агента."""
    provider: str = os.getenv("PROVIDER", "gemini").lower()
    gemini_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")
    mcp_config_path: str = "mcp.json"
    prompt_path: str = "prompt.txt"
    system_prompt_default: str = "Ты полезный AI-ассистент."
    #session_size: int = int(os.getenv("SESSION_SIZE", "6"))

    # Настройки памяти (читаем из .env)
    use_long_term_memory: bool = os.getenv("LONG_TERM_MEMORY", "false").lower() == "true"
    memory_db_path: str = "./memory_db"
    session_size: int = int(os.getenv("SESSION_SIZE", "6"))

@lru_cache(maxsize=1)
def load_system_prompt(path_str: str = "prompt.md") -> str:
    """Читает системный промпт с диска с кэшированием."""
    path = Path.cwd() / path_str
    default = "Ты полезный AI-ассистент."
    
    if not path.exists():
        return default
    
    try:
        content = path.read_text(encoding='utf-8')
        return f"{content}\n\nCWD: {Path.cwd()}"
    except Exception as e:
        logger.error(f"Ошибка чтения промпта {path}: {e}")
        return default

def load_mcp_config(path_str: str = "mcp.json") -> Dict[str, Any]:
    """Читает конфиг MCP серверов."""
    path = Path.cwd() / path_str
    if not path.exists():
        return {}
    
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        current_dir = str(Path.cwd())
        filtered = {}
        for name, cfg in config.items():
            if not cfg.get("enabled", True):
                continue
            # Копируем, чтобы не менять исходный словарь
            clean = cfg.copy()
            clean.pop("enabled", None)
            
            if "args" in clean:
                clean["args"] = [a.replace("{filesystem_path}", current_dir) for a in clean["args"]]
            filtered[name] = clean
        return filtered
    except Exception as e:
        logger.error(f"Ошибка конфига MCP: {e}")
        return {}

# === СОЗДАНИЕ КОМПОНЕНТОВ ===

def create_memory_tools(db_path: str, session_size: int) -> List[Any]:
    """Создает инструменты для работы с памятью, если включено."""
    try:
        from memory_manager import MemoryManager
        
        memory = MemoryManager(db_path=db_path, session_size=session_size)
        print(f"🧠 Long-term memory loaded from {db_path}")

        # === 1. Инструмент сохранения ===
        @tool
        def remember_fact(text: str, category: str = "general") -> str:
            """
            Сохраняет важный факт о пользователе или проекте в долговременную память.
            ... (описание) ...
            """
            try:
                # ВАЖНО: Убедитесь, что в memory_manager.py есть import logging и logger
                memory.remember(text, metadata={"type": category}) 
                return f"✅ Запомнил: {text}"
            except Exception as e:
                return f"Error saving to memory: {e}"

        # === 2. Инструмент поиска ===
        @tool
        def recall_facts(query: str) -> str:
            """
            Ищет информацию в долговременной памяти.
            ... (описание) ...
            """
            try:
                facts = memory.recall(query)
                if not facts:
                    return "Ничего релевантного в памяти не найдено."
                return "Найдено в памяти:\n" + "\n".join(f"- {f}" for f in facts)
            except Exception as e:
                return f"Error recalling memory: {e}"

        # === 3. Инструмент удаления (новый) ===
        @tool
        def delete_facts(query: str) -> str:
            """
            Удаляет неактуальный или ошибочный факт из долговременной памяти по поисковому запросу.
            ... (описание) ...
            """
            try:
                # ВАЖНО: Убедитесь, что в MemoryManager есть delete_fact_by_query
                count = memory.delete_fact_by_query(query) 
                if count > 0:
                    return f"✅ Удалено {count} устаревших фактов, связанных с запросом: '{query}'."
                return f"Ничего релевантного для удаления по запросу '{query}' не найдено."
            except Exception as e:
                return f"Error deleting memory: {str(e)}"

        # === Возвращаем ВСЕ инструменты ===
        return [remember_fact, recall_facts, delete_facts]

    except ImportError:
        logger.warning("⚠️ Модуль memory_manager.py не найден или не установлены зависимости (chromadb, sentence_transformers). Память отключена.")
        return []
    except Exception as e:
        logger.error(f"⚠️ Ошибка инициализации памяти: {e}")
        return []        
#(начало функции init_tools) ...
async def init_tools(config: Optional[AgentConfig] = None) -> List[Any]:
    """Инициализирует MCP клиент и локальные инструменты."""
    if config is None:
        config = AgentConfig()
        
    all_tools = []
    
    # 1. Загрузка MCP Tools
    mcp_cfg = load_mcp_config(config.mcp_config_path)
    if mcp_cfg:
        try:
            async with asyncio.timeout(10):
                client = MultiServerMCPClient(mcp_cfg)
                mcp_tools = await client.get_tools()
                logger.info(f"MCP Tools initialized: {len(mcp_tools)}")
                all_tools.extend(mcp_tools)
        except Exception as e:
            logger.error(f"MCP Init Failed: {e}")

    # 2. Загрузка локальных инструментов
    if SafeDeleteFileTool and SafeDeleteDirectoryTool:
        try:
            work_dir = Path.cwd()
            local_tools = [
                SafeDeleteFileTool(root_dir=work_dir),
                SafeDeleteDirectoryTool(root_dir=work_dir)
            ]
            all_tools.extend(local_tools)
            logger.info(f"Local Tools initialized: {len(local_tools)}")
        except Exception as e:
            logger.error(f"Local Tools Init Failed: {e}")

# --- 3. Загрузка инструментов памяти (НОВОЕ) ---
    if config.use_long_term_memory:
        mem_tools = create_memory_tools(config.memory_db_path, config.session_size)
        if mem_tools:
            all_tools.extend(mem_tools)
            logger.info(f"Memory Tools initialized: {len(mem_tools)}")
    else:
        logger.info("Memory Tools disabled (check .env LONG_TERM_MEMORY)")

    # === ДОБАВИТЬ ЭТОТ БЛОК (ВАЖНО!) ===
    # Включаем автоматическую обработку ошибок для всех инструментов
    for tool in all_tools:
        # Если инструмент упадет с ошибкой, она вернется агенту как текст,
        # а не крашнет программу.
        tool.handle_tool_error = True 
    # ===================================

    return all_tools

def create_llm(config: Optional[AgentConfig] = None) -> BaseChatModel:
    """Создает LLM (Gemini или OpenAI) на основе конфига."""
    if config is None:
        config = AgentConfig()
        
    if config.provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            temperature=0.2,
            google_api_key=config.gemini_key,
            streaming=True
        )
    else:
        return ChatOpenAI(
            model=config.openai_model,
            temperature=0.2,
            api_key=config.openai_key,
            base_url=config.openai_base_url,
            streaming=True
        )

async def create_agent_graph(config: Optional[AgentConfig] = None):
    """Сборка всего графа агента."""
    if config is None:
        config = AgentConfig()
        
    tools = await init_tools(config)
    llm = create_llm(config)
    prompt = load_system_prompt(config.prompt_path)

    agent = create_react_agent(
        llm,
        tools,
        prompt=prompt,
        checkpointer=MemorySaver()
    )
    return agent
