import os
import json
import logging
import warnings
import asyncio
from pathlib import Path
from typing import List, Any, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime

# LangChain / LangGraph
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

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
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay: int = int(os.getenv("RETRY_DELAY", "2"))
    mcp_config_path: str = "mcp.json"
    prompt_path: str = "prompt.txt"
    system_prompt_default: str = "Ты полезный AI-ассистент."
    # Настройки памяти (читаем из .env)
    use_long_term_memory: bool = os.getenv("LONG_TERM_MEMORY", "false").lower() == "true"
    memory_db_path: str = "./memory_db"
    session_size: int = int(os.getenv("SESSION_SIZE", "6"))

@lru_cache(maxsize=1)
def _read_prompt_template(path_str: str) -> str:
    base_dir = Path(__file__).parent
    path = base_dir / path_str
    default = "Ты полезный AI-ассистент."
    
    if path.exists():
        try:
            return path.read_text(encoding='utf-8')
        except Exception as e:
            # Используем глобальный объект logger, который вы создали выше через setup_logging()
            logger.error(f"Ошибка чтения промпта {path}: {e}")
            return default
    return default

# 2. Публичная функция: собирает промпт с актуальным временем (НЕ кэшируется)
def load_system_prompt(path_str: str = "prompt.txt") -> str:
    """
    Читает системный промпт с диска и внедряет динамические переменные (дата, CWD).
    """
    # Получаем текст из кэша
    content = _read_prompt_template(path_str)

    # Вычисляем свежее время
    now = datetime.now()
    current_date_str = now.strftime("%Y-%m-%d (%A)")
    current_time_str = now.strftime("%H:%M")

    # Подставляем переменные
    if "{{current_date}}" in content:
        content = content.replace("{{current_date}}", current_date_str)
        # Можно добавить поддержку времени, если нужно
        if "{{current_time}}" in content:
             content = content.replace("{{current_time}}", current_time_str)
    else:
        # Если меток нет, добавляем блок в конец
        content += f"\n\n[System Info]\nCurrent Date: {current_date_str}\nCurrent Time: {current_time_str}"

    return f"{content}\n\nCWD: {Path.cwd()}"
    
def load_mcp_config(path_str: str = "mcp.json") -> Dict[str, Any]:
    """Читает конфиг MCP серверов."""
    path = Path(__file__).parent / path_str
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

        # Конструктор MemoryManager может упасть из-за проблем с ChromaDB/файловой системой.
        try:
            memory = MemoryManager(db_path=db_path, session_size=session_size)
        except Exception as db_err:
            logger.error(f"ChromaDB init failed: {db_err}")
            return []

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
    
    provider = config.provider
    if provider not in ("gemini", "openai"):
        raise ValueError(f"Неподдерживаемый PROVIDER: {provider}")
    
    if provider == "gemini":
        if not config.gemini_key:
            raise RuntimeError("GEMINI_API_KEY не задан. Установите его в .env")
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            temperature=config.temperature,
            google_api_key=config.gemini_key,
            max_retries=config.max_retries,
            streaming=True
        )
    else:
        if not config.openai_key:
            raise RuntimeError("OPENAI_API_KEY не задан. Установите его в .env")
        return ChatOpenAI(
            model=config.openai_model,
            temperature=config.temperature,
            api_key=config.openai_key,
            base_url=config.openai_base_url,
            streaming=True
        )

async def create_agent_graph(config: Optional[AgentConfig] = None):
    """Сборка всего графа агента с гарантированным Retry."""
    if config is None:
        config = AgentConfig()
        
    tools = await init_tools(config)
    llm = create_llm(config)
    prompt = load_system_prompt(config.prompt_path)

    # === ИСПРАВЛЕНИЕ: ПРИНУДИТЕЛЬНЫЙ RETRY ===
    
    # 1. Сначала привязываем инструменты.
    # Это важно сделать ДО retry, чтобы модель знала о функциях.
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. Оборачиваем модель с инструментами в логику повторов.
    # stop_after_attempt - сколько всего попыток (включая первую).
    # wait_exponential_jitter - умная задержка (1с, 2с, 4с...), чтобы не дудосить API.
    llm_robust = llm_with_tools.with_retry(
        stop_after_attempt=config.max_retries,
        wait_exponential_jitter=True
    )
    
    # 3. Создаем агента, передавая уже "обернутую" модель.
    agent = create_react_agent(
        model=llm_robust,  # Передаем модель с Retry
        tools=tools,
        prompt=prompt,
        checkpointer=MemorySaver()
    )
    # =========================================
    
    return agent