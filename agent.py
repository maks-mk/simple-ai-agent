import os
import json
import warnings
import asyncio
from pathlib import Path
from typing import List, Any, Dict, Optional, Literal
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime

# Third-party imports
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# LLM Providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# MCP
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None

# Local imports
from logging_config import setup_logging

# Опциональные локальные инструменты
try:
    from delete_tools import SafeDeleteFileTool, SafeDeleteDirectoryTool
except ImportError:
    SafeDeleteFileTool = None
    SafeDeleteDirectoryTool = None

# Настройка окружения
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*create_react_agent has been moved.*")
logger = setup_logging()

# === КОНФИГУРАЦИЯ ===

@dataclass
class AgentConfig:
    """Конфигурация агента с загрузкой из переменных окружения."""
    provider: Literal["gemini", "openai"] = field(default="gemini")
    
    # API Keys & Models
    gemini_key: Optional[str] = field(default=None)
    gemini_model: str = field(default=None)
    openai_key: Optional[str] = field(default=None)
    openai_model: str = field(default=None)
    openai_base_url: Optional[str] = field(default=None)
    
    # Parameters
    temperature: float = 0.5
    max_retries: int = 3
    retry_delay: int = 2
    
    # Paths
    mcp_config_path: Path = Path("mcp.json")
    prompt_path: Path = Path("prompt.txt")
    memory_db_path: str = "./memory_db"
    
    # Flags
    use_long_term_memory: bool = False
    session_size: int = 6

    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Фабричный метод для создания конфига из .env"""
        load_dotenv()
        return cls(
            provider=os.getenv("PROVIDER", "gemini").lower(),
            gemini_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL"),
            openai_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.5")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("RETRY_DELAY", "2")),
            use_long_term_memory=os.getenv("LONG_TERM_MEMORY", "false").lower() == "true",
            session_size=int(os.getenv("SESSION_SIZE", "6")),
        )

# === РАБОТА С ПРОМПТАМИ ===

@lru_cache(maxsize=1)
def _read_prompt_template(path: Path) -> str:
    """Читает шаблон промпта с диска (кэшируется)."""
    base_dir = Path(__file__).parent
    full_path = base_dir / path
    
    if full_path.exists():
        try:
            return full_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Ошибка чтения промпта {full_path}: {e}")
    
    return "Ты полезный AI-ассистент."

def get_system_prompt(config: AgentConfig) -> str:
    """Формирует финальный системный промпт с динамическими данными."""
    content = _read_prompt_template(config.prompt_path)
    
    now = datetime.now()
    replacements = {
        "{{current_date}}": now.strftime("%Y-%m-%d (%A)"),
        "{{current_time}}": now.strftime("%H:%M")
    }

    for key, value in replacements.items():
        if key in content:
            content = content.replace(key, value)
    
    # Если плейсхолдеров не было, добавляем инфо в конец
    if "{{current_date}}" not in _read_prompt_template(config.prompt_path):
        content += f"\n\n[System Info]\nDate: {replacements['{{current_date}}']}\nTime: {replacements['{{current_time}}']}"
    
    return f"{content}\n\nCWD: {Path.cwd()}"

def load_mcp_config(config_path: Path) -> Dict[str, Any]:
    """Читает и валидирует конфиг MCP."""
    base_dir = Path(__file__).parent
    path = base_dir / config_path
    
    if not path.exists():
        return {}
    
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        current_dir = str(Path.cwd())
        
        filtered = {}
        for name, cfg in data.items():
            if not cfg.get("enabled", True):
                continue
            
            clean_cfg = cfg.copy()
            clean_cfg.pop("enabled", None)
            
            # Подстановка пути к текущей директории
            if "args" in clean_cfg:
                clean_cfg["args"] = [
                    arg.replace("{filesystem_path}", current_dir) 
                    for arg in clean_cfg["args"]
                ]
            filtered[name] = clean_cfg
        return filtered
    except Exception as e:
        logger.error(f"Ошибка парсинга MCP конфига: {e}")
        return {}

# === ИНСТРУМЕНТЫ ПАМЯТИ ===

class MemoryToolsFactory:
    """Фабрика для создания инструментов памяти с замыканием на экземпляр БД."""
    
    @staticmethod
    def create(db_path: str, session_size: int) -> List[BaseTool]:
        try:
            from memory_manager import MemoryManager
            # Инициализация менеджера памяти
            memory = MemoryManager(db_path=db_path, session_size=session_size)
            logger.info(f"🧠 Long-term memory loaded from {db_path}")

            # Определение инструментов
            @tool
            def remember_fact(text: str, category: str = "general") -> str:
                """Сохраняет важный факт о пользователе или проекте в долговременную память."""
                try:
                    memory.remember(text, metadata={"type": category})
                    return f"✅ Запомнил: {text}"
                except Exception as e:
                    return f"Error saving to memory: {e}"

            @tool
            def recall_facts(query: str) -> str:
                """Ищет информацию в долговременной памяти."""
                try:
                    facts = memory.recall(query)
                    if not facts:
                        return "Ничего релевантного в памяти не найдено."
                    return "Найдено в памяти:\n" + "\n".join(f"- {f}" for f in facts)
                except Exception as e:
                    return f"Error recalling memory: {e}"

            @tool
            def delete_facts(query: str) -> str:
                """Удаляет неактуальный факт из памяти по поисковому запросу."""
                try:
                    count = memory.delete_fact_by_query(query)
                    if count > 0:
                        return f"✅ Удалено {count} фактов по запросу: '{query}'."
                    return f"Факты для удаления по запросу '{query}' не найдены."
                except Exception as e:
                    return f"Error deleting memory: {e}"

            return [remember_fact, recall_facts, delete_facts]

        except ImportError:
            logger.warning("⚠️ Модуль memory_manager не найден. Память отключена.")
            return []
        except Exception as e:
            logger.error(f"⚠️ Ошибка инициализации памяти: {e}")
            return []

# === ИНИЦИАЛИЗАЦИЯ ИНСТРУМЕНТОВ ===

async def init_tools(config: AgentConfig) -> List[BaseTool]:
    """Сбор всех инструментов (MCP, локальные, память)."""
    all_tools = []
    
    # 1. MCP Tools
    if MultiServerMCPClient:
        mcp_cfg = load_mcp_config(config.mcp_config_path)
        if mcp_cfg:
            try:
                async with asyncio.timeout(10):
                    client = MultiServerMCPClient(mcp_cfg)
                    mcp_tools = await client.get_tools()
                    all_tools.extend(mcp_tools)
                    logger.info(f"MCP Tools initialized: {len(mcp_tools)}")
            except Exception as e:
                logger.error(f"MCP Init Failed: {e}")
    else:
        logger.warning("MCP client library not installed.")

    # 2. Local File Tools
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

    # 3. Memory Tools
    if config.use_long_term_memory:
        mem_tools = MemoryToolsFactory.create(config.memory_db_path, config.session_size)
        all_tools.extend(mem_tools)
    
    # Включаем глобальную обработку ошибок для инструментов
    for t in all_tools:
        t.handle_tool_error = True
        
    return all_tools

# === СОЗДАНИЕ LLM ===

def create_llm(config: AgentConfig) -> BaseChatModel:
    """Создает экземпляр LLM."""
    if config.provider == "gemini":
        if not config.gemini_key:
            raise ValueError("GEMINI_API_KEY не задан")
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            temperature=config.temperature,
            google_api_key=config.gemini_key,
            max_retries=config.max_retries,
            transport="rest",
        )
    elif config.provider == "openai":
        if not config.openai_key:
            raise ValueError("OPENAI_API_KEY не задан")
        return ChatOpenAI(
            model=config.openai_model,
            temperature=config.temperature,
            api_key=config.openai_key,
            base_url=config.openai_base_url,
            max_retries=config.max_retries,
        )
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

# === СБОРКА ГРАФА ===

async def create_agent_graph(config: Optional[AgentConfig] = None):
    """
    Создает готовый к запуску граф агента.
    """
    if config is None:
        config = AgentConfig.from_env()

    # Параллельная/последовательная инициализация ресурсов
    tools = await init_tools(config)
    llm = create_llm(config)
    
    # Привязка инструментов и настройка повторов
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm # Агент без инструментов
        
    llm_robust = llm_with_tools.with_retry(
        stop_after_attempt=config.max_retries,
        wait_exponential_jitter=True
    )

    # Генерация системного промпта
    system_prompt = get_system_prompt(config)

    # Создание агента
    # FIX: Используем messages_modifier вместо state_modifier для совместимости
    try:
        agent = create_react_agent(
            model=llm_robust,
            tools=tools,
            messages_modifier=system_prompt, 
            checkpointer=MemorySaver()
        )
    except TypeError:
        # Fallback для очень старых версий, где аргумент назывался prompt
        agent = create_react_agent(
            model=llm_robust,
            tools=tools,
            prompt=system_prompt,
            checkpointer=MemorySaver()
        )
    
    return agent