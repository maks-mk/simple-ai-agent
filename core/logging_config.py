import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None

class NoisyLogFilter(logging.Filter):
    """
    Фильтр для подавления конкретных назойливых сообщений,
    которые пробиваются даже через настройки уровней.
    """
    BLOCKED_PHRASES = [
        "Key 'additionalProperties' is not supported",
        "Key '$schema' is not supported",
        "AFC is enabled",
        "HTTP Request: POST",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(phrase in msg for phrase in self.BLOCKED_PHRASES)

from core.constants import BASE_DIR

def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Настраивает логирование приложения с поддержкой Rich и файлового вывода.
    """
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    if log_file is None:
        # Используем абсолютный путь через BASE_DIR
        rel_path = os.getenv("LOG_FILE", "logs/agent.log")
        log_file = str(BASE_DIR / rel_path)

    handlers: List[logging.Handler] = []

    # 1. Console Handler (Rich or Standard)
    if RichHandler:
        console_handler = RichHandler(
            rich_tracebacks=False,
            markup=True,
            show_path=False,
            show_time=True,
            omit_repeated_times=False
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
    
    console_handler.setLevel(level)
    handlers.append(console_handler)

    # 2. File Handler
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # В файл пишем всё
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            handlers.append(file_handler)
        except Exception as e:
            sys.stderr.write(f"⚠️ Warning: Could not create log file: {e}\n")

    # 3. Apply Configuration
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # 4. Apply Filters
    noise_filter = NoisyLogFilter()
    for h in handlers:
        h.addFilter(noise_filter)

    # 5. Suppress Noisy Libraries
    _suppress_library_logs(level)

    # 6. Ensure 'agent' logger captures everything (handlers will filter)
    agent_logger = logging.getLogger("agent")
    agent_logger.setLevel(logging.DEBUG)

    return agent_logger

def _suppress_library_logs(root_level: int):
    """
    Подавляет логи болтливых библиотек.
    """
    noisy_modules = [
        "langchain_google_genai", "google.ai.generativelanguage", "google.auth",
        "openai", "httpx", "httpcore", "urllib3",
        "langchain", "langchain_core", "langgraph", "langchain_mcp_adapters",
        "mcp", "pydantic", "jsonschema", "chromadb", 
        "hnswlib", "sentence_transformers", "filelock",
        "grpc", "grpc._cython", "multipart",
        "markdown_it", "markdown_it.rules_block", "markdown_it.rules_inline"
    ]
    
    lib_level = logging.WARNING if root_level == logging.DEBUG else logging.ERROR
    
    for module_name in noisy_modules:
        logging.getLogger(module_name).setLevel(lib_level)
