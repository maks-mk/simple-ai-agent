import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

# Импорт Rich для красивого вывода
try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None

class IgnoreSchemaWarnings(logging.Filter):
    """Фильтр для подавления специфических предупреждений Pydantic/LangChain."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "Key 'additionalProperties' is not supported" not in msg and \
               "Key '$schema' is not supported" not in msg

def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    
    # 1. Определяем уровень логирования
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    if log_file is None:
        log_file = os.getenv("LOG_FILE", "logs/agent.log")

    handlers: List[logging.Handler] = []

    # 2. Консольный хендлер (Rich или стандартный)
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
        console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    
    console_handler.setLevel(level)
    handlers.append(console_handler)

    # 3. Файловый хендлер (Всегда пишет DEBUG для истории)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG) 
            file_handler.setFormatter(file_fmt)
            handlers.append(file_handler)
        except Exception as e:
            # Не падаем, если нет прав на запись лога, просто пишем в stderr
            sys.stderr.write(f"⚠️ Warning: Could not create log file: {e}\n")

    # 4. Применяем конфигурацию
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # 5. Фильтры шума
    schema_filter = IgnoreSchemaWarnings()
    for h in handlers:
        h.addFilter(schema_filter)

    # Список болтливых библиотек
    noisy_modules = [
        "langchain_mcp_adapters", "mcp", "jsonschema", 
        "langchain_google_genai", "google.ai.generativelanguage",
        "httpcore", "httpx", "openai", "urllib3", "multipart",
        "chromadb", "hnswlib", "sentence_transformers", "filelock",
        "grpc", "grpc._cython", "pydantic", "langgraph"
    ]
    
    # Если мы в режиме DEBUG, библиотекам даем WARNING, иначе ERROR
    lib_level = logging.WARNING if level == logging.DEBUG else logging.ERROR
    
    for module_name in noisy_modules:
        logging.getLogger(module_name).setLevel(lib_level)

    return logging.getLogger("Agent")