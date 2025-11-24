# logging_config.py
import logging
import os
import sys  # <--- Добавлено для stderr
from typing import Optional

class IgnoreSchemaWarnings(logging.Filter):
    """Фильтр для подавления предупреждений о схемах LangChain."""
    def filter(self, record):
        ignore_messages = [
            "Key 'additionalProperties' is not supported",
            "Key '$schema' is not supported",
        ]
        return not any(msg in record.getMessage() for msg in ignore_messages)

class MaxLevelFilter(logging.Filter):
    """Пропускает сообщения только до определенного уровня (для stdout)."""
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record) -> bool:  # type: ignore[override]
        return record.levelno <= self.max_level

def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    if log_file is None:
        log_file = os.getenv("LOG_FILE", "ai_agent.log")

    handlers: list[logging.Handler] = []

    # 1. STDOUT: Только INFO и WARNING (обычные сообщения)
    # Это не сломает CLI интерфейс Rich, так как туда обычно идут только print()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(MaxLevelFilter(logging.WARNING))
    stdout_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(stdout_handler)

    # 2. STDERR: Только ERROR и CRITICAL (Ошибки всегда должны быть видны!)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(stderr_handler)

    # 3. FILE: Все подряд (для отладки)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(format_string))
            handlers.append(file_handler)
        except Exception as e:
            print(f"⚠️ Не удалось создать лог-файл: {e}", file=sys.stderr)

    # Настройка root логгера
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Фильтр схем (применяем ко всем)
    schema_filter = IgnoreSchemaWarnings()
    for handler in handlers:
        handler.addFilter(schema_filter)

    # Подавление шума библиотек (Ваш список корректен для вашего стека)
    noisy_loggers = [
        "langchain_mcp_adapters",
        "mcp",
        "jsonschema",
        "langchain_google_genai",
        "httpcore",
        "httpx",
        "openai",
        "chromadb", 
        "hnswlib",
        "google.ai.generativelanguage" # Иногда тоже шумит
    ]
    
    lib_level = logging.ERROR if level > logging.DEBUG else logging.WARNING
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(lib_level)

    return logging.getLogger("AgentCore") # Даем имя логгеру, чтобы видеть источник