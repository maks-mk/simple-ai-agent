"""
Настройка логирования для Smart Gemini Agent
"""

import logging
import os
from typing import Optional

class IgnoreSchemaWarnings(logging.Filter):
    """Фильтр для подавления предупреждений о схемах"""

    def filter(self, record):
        ignore_messages = [
            "Key 'additionalProperties' is not supported in schema, ignoring",
            "Key '$schema' is not supported in schema, ignoring",
        ]
        return not any(msg in record.getMessage() for msg in ignore_messages)


class MaxLevelFilter(logging.Filter):
    """Фильтр, который пропускает только сообщения не выше заданного уровня."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record) -> bool:  # type: ignore[override]
        return record.levelno <= self.max_level


def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    format_string: str = "%(asctime)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Настройка логирования для агента.
    
    Если параметры не переданы, пытается взять их из переменных окружения:
    - LOG_LEVEL (DEBUG, INFO, WARNING, ERROR)
    - LOG_FILE (путь к файлу)
    """
    
    # Определяем уровень логирования
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    # Определяем файл логов
    if log_file is None:
        log_file = os.getenv("LOG_FILE", "ai_agent.log")

    # Создаем обработчики
    handlers: list[logging.Handler] = []
    
    # 1. Console Handler (Stdout) - только до WARNING
    # Мы хотим видеть обычные логи в консоли, но ошибки - красным цветом (если Rich) 
    # или просто отдельно. В данной конфигурации фильтруем ERROR/CRITICAL, 
    # чтобы они не дублировались, если приложение само их печатает.
    stream_handler = logging.StreamHandler()
    stream_handler.addFilter(MaxLevelFilter(logging.WARNING))
    handlers.append(stream_handler)

    # 2. File Handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            handlers.append(file_handler)
        except Exception as e:
            print(f"⚠️ Не удалось создать лог-файл {log_file}: {e}")

    # Базовая настройка
    # force=True нужен, чтобы переопределить конфигурацию, если она уже была создана
    logging.basicConfig(level=level, format=format_string, handlers=handlers, force=True)

    # Применить фильтр схем ко всем обработчикам
    schema_filter = IgnoreSchemaWarnings()
    for handler in logging.root.handlers:
        handler.addFilter(schema_filter)

    # Подавление шума от библиотек
    noisy_loggers = [
        "langchain_mcp_adapters",
        "mcp",
        "jsonschema",
        "langchain_google_genai",
        "httpcore",
        "httpx",
        "openai",
        "chromadb",     # <-- ДОБАВЛЕНО: подавление логов ChromaDB
        "hnswlib"
    ]
    
    # Если мы в DEBUG режиме, возможно мы хотим видеть часть этого, 
    # но обычно библиотеки слишком болтливы.
    lib_level = logging.ERROR if level > logging.DEBUG else logging.WARNING
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(lib_level)

    return logging.getLogger(__name__)
