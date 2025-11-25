import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

class IgnoreSchemaWarnings(logging.Filter):
    """Фильтр для подавления специфических предупреждений схем Pydantic/LangChain."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "Key 'additionalProperties' is not supported" not in msg and \
               "Key '$schema' is not supported" not in msg

class MaxLevelFilter(logging.Filter):
    """Пропускает сообщения только до определенного уровня (для разделения stdout/stderr)."""
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level

def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    
    # 1. Определение уровня логирования
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    # 2. Определение пути к файлу
    if log_file is None:
        log_file = os.getenv("LOG_FILE", "ai_agent.log")

    handlers: List[logging.Handler] = []
    formatter = logging.Formatter(format_string)

    # --- H1: STDOUT (INFO и ниже, без ошибок) ---
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG) # Принимаем всё, фильтр отсечет лишнее
    stdout_handler.addFilter(MaxLevelFilter(logging.WARNING))
    stdout_handler.setFormatter(formatter)
    handlers.append(stdout_handler)

    # --- H2: STDERR (ERROR и CRITICAL) ---
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)
    handlers.append(stderr_handler)

    # --- H3: FILE (Всё подряд) ---
    if log_file:
        try:
            log_path = Path(log_file)
            # Создаем директорию логов, если её нет
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG) # В файл пишем всё
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except Exception as e:
            # Пишем в stderr, так как логгер еще не настроен
            sys.stderr.write(f"⚠️ Не удалось создать лог-файл: {e}\n")

    # Настройка корневого логгера
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Применение фильтров
    schema_filter = IgnoreSchemaWarnings()
    for h in handlers:
        h.addFilter(schema_filter)

    # Подавление шума сторонних библиотек
    noisy_modules = [
        "langchain_mcp_adapters", "mcp", "jsonschema", "langchain_google_genai",
        "httpcore", "httpx", "openai", "chromadb", "hnswlib", 
        "google.ai.generativelanguage", "urllib3", "multipart"
    ]
    
    # Если общий уровень DEBUG, библиотеки ставим в ERROR, иначе WARNING
    lib_level = logging.ERROR if level > logging.DEBUG else logging.WARNING
    
    for module_name in noisy_modules:
        logging.getLogger(module_name).setLevel(lib_level)

    return logging.getLogger("AgentCore")