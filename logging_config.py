import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

# Импорт Rich для красивого вывода
from rich.logging import RichHandler
from rich.console import Console

class IgnoreSchemaWarnings(logging.Filter):
    """Фильтр для подавления специфических предупреждений."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "Key 'additionalProperties' is not supported" not in msg and \
               "Key '$schema' is not supported" not in msg

def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    format_string: str = "%(message)s" # Rich сам добавит время и уровень
) -> logging.Logger:
    
    # 1. Определение уровня (по умолчанию INFO)
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    # 2. Определение пути к файлу
    if log_file is None:
        log_file = os.getenv("LOG_FILE", "ai_agent.log")

    handlers: List[logging.Handler] = []

    # --- H1: КРАСИВЫЙ КОНСОЛЬНЫЙ ВЫВОД (RICH) ---
    # markup=True позволяет использовать [bold red]...[/] прямо в логах
    # rich_tracebacks=True делает красивые стектрейсы ошибок
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False, # Скрываем путь к файлу (line 123), чтобы не засорять вид
        show_time=True,
        omit_repeated_times=False
    )
    console_handler.setLevel(level)
    # Формат для Rich проще, так как он сам строит колонки
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(console_handler)

    # --- H2: ФАЙЛОВЫЙ ВЫВОД (КЛАССИЧЕСКИЙ) ---
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # В файл пишем подробно с датой
            file_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG) # В файл пишем всё
            file_handler.setFormatter(file_fmt)
            handlers.append(file_handler)
        except Exception as e:
            sys.stderr.write(f"⚠️ Не удалось создать лог-файл: {e}\n")

    # Настройка корневого логгера
    # Убираем существующие хендлеры, чтобы не было дублей при перезагрузке
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Применение фильтров
    schema_filter = IgnoreSchemaWarnings()
    for h in handlers:
        h.addFilter(schema_filter)

    # --- ПОДАВЛЕНИЕ ШУМА ---
    # Добавлен sentence_transformers, чтобы убрать технические логи загрузки PyTorch
    noisy_modules = [
        "langchain_mcp_adapters", "mcp", "jsonschema", "langchain_google_genai",
        "httpcore", "httpx", "openai", "chromadb", "hnswlib", 
        "google.ai.generativelanguage", "urllib3", "multipart",
        "sentence_transformers", "filelock" # <-- Новые добавления
    ]
    
    # Если общий уровень DEBUG, библиотеки ставим в ERROR, иначе WARNING
    lib_level = logging.ERROR if level > logging.DEBUG else logging.WARNING
    
    for module_name in noisy_modules:
        logging.getLogger(module_name).setLevel(lib_level)

    return logging.getLogger("AgentCore")