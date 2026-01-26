import logging
import os
from typing import List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage

# [NEW] Импортируем конфиг для доступа к переменным .env
# from core.config import AgentConfig  <-- Removed to decouple

# Настраиваем логгер для этого модуля
logger = logging.getLogger("safety_guard")

class SafetyGuard:
    """
    Модуль политик безопасности (Guardrails).
    Решает, разрешено ли агенту выполнять опасные действия.
    Управляется переменной SAFETY_GUARD_ENABLED в .env.
    """
    
    # [NEW] Загружаем конфигурацию один раз при инициализации класса
    # try:
    #     _config = AgentConfig()
    # except Exception as e:
    #     logger.warning(f"SafetyGuard config load failed ({e}). Defaulting to ENABLED.")
    #     # Fallback: если конфиг сломан, защита включена по умолчанию (Safety First)
    #     class MockConfig:
    #         safety_guard_enabled = True
    #     _config = MockConfig()

    # Ключевые слова для определения типа действий
    # Разделяем на просто запись и деструктивные действия
    DESTRUCTIVE_ROOTS = {'delete', 'remove', 'unlink', 'rmdir', 'format'}
    WRITE_ROOTS = {'write', 'save', 'append', 'edit', 'store', 'update', 'replace', 'move', 'create', 'mkdir', 'put', 'post', 'send', 'upload'} | DESTRUCTIVE_ROOTS
    
    # Слова, указывающие на творческую задачу (разрешают генерацию без поиска)
    # Исключаем слова, которые могут быть двусмысленными (например, "code" можно интерпретировать как "code deletion")
    CREATIVE_TRIGGERS = {
        "script", "story", "poem", "essay", "joke", 
        "guide", "tutorial", "instruction", "example",
        "draft", "template", "boilerplate",
        "write a python script", "create a bash script", # Более специфичные фразы
        # Русские триггеры
        "скрипт", "код", "программу", "стих", "истори", "сказк", 
        "пример", "инструкци", "гайд", "черновик", "шаблон",
        "напиши", "создай", "сгенерируй"
    }

    # Инструменты, которые считаются источниками знаний
    RETRIEVAL_WHITELIST = {
        'search', 'read', 'fetch', 'get', 'query', 
        'load', 'list', 'retrieve', 'browse', 'ask', 'lookup',
        'deep_search'
    }

    # Инструменты, которые мы игнорируем при поиске знаний
    MODIFICATION_BLACKLIST = {
        'write', 'save', 'edit', 'append', 'delete', 
        'remove', 'update', 'put', 'post', 'send', 'upload'
    }

    @classmethod
    def is_unsafe_write(cls, response: AIMessage, history: List[BaseMessage]) -> bool:
        """
        Возвращает True, если действие записи файла считается небезопасным
        (нет источников данных и не похоже на творчество).
        """
        # 0. [MODIFIED] Проверка глобального переключателя из .env
        if os.getenv("SAFETY_GUARD_ENABLED", "True").lower() == "false":
            return False

        if not response.tool_calls:
            return False
        
        # 1. Проверяем, пытается ли агент что-то записать/изменить
        is_writing = False
        is_destructive = False
        
        for tc in response.tool_calls:
            t_name = tc['name'].lower()
            if any(root in t_name for root in cls.DESTRUCTIVE_ROOTS):
                is_destructive = True
                is_writing = True
                break
            if any(root in t_name for root in cls.WRITE_ROOTS):
                is_writing = True
        
        if not is_writing:
            return False

        # 2. Проверяем Bypass для творчества (Creative Intent)
        # Если действие деструктивное, творческий bypass НЕ РАБОТАЕТ (удаление требует обоснования)
        if not is_destructive:
            # Ищем последнее сообщение пользователя
            last_human = next((m for m in reversed(history) if isinstance(m, HumanMessage)), None)
            
            if last_human:
                text = last_human.content.lower()
                if any(trigger in text for trigger in cls.CREATIVE_TRIGGERS):
                    logger.info(f"🛡️ SafetyGuard: Bypass allowed (Creative intent detected in '{text[:20]}...')")
                    return False

        # 3. Анализ истории: ищем доказательства знаний (Grounding)
        has_data = False
        
        for m in history:
            if isinstance(m, ToolMessage):
                content = str(m.content)
                # Игнорируем ошибки и короткие отписки
                if len(content) < 20 or content.startswith(("System:", "Error:")):
                    continue

                t_name = m.name.lower()
                
                # Если это тул модификации - пропускаем
                if any(bad in t_name for bad in cls.MODIFICATION_BLACKLIST):
                    continue

                # Если это тул чтения - ура, данные есть
                if any(good in t_name for good in cls.RETRIEVAL_WHITELIST):
                    has_data = True
                    break
                    
        if not has_data:
            logger.warning("🛡️ SafetyGuard: Blocked write action (no data source found).")
            return True
            
        return False