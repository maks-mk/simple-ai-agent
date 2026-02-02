import logging
import os
from typing import List, Set, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from core import constants

logger = logging.getLogger("safety_guard")

class SafetyGuard:
    """
    Модуль политик безопасности (Guardrails).
    Решает, разрешено ли агенту выполнять опасные действия.
    """
    
    # Ключевые слова для определения типа действий
    DESTRUCTIVE_ROOTS: Set[str] = constants.DESTRUCTIVE_ROOTS
    WRITE_ROOTS: Set[str] = constants.WRITE_ROOTS
    
    # Слова, указывающие на творческую задачу (разрешают генерацию без поиска)
    CREATIVE_TRIGGERS: Set[str] = constants.CREATIVE_TRIGGERS

    # Инструменты, которые считаются источниками знаний
    RETRIEVAL_WHITELIST: Set[str] = constants.RETRIEVAL_WHITELIST

    # Инструменты, которые мы игнорируем при поиске знаний
    MODIFICATION_BLACKLIST: Set[str] = constants.MODIFICATION_BLACKLIST

    @classmethod
    def is_unsafe_write(cls, response: AIMessage, history: List[BaseMessage]) -> bool:
        """
        Возвращает True, если действие записи файла считается небезопасным
        (нет источников данных и не похоже на творчество).
        """
        # 0. Глобальный выключатель
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
        # Деструктивные действия требуют обоснования всегда
        if not is_destructive:
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
                
                if any(bad in t_name for bad in cls.MODIFICATION_BLACKLIST):
                    continue

                if any(good in t_name for good in cls.RETRIEVAL_WHITELIST):
                    has_data = True
                    break
                    
        if not has_data:
            logger.warning("🛡️ SafetyGuard: Blocked write action (no data source found).")
            return True
            
        return False
