import logging
from typing import List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage

# Настраиваем логгер для этого модуля
logger = logging.getLogger("safety_guard")

class SafetyGuard:
    """
    Модуль политик безопасности (Guardrails).
    Решает, разрешено ли агенту выполнять опасные действия.
    """
    
    # Глобальный переключатель для экспериментов
    # Поставьте False, чтобы отключить блокировщик полностью
    ENABLED = True 

    # Ключевые слова для определения типа действий
    WRITE_ROOTS = {'write', 'save', 'append', 'edit', 'delete', 'store'}
    
    # Слова, указывающие на творческую задачу (разрешают генерацию без поиска)
    CREATIVE_TRIGGERS = {
        "script", "code", "python", "bash", "sh", "js", 
        "story", "poem", "essay", "joke", 
        "guide", "tutorial", "instruction", "example",
        "draft", "template", "boilerplate"
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
        # 0. Если защита отключена глобально
        if not cls.ENABLED:
            return False

        if not response.tool_calls:
            return False
        
        # 1. Проверяем, пытается ли агент что-то записать/изменить
        is_writing = False
        for tc in response.tool_calls:
            t_name = tc['name'].lower()
            if any(root in t_name for root in cls.WRITE_ROOTS):
                is_writing = True
                break
        
        if not is_writing:
            return False

        # 2. Проверяем Bypass для творчества (Creative Intent)
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