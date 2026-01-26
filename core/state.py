from typing import TypedDict, Annotated, Optional, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Состояние графа агента.
    """
    # История сообщений (автоматически склеивается через add_messages)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Сжатая память (саммари)
    summary: str
    
    # Счетчик шагов (для защиты от бесконечных циклов)
    steps: int
    
    # --- validation support ---
    # Последний вызов инструмента для валидации
    last_tool_call: Optional[dict]
    
    # Счетчик попыток исправления ошибок для каждого инструмента
    tool_retries: dict
    
    # Список имен инструментов, разрешенных на этом шаге
    # Если None — разрешены все.
    allowed_tools: Optional[List[str]]