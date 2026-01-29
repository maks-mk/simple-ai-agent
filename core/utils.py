import json
import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

logger = logging.getLogger(__name__)

class AgentUtils:
    """
    Вспомогательные утилиты для работы с токенами и подсчета стоимости.
    """
    
    def __init__(self):
        self._encoder = None
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("Tiktoken not installed. Token counting will be approximate.")
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken: {e}")

    def count_tokens(self, text: str) -> int:
        """
        Считает количество токенов в тексте.
        Если tiktoken недоступен, использует эвристику (len/3).
        """
        if not text:
            return 0
        if self._encoder:
            return len(self._encoder.encode(text))
        return len(text) // 3

    def estimate_payload_tokens(self, messages: List[BaseMessage], tools: List[BaseTool]) -> int:
        """
        Оценивает общее количество токенов в сообщениях и определениях инструментов.
        """
        total = 0
        for m in messages:
            content = m.content if isinstance(m.content, str) else ""
            if isinstance(m.content, list):
                # Обработка мультимодального контента (text + image)
                content = " ".join([str(x) for x in m.content])
            total += self.count_tokens(content)
        
        if tools:
            try:
                tool_schemas = [convert_to_openai_tool(t) for t in tools]
                tools_json = json.dumps(tool_schemas, ensure_ascii=False)
                total += self.count_tokens(tools_json)
            except Exception as e:
                logger.debug(f"Error estimating tool tokens: {e}")
        
        return total
