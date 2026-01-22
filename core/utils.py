import json
import logging
from typing import List
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger(__name__)

class AgentUtils:
    """Вспомогательные функции для токенов."""
    
    def __init__(self):
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base") if tiktoken else None
        except Exception:
            self._encoder = None

    def count_tokens(self, text: str) -> int:
        if not text: return 0
        if self._encoder:
            return len(self._encoder.encode(text))
        return len(text) // 3

    def estimate_payload_tokens(self, messages: List[BaseMessage], tools: List[BaseTool]) -> int:
        total = 0
        for m in messages:
            content = m.content if isinstance(m.content, str) else ""
            if isinstance(m.content, list):
                content = " ".join([str(x) for x in m.content])
            total += self.count_tokens(content)
        
        if tools:
            try:
                tool_schemas = [convert_to_openai_tool(t) for t in tools]
                tools_json = json.dumps(tool_schemas, ensure_ascii=False)
                total += self.count_tokens(tools_json)
            except Exception:
                pass
        return total