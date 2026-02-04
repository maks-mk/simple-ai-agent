from typing import Optional, TypedDict, Any
from langchain_core.messages import ToolMessage
from core.constants import MISSING_RESOURCE_MARKERS

class ValidationResult(TypedDict):
    is_valid: bool
    error_message: Optional[str]
    retry_needed: bool

def _is_missing_resource_error(content: str) -> bool:
    content_lower = content.lower()
    return any(m in content_lower for m in MISSING_RESOURCE_MARKERS)

def validate_tool_execution(
    tool_msg: ToolMessage,
    tool_args: dict,
    tool_name: str
) -> ValidationResult:
    """
    Универсальная валидация результата выполнения инструмента.
    Определяет, нужно ли агенту повторить попытку или остановиться.
    """
    content = str(tool_msg.content)
    is_missing = _is_missing_resource_error(content)

    # 1. Проверка нативного статуса LangChain
    if getattr(tool_msg, "status", "") == "error":
        if is_missing:
            return {
                "is_valid": False,
                "error_message": f"Resource missing (System Error): {content}",
                "retry_needed": False 
            }
        return {
            "is_valid": False,
            "error_message": f"Tool '{tool_name}' crashed: {content}",
            "retry_needed": True
        }

    # 2. Проверка структурированной ошибки (Artifacts)
    artifact = getattr(tool_msg, "artifact", None)
    if isinstance(artifact, dict) and artifact.get("is_error"):
        return {
            "is_valid": False,
            "error_message": f"Tool '{tool_name}' logic error: {content}",
            "retry_needed": True
        }

    # 3. Проверка текстовых ошибок (MCP и CLI)
    if content.strip().startswith(("Error:", "Ошибка:", "MCP error")):
        if is_missing:
            return {
                "is_valid": False,
                "error_message": f"Resource missing: {content}",
                "retry_needed": False 
            }
        return {
            "is_valid": False,
            "error_message": f"Tool execution returned error text: {content}",
            "retry_needed": True
        }

    # 4. Проверка системных сообщений (Config / Access)
    if not artifact:
        if content.startswith("System Config Error:"):
             return {
                "is_valid": False,
                "error_message": f"Configuration error in '{tool_name}'. Do not retry without changing args.",
                "retry_needed": False
            }
        if "ACCESS DENIED" in content or "access denied" in content.lower():
            return {
                "is_valid": False,
                "error_message": f"Permission denied for '{tool_name}'. Do not retry.",
                "retry_needed": False
            }
        
        # Повторная проверка на missing resource для plain text
        if is_missing:
             return {
                "is_valid": False,
                "error_message": f"Resource missing: {content}",
                "retry_needed": False 
            }

    # 5. Проверка на пустоту
    if not content.strip() and not artifact:
        return {
            "is_valid": False,
            "error_message": f"Tool '{tool_name}' returned empty result. Check arguments.",
            "retry_needed": True
        }

    # Все проверки пройдены
    return {"is_valid": True, "error_message": None, "retry_needed": False}
