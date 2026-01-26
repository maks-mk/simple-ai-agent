from typing import Optional, TypedDict, Any
from langchain_core.messages import ToolMessage

class ValidationResult(TypedDict):
    is_valid: bool
    error_message: Optional[str]
    retry_needed: bool

def validate_tool_execution(
    tool_msg: ToolMessage,
    tool_args: dict,
    tool_name: str
) -> ValidationResult:
    """
    Универсальная валидация результата выполнения инструмента.
    """
    content = str(tool_msg.content)
    lower_content = content.lower()
    
    # Список фраз, означающих отсутствие ресурса (нет смысла повторять)
    MISSING_RESOURCE_MARKERS = [
        "no such file", 
        "not found", 
        "enoent",       # Node.js Error No Entry
        "does not exist"
    ]
    
    is_missing_resource = any(m in lower_content for m in MISSING_RESOURCE_MARKERS)

    # 1. Проверка нативного статуса LangChain (исключения)
    if getattr(tool_msg, "status", "") == "error":
        # [FIX] Даже если инструмент упал, проверяем, не является ли это ошибкой "Файл не найден"
        if is_missing_resource:
             return {
                "is_valid": False,
                "error_message": f"Resource missing (System Error): {content}",
                "retry_needed": False  # <--- ВАЖНО: Не повторять, файла нет
            }
            
        return {
            "is_valid": False,
            "error_message": f"Tool '{tool_name}' crashed: {content}",
            "retry_needed": True
        }

    artifact = tool_msg.artifact if hasattr(tool_msg, "artifact") else None

    # 2. Проверка структурированной ошибки (через Artifacts)
    if isinstance(artifact, dict) and artifact.get("is_error"):
        return {
            "is_valid": False,
            "error_message": f"Tool '{tool_name}' logic error: {content}",
            "retry_needed": True
        }

    # 3. Проверка текстовых ошибок (MCP и другие)
    if content.strip().startswith(("Error:", "Ошибка:", "MCP error")):
        # [FIX] Проверяем текст ошибки на отсутствие файла
        if is_missing_resource:
            return {
                "is_valid": False,
                "error_message": f"Resource missing: {content}",
                "retry_needed": False # <--- ВАЖНО: Не повторять
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
        if "ACCESS DENIED" in content or "access denied" in lower_content:
            return {
                "is_valid": False,
                "error_message": f"Permission denied for '{tool_name}'. Do not retry.",
                "retry_needed": False
            }
            
        # Повторная проверка на всякий случай (для простого текста)
        if is_missing_resource:
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