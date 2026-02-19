import sys
from pathlib import Path

# Определение корневой директории проекта
if getattr(sys, 'frozen', False):
    # Если запущено как exe, используем директорию исполняемого файла для конфигов,
    # но рабочей директорией оставим текущую (cwd), откуда запущен процесс.
    BASE_DIR = Path(sys.executable).parent
else:
    # core/constants.py -> core/ -> root/
    BASE_DIR = Path(__file__).resolve().parent.parent

# --- PROMPTS ---

SUMMARY_PROMPT_TEMPLATE = (
    "Current memory context:\n<previous_context>\n{summary}\n</previous_context>\n\n"
    "New events:\n{history_text}\n\n"
    "Update <previous_context>. This is a technical log of a software development session. "
    "Keep only key facts, decisions, and results. "
    "Remove chit-chat. Return only the updated context text."
)

REFLECTION_PROMPT = (
    "SYSTEM HINT: The previous tool execution failed. "
    "Review the error, correct your arguments, or use a different tool. "
    "Act immediately."
)

