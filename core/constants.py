import sys
from pathlib import Path

# Определение корневой директории проекта
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    # core/constants.py -> core/ -> root/
    BASE_DIR = Path(__file__).resolve().parent.parent

# --- PROMPTS ---

SUMMARY_PROMPT_TEMPLATE = (
    "Current memory context:\n<previous_context>\n{summary}\n</previous_context>\n\n"
    "New events:\n{history_text}\n\n"
    "Update <previous_context>. Keep only key facts, decisions, and results. "
    "Remove chit-chat. Return only the updated context text."
)

INTENT_CLASSIFICATION_SYSTEM_PROMPT = (
    "Analyze the conversation and determine the user's intent. "
    "Return a JSON object with 'intent' and 'reasoning'. "
    "If the user wants to create, edit, save, delete, or modify files/system -> 'write_action'. "
    "If the user just wants to search, read, or ask questions -> 'read_only'."
)

# --- KEYWORDS & TRIGGERS ---

INTENT_HINTS = (
    "создай", "создать", "запиши", "записать", "сохрани",
    "сделай", "сделать", "напиши", "написать", "измени",
    "добавь", "добавить", "обнови", "обновить", "исправь", "почини",
    "create", "write", "save", "generate", "edit", "update", "delete",
    "add", "insert", "modify", "fix", "replace", "patch"
)

# Safety Guard Lists
DESTRUCTIVE_ROOTS = {'delete', 'remove', 'unlink', 'rmdir', 'format'}
WRITE_ROOTS = {'write', 'save', 'append', 'edit', 'store', 'update', 'replace', 'move', 'create', 'mkdir', 'put', 'post', 'send', 'upload'} | DESTRUCTIVE_ROOTS

CREATIVE_TRIGGERS = {
    "script", "story", "poem", "essay", "joke", 
    "guide", "tutorial", "instruction", "example",
    "draft", "template", "boilerplate",
    "write a python script", "create a bash script",
    # Русские триггеры
    "скрипт", "код", "программу", "стих", "истори", "сказк", 
    "пример", "инструкци", "гайд", "черновик", "шаблон",
    "напиши", "создай", "сгенерируй"
}

RETRIEVAL_WHITELIST = {
    'search', 'read', 'fetch', 'get', 'query', 
    'load', 'list', 'retrieve', 'browse', 'ask', 'lookup',
    'deep_search'
}

MODIFICATION_BLACKLIST = {
    'write', 'save', 'edit', 'append', 'delete', 
    'remove', 'update', 'put', 'post', 'send', 'upload'
}

FATAL_ERRORS = ["401", "unauthorized", "quota", "billing", "context_length_exceeded"]

# --- SYSTEM & ERROR HANDLING ---

MISSING_RESOURCE_MARKERS = [
    "no such file", 
    "not found", 
    "enoent",       # Node.js Error No Entry
    "does not exist",
    "cannot find the path", # Windows specific
    "directory not found",
    "unable to open"
]

REFLECTION_PROMPT = (
    "REFLECTION:\n"
    "- The previous action failed.\n"
    "- Identify WHY it failed (invalid args, missing data, wrong tool).\n"
    "- DO NOT repeat the same tool with the same arguments.\n"
    "- Change strategy (e.g. use a different tool or gather data first).\n"
    "Reply with a brief plan, then act."
)

