import re
import time
from pathlib import Path
from typing import List, Any, Optional, Dict

class ToolSanitizer:
    """
    Модуль для очистки и нормализации аргументов инструментов.
    """

    @staticmethod
    def sanitize_tool_calls(tool_calls: List[dict]):
        for tc in tool_calls:
            if "name" in tc:
                tc["name"] = ToolSanitizer._clean_tool_name(tc["name"])
            if "args" in tc:
                tc["args"] = ToolSanitizer._walk_and_clean(tc["args"])

    @staticmethod
    def _clean_tool_name(name: str) -> str:
        if not isinstance(name, str): return str(name)
        if "<|" in name: name = name.split("<|")[0]
        name = re.sub(r'[^\w\-\.]', '', name)
        return name.strip()

    @classmethod
    def _walk_and_clean(cls, data: Any, key_hint: Optional[str] = None) -> Any:
        if isinstance(data, dict):
            return {k: cls._walk_and_clean(v, key_hint=k) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls._walk_and_clean(item, key_hint=key_hint) for item in data]
        elif isinstance(data, str):
            return cls._sanitize_scalar_string(data, key_hint)
        return data

    @staticmethod
    def _sanitize_scalar_string(value: str, key_hint: Optional[str]) -> str:
        clean_v = value.strip().strip("'").strip('"')

        if not key_hint: return clean_v
        key = key_hint.lower()

        # --- СТРАТЕГИЯ 1: ФАЙЛОВЫЕ ПУТИ ---
        if any(k in key for k in ["path", "file", "dir", "dest", "src", "output"]):
            # Удаляем запрещенные символы, НО разрешаем точку и слэши
            clean_v = re.sub(r'[<>:"|?*]+', '', clean_v).strip()
            
            # [FIX] Разрешаем пустой путь или точку (CWD)
            # Это критично для list_directory, ls, cd
            if not clean_v or clean_v == ".":
                return "."

            path_obj = Path(clean_v)
            if path_obj.is_absolute():
                return str(path_obj)
                
            parts = [p for p in path_obj.parts if p not in (".", "..", "\\", "/")]
            
            # Если после чистки ничего не осталось, возвращаем текущую директорию
            if not parts:
                return "."
            
            return str(Path(*parts))

        # --- СТРАТЕГИЯ 2: URL ---
        if any(k in key for k in ["url", "link", "site", "href"]):
            match = re.search(r'(https?://[^\s\'"<>\[\]{}]+)', value)
            if match: return match.group(1)
            return clean_v

        # --- СТРАТЕГИЯ 3: ЧИСЛА ---
        if any(k in key for k in ["count", "limit", "amount", "id"]):
            digits = re.sub(r'\D', '', clean_v)
            if digits and len(digits) == len(clean_v):
                return clean_v 
        
        return clean_v