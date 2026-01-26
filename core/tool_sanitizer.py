import re
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
            
            # 1. Попытка нормализации через pathlib (обработка абсолютных путей)
            try:
                path_obj = Path(clean_v)
                
                # Если путь абсолютный (например, получен от list_allowed_directories или системного вызова)
                if path_obj.is_absolute():
                    try:
                        # Если файл внутри текущей рабочей директории -> делаем относительным
                        # Это превращает "C:\Users\Project\README.md" -> "README.md"
                        return str(path_obj.relative_to(Path.cwd()))
                    except ValueError:
                        # Путь абсолютный, но находится ВНЕ текущей папки (или на другом диске).
                        # Мы НЕ должны ломать его (удалять двоеточие), но должны почистить от мусора.
                        pass 
            except Exception:
                pass

            # 2. Очистка строки регулярными выражениями
            # Сценарий A: Windows Absolute Path (начинается с буквы диска, например "C:\")
            if re.match(r'^[a-zA-Z]:[\\/]', clean_v):
                # Сохраняем "C:", чистим остальное
                drive = clean_v[:2]
                rest = clean_v[2:]
                # В хвосте пути двоеточия недопустимы (кроме потоков NTFS, которые мы режем)
                rest_clean = re.sub(r'[<>:"|?*]+', '', rest)
                clean_v = drive + rest_clean
                
            # Сценарий B: Относительный путь или Unix
            else:
                # Удаляем все запрещенные символы, включая двоеточие
                clean_v = re.sub(r'[<>:"|?*]+', '', clean_v)

            clean_v = clean_v.strip()

            # 3. Финальная защита от Path Traversal (для относительных путей)
            # Разрешаем пустой путь или точку (CWD)
            if not clean_v or clean_v == ".":
                return "."
            
            # Если путь остался относительным, убираем попытки выхода ".."
            # (Если путь абсолютный Windows, мы доверяем букве диска, обработанной выше)
            if not re.match(r'^[a-zA-Z]:', clean_v) and not clean_v.startswith(("/", "\\")):
                 path_obj = Path(clean_v)
                 parts = [p for p in path_obj.parts if p not in (".", "..", "\\", "/")]
                 if not parts:
                     return "."
                 return str(Path(*parts))

            return clean_v

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