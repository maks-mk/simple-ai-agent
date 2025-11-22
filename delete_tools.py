import shutil
import asyncio
import logging
from pathlib import Path
from typing import Type, Optional, Tuple
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ==========================================
# 1. Базовый класс (Логика безопасности)
# ==========================================

class FileSystemTool(BaseTool):
    """
    Базовый абстрактный класс для инструментов файловой системы.
    Обеспечивает безопасность путей (sandbox), чтобы агент не удалил системные файлы.
    """
    
    # root_dir исключаем из схемы (exclude=True), чтобы LLM не пыталась его заполнить.
    root_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    def _validate_path(self, relative_path: str) -> Tuple[bool, Optional[str], Optional[Path]]:
        """
        Проверяет, находится ли путь внутри разрешенной директории (root_dir).
        Защищает от попыток выхода через '../'
        """
        try:
            # Превращаем относительный путь в абсолютный, опираясь на root_dir
            target_path = (self.root_dir / relative_path).resolve()
            resolved_root = self.root_dir.resolve()

            # Проверка безопасности: итоговый путь ОБЯЗАН начинаться с root_dir
            if not str(target_path).startswith(str(resolved_root)):
                msg = f"ACCESS DENIED: Путь '{relative_path}' выходит за пределы рабочей директории."
                logger.warning(f"{self.name}: {msg}")
                return False, msg, None

            return True, None, target_path

        except Exception as e:
            msg = f"Path Error: {str(e)}"
            logger.error(f"{self.name}: {msg}")
            return False, msg, None

    def _format_error(self, msg: str) -> str:
        return f"Ошибка: {msg}"

    def _format_success(self, msg: str) -> str:
        return f"Успешно: {msg}"

# ==========================================
# 2. Инструменты удаления
# ==========================================

class DeleteFileInput(BaseModel):
    """Входные параметры для удаления файла"""
    file_path: str = Field(description="Относительный путь к файлу для удаления")

class SafeDeleteFileTool(FileSystemTool):
    """Безопасный инструмент для удаления файлов"""
    name: str = "safe_delete_file"
    description: str = "Безопасно удаляет файл внутри рабочей директории."
    args_schema: Type[BaseModel] = DeleteFileInput

    def _run(self, file_path: str) -> str:
        try:
            is_valid, error, target = self._validate_path(file_path)
            if not is_valid: return self._format_error(error)

            if not target.exists():
                return self._format_error(f"Файл не найден: {file_path}")
            if target.is_dir():
                return self._format_error(f"{file_path} - это директория. Используйте safe_delete_directory.")

            target.unlink()
            logger.info(f"FILE DELETED: {target}")
            return self._format_success(f"Файл {file_path} удален.")
        except Exception as e:
            logger.error(f"Delete File Error ({file_path}): {e}")
            return self._format_error(str(e))

    async def _arun(self, file_path: str) -> str:
        return await asyncio.to_thread(self._run, file_path)


class DeleteDirectoryInput(BaseModel):
    """Входные параметры для удаления директории"""
    dir_path: str = Field(description="Относительный путь к директории")
    recursive: bool = Field(default=False, description="Удалить рекурсивно (со всем содержимым)")

class SafeDeleteDirectoryTool(FileSystemTool):
    """Безопасный инструмент для удаления директорий"""
    name: str = "safe_delete_directory"
    description: str = "Безопасно удаляет директорию. Используйте recursive=True для непустых папок."
    args_schema: Type[BaseModel] = DeleteDirectoryInput

    def _run(self, dir_path: str, recursive: bool = False) -> str:
        try:
            is_valid, error, target = self._validate_path(dir_path)
            if not is_valid: return self._format_error(error)

            if not target.exists():
                return self._format_error(f"Директория не найдена: {dir_path}")
            if not target.is_dir():
                return self._format_error(f"{dir_path} - это файл. Используйте safe_delete_file.")

            if recursive:
                shutil.rmtree(str(target))
                logger.info(f"DIR DELETED (Recursive): {target}")
                return self._format_success(f"Директория {dir_path} удалена рекурсивно.")
            else:
                target.rmdir()
                logger.info(f"DIR DELETED: {target}")
                return self._format_success(f"Пустая директория {dir_path} удалена.")

        except OSError as e:
            if "not empty" in str(e) or "WinError 145" in str(e):
                return self._format_error(f"Папка не пуста. Установите recursive=True.")
            logger.error(f"Delete Dir OSError ({dir_path}): {e}")
            return self._format_error(str(e))
        except Exception as e:
            logger.error(f"Delete Dir Error ({dir_path}): {e}")
            return self._format_error(str(e))

    async def _arun(self, dir_path: str, recursive: bool = False) -> str:
        return await asyncio.to_thread(self._run, dir_path, recursive)
