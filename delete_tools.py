import asyncio
import logging
import shutil
import errno
from pathlib import Path
from typing import Type, Tuple, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class FileSystemTool(BaseTool):
    """
    Базовый класс для безопасной работы с файловой системой.
    """
    root_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    def _validate_path(self, relative_path: str) -> Tuple[bool, Optional[str], Optional[Path]]:
        try:
            # 1. Проверка на абсолютный путь
            if Path(relative_path).is_absolute():
                msg = f"ACCESS DENIED: Абсолютные пути не разрешены: '{relative_path}'"
                logger.warning(f"{self.name}: {msg}")
                return False, msg, None

            # 2. Разрешаем путь относительно root_dir
            target_path = (self.root_dir / relative_path).resolve()
            root_resolved = self.root_dir.resolve()

            # 3. Проверка на выход за пределы (Path Traversal)
            if not target_path.is_relative_to(root_resolved):
                msg = f"ACCESS DENIED: Путь '{relative_path}' выходит за пределы рабочей директории."
                logger.warning(f"{self.name}: {msg}")
                return False, msg, None

            return True, None, target_path
        except Exception as e:
            return False, f"Path Error: {e}", None

    def _format_error(self, msg: str) -> str:
        return f"Ошибка: {msg}"

    def _format_success(self, msg: str) -> str:
        return f"Успешно: {msg}"


class DeleteFileInput(BaseModel):
    file_path: str = Field(description="Relative path to the file")

class SafeDeleteFileTool(FileSystemTool):
    name: str = "safe_delete_file"
    description: str = "Safely deletes a file within the working directory."
    args_schema: Type[BaseModel] = DeleteFileInput

    def _run(self, file_path: str) -> str:
        is_valid, error, target = self._validate_path(file_path)
        if not is_valid: 
            return self._format_error(error)

        if not target.exists():
            return self._format_error(f"Файл не найден: {file_path}")
        if target.is_dir():
            return self._format_error(f"{file_path} - это директория. Используйте safe_delete_directory.")

        try:
            target.unlink()
            logger.info(f"FILE DELETED: {target}")
            return self._format_success(f"Файл {file_path} удален.")
        except Exception as e:
            logger.error(f"Delete File Error: {e}")
            return self._format_error(str(e))

    async def _arun(self, file_path: str) -> str:
        return await asyncio.to_thread(self._run, file_path)


class DeleteDirectoryInput(BaseModel):
    dir_path: str = Field(description="Relative path to the directory")
    recursive: bool = Field(default=False, description="Delete recursively (for non-empty dirs)")

class SafeDeleteDirectoryTool(FileSystemTool):
    name: str = "safe_delete_directory"
    description: str = "Safely deletes a directory. Use recursive=True for non-empty folders."
    args_schema: Type[BaseModel] = DeleteDirectoryInput

    def _run(self, dir_path: str, recursive: bool = False) -> str:
        is_valid, error, target = self._validate_path(dir_path)
        if not is_valid: 
            return self._format_error(error)

        if not target.exists():
            return self._format_error(f"Директория не найдена: {dir_path}")
        if not target.is_dir():
            return self._format_error(f"{dir_path} - это файл.")

        try:
            if recursive:
                shutil.rmtree(target)
                msg = f"Директория {dir_path} удалена рекурсивно."
            else:
                target.rmdir()
                msg = f"Пустая директория {dir_path} удалена."
            
            logger.info(f"DIR DELETED: {target}")
            return self._format_success(msg)

        except OSError as e:
            if e.errno == errno.ENOTEMPTY:
                return self._format_error("Папка не пуста. Установите recursive=True.")
            return self._format_error(str(e))
        except Exception as e:
            logger.error(f"Delete Dir Error: {e}")
            return self._format_error(str(e))

    async def _arun(self, dir_path: str, recursive: bool = False) -> str:
        return await asyncio.to_thread(self._run, dir_path, recursive)