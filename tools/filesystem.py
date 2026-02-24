"""
Advanced Filesystem Tools Module.
Based on deepagents filesystem implementation.
Features:
- Virtual Mode (Security Sandbox)
- Unified Diff generation for edits
- Pagination for reading large files
- Safe Path Resolution
"""

import os
import difflib
import aiofiles
from pathlib import Path
from typing import Union, Optional
import logging
import httpx
from langchain_core.tools import tool
from tools.system_tools import get_net_client
from core.safety_policy import SafetyPolicy
from core.errors import format_error, ErrorType
from core.utils import truncate_output

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_READ_LIMIT = 2000

# Directories to skip during recursive search (noise / non-user code)
IGNORED_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".next", ".nuxt",
    "venv", ".venv", "env", ".env",
    "dist", "build", "out", "target",
    ".idea", ".vscode",
}

class FilesystemManager:
    """
    Manages filesystem operations with security checks.
    """
    def __init__(self, root_dir: Union[str, Path] = None, virtual_mode: bool = True):
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.safety_policy: Optional[SafetyPolicy] = None

    def set_policy(self, policy: SafetyPolicy):
        self.safety_policy = policy

    def _resolve_path(self, path_str: str) -> Path:
        """
        Resolves path with security checks against Path Traversal.
        """
        if not path_str:
            raise ValueError("Path cannot be empty")
            
        # 1. Normalize separators
        clean_path = str(path_str).replace("\\", "/")
        path_obj = Path(clean_path)
        
        # ✅ Исправлено: Явный блок абсолютных путей (защита Windows: C:/Windows)
        if self.virtual_mode and path_obj.is_absolute():
            raise ValueError(f"ACCESS DENIED: Absolute paths not allowed in virtual mode: {path_str}")
        
        # 2. Virtual Mode Checks (Strict Sandbox)
        if self.virtual_mode:
            if clean_path.startswith("/"):
                clean_path = clean_path.lstrip("/")
            
            # Block traversal sequences
            if ".." in clean_path or clean_path.startswith("~"):
                raise ValueError(f"ACCESS DENIED: Path traversal not allowed in virtual mode: {path_str}")
                
            full_path = (self.cwd / clean_path).resolve()
            
            # Final verify: must be inside root
            try:
                full_path.relative_to(self.cwd)
            except ValueError:
                raise ValueError(f"ACCESS DENIED: Path is outside working directory: {full_path}")
                
            return full_path

        # 3. Legacy Mode (Less Secure)
        if path_obj.is_absolute():
            return path_obj.resolve()
        return (self.cwd / path_obj).resolve()

    def read_file(self, path: str, offset: int = 0, limit: int = DEFAULT_READ_LIMIT) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            if not target.is_file(): return format_error(ErrorType.VALIDATION, f"'{path}' is not a file.")
            
            # Check size (skip huge files)
            stats = target.stat()
            max_size = self.safety_policy.max_file_size if self.safety_policy else 10 * 1024 * 1024
            if stats.st_size > max_size:
                return format_error(ErrorType.LIMIT_EXCEEDED, f"File is too large ({stats.st_size} bytes). Max: {max_size}.")

            try:
                content = target.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                return format_error(ErrorType.VALIDATION, "File binary or unknown encoding.")

            if not content:
                return "System reminder: File exists but has empty contents."

            lines = content.splitlines()
            total_lines = len(lines)
            
            # Policy limit for read lines
            policy_limit = self.safety_policy.max_read_lines if self.safety_policy else DEFAULT_READ_LIMIT
            if limit > policy_limit:
                limit = policy_limit

            # Pagination
            if offset >= total_lines:
                return format_error(ErrorType.VALIDATION, f"Line offset {offset} exceeds file length ({total_lines} lines).")
            
            end_index = min(offset + limit, total_lines)
            selected_lines = lines[offset:end_index]
            
            # Format with line numbers
            result = []
            for i, line in enumerate(selected_lines):
                result.append(f"{offset + i + 1:6}  {line}")
                
            output = "\n".join(result)
            
            # Add context info if truncated
            if total_lines > end_index:
                output += f"\n\n... (Showing lines {offset+1}-{end_index} of {total_lines}. Use offset/limit to read more)"
                
            return output

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error reading file: {e}")

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self._resolve_path(path)
            
            # Ensure parent exists
            target.parent.mkdir(parents=True, exist_ok=True)
            
            target.write_text(content, encoding='utf-8')
            return f"Success: File '{path}' saved ({len(content)} chars)."
        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error writing file: {e}")

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        """
        Exact string replacement with Unified Diff output.
        """
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            
            content = target.read_text(encoding='utf-8')
            
            # ✅ Исправлено: Нормализация и строгая проверка уникальности вхождений
            content_norm = content.replace('\r\n', '\n')
            old_string_norm = old_string.replace('\r\n', '\n')
            
            count = content_norm.count(old_string_norm)
            
            if count == 0:
                snippet = old_string_norm[:50].replace('\n', '\\n')
                return format_error(ErrorType.VALIDATION, f"Could not find exact target text starting with: '{snippet}...'")
            elif count > 1:
                return format_error(
                    ErrorType.VALIDATION, 
                    f"Found {count} identical occurrences of the target text. "
                    "Please provide more context (e.g., surrounding lines) to uniquely identify the block to replace."
                )

            # Perform safe replacement (exactly 1 occurrence)
            new_content = content_norm.replace(old_string_norm, new_string, 1)
            
            target.write_text(new_content, encoding='utf-8')
            
            # Generate Diff
            diff = difflib.unified_diff(
                content_norm.splitlines(),
                new_content.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm=""
            )
            diff_text = "\n".join(list(diff))
            
            return f"Success: File edited.\n\nDiff:\n```diff\n{diff_text}\n```"

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error editing file: {e}")

    def search_in_file(self, path: str, pattern: str, use_regex: bool = False, ignore_case: bool = False) -> str:
        import re
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            if not target.is_file(): return format_error(ErrorType.VALIDATION, f"'{path}' is not a file.")

            try:
                content = target.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                return format_error(ErrorType.VALIDATION, "File is binary or has unknown encoding.")

            flags = re.IGNORECASE if ignore_case else 0

            if use_regex:
                try:
                    compiled = re.compile(pattern, flags)
                except re.error as e:
                    return format_error(ErrorType.VALIDATION, f"Invalid regex pattern: {e}")
                matches = [
                    (i + 1, line)
                    for i, line in enumerate(content.splitlines())
                    if compiled.search(line)
                ]
            else:
                needle = pattern.lower() if ignore_case else pattern
                matches = [
                    (i + 1, line)
                    for i, line in enumerate(content.splitlines())
                    if needle in (line.lower() if ignore_case else line)
                ]

            if not matches:
                return f"No matches found for '{pattern}' in '{path}'."

            results = "\n".join(f"{lineno:6}  {line}" for lineno, line in matches)
            output = f"Found {len(matches)} match(es) in '{path}':\n\n{results}"
            limit = self.safety_policy.max_tool_output if self.safety_policy else 5000
            return truncate_output(output, limit, source="filesystem")

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error searching in file: {e}")

    def search_in_directory(self, path: str, pattern: str, use_regex: bool = False,
                            ignore_case: bool = False, extensions: Optional[str] = None,
                            max_matches: int = 500, max_files: int = 200,
                            max_depth: Optional[int] = None) -> str:
        import re
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"Path '{path}' not found.")
            if not target.is_dir(): return format_error(ErrorType.VALIDATION, f"'{path}' is not a directory.")

            ext_filter = None
            if extensions:
                ext_filter = {e.strip() if e.strip().startswith(".") else f".{e.strip()}"
                              for e in extensions.split(",")}

            flags = re.IGNORECASE if ignore_case else 0
            if use_regex:
                try:
                    compiled = re.compile(pattern, flags)
                    match_fn = lambda line: bool(compiled.search(line))
                except re.error as e:
                    return format_error(ErrorType.VALIDATION, f"Invalid regex pattern: {e}")
            else:
                needle = pattern.lower() if ignore_case else pattern
                match_fn = lambda line: needle in (line.lower() if ignore_case else line)

            all_results: list[str] = []
            files_scanned = 0
            dirs_skipped_ignored = 0
            files_skipped_binary = 0
            max_size = self.safety_policy.max_file_size if self.safety_policy else 10 * 1024 * 1024
            truncated = False

            # ✅ Исправлено: os.walk вместо rglob("*") для предотвращения Out of Memory
            for root, dirs, files in os.walk(target):
                # ── 🔴 In-place фильтрация директорий (не сканируем мусорные папки) ──
                original_dirs_count = len(dirs)
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
                dirs_skipped_ignored += (original_dirs_count - len(dirs))
                
                root_path = Path(root)
                try:
                    rel_to_target = root_path.relative_to(target)
                except ValueError:
                    rel_to_target = Path(".")

                # ── 🟡 Depth limit ──
                if max_depth is not None:
                    depth = len(rel_to_target.parts)
                    if depth >= max_depth:
                        dirs[:] = []  # Блокируем дальнейшее погружение
                    if depth > max_depth:
                        continue

                for file_name in files:
                    file = root_path / file_name

                    if ext_filter and file.suffix not in ext_filter:
                        continue
                    
                    try:
                        if file.stat().st_size > max_size:
                            continue
                    except OSError:
                        continue

                    # ── 🔴 File cap ──
                    if files_scanned >= max_files:
                        truncated = True
                        break

                    try:
                        lines_content = file.read_text(encoding='utf-8').splitlines()
                    except (UnicodeDecodeError, OSError):
                        files_skipped_binary += 1
                        continue

                    files_scanned += 1

                    try:
                        rel = file.relative_to(self.cwd)
                    except ValueError:
                        rel = file

                    for i, line in enumerate(lines_content):
                        if match_fn(line):
                            all_results.append(f"{rel}:{i + 1}  {line}")
                            # ── 🔴 Match cap ──
                            if len(all_results) >= max_matches:
                                truncated = True
                                break
                    
                    if truncated:
                        break
                
                if truncated:
                    break

            # ── Build output ──
            if not all_results:
                return (f"No matches found for '{pattern}' in '{path}'. "
                        f"Scanned {files_scanned} file(s), "
                        f"skipped {dirs_skipped_ignored} ignored dirs, "
                        f"{files_skipped_binary} binary files.")

            header_lines = [
                f"Found {len(all_results)} match(es) across {files_scanned} file(s).",
            ]
            if truncated:
                header_lines.append(
                    f"⚠ Results truncated (max_matches={max_matches}, max_files={max_files}). "
                    "Narrow your search or increase limits."
                )
            if dirs_skipped_ignored:
                header_lines.append(f"  Skipped {dirs_skipped_ignored} ignored directory branches.")

            output = "\n".join(header_lines) + "\n\n" + "\n".join(all_results)
            limit = self.safety_policy.max_tool_output if self.safety_policy else 5000
            return truncate_output(output, limit, source="filesystem")

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error searching in directory: {e}")

    def tail_file(self, path: str, lines: int = 50) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            if not target.is_file(): return format_error(ErrorType.VALIDATION, f"'{path}' is not a file.")

            policy_limit = self.safety_policy.max_read_lines if self.safety_policy else DEFAULT_READ_LIMIT
            lines = min(lines, policy_limit)

            collected: list[bytes] = []
            newlines_found = 0
            chunk_size = 8192

            with open(target, "rb") as f:
                f.seek(0, 2)
                file_size = f.tell()
                remaining = file_size

                while newlines_found <= lines and remaining > 0:
                    step = min(chunk_size, remaining)
                    remaining -= step
                    f.seek(remaining)
                    chunk = f.read(step)
                    collected.append(chunk)
                    newlines_found += chunk.count(b"\n")

            raw = b"".join(reversed(collected))

            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")

            all_lines = text.splitlines()

            if all_lines and all_lines[-1] == "":
                all_lines = all_lines[:-1]

            selected = all_lines[-lines:] if len(all_lines) > lines else all_lines

            result = "\n".join(f"{i + 1:6}  {line}" for i, line in enumerate(selected))
            header = f"Last {len(selected)} line(s) of '{path}':\n\n"
            return header + result

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error reading tail of file: {e}")

    def list_files(self, path: str, include_hidden: bool = False) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"Path '{path}' not found.")
            
            results = []
            if target.is_file():
                st = target.stat()
                return f"[FILE] {target.name} ({st.st_size} bytes)"
            
            # ✅ Исправлено: Возможность отображения скрытых файлов по флагу
            for child in sorted(target.iterdir()):
                if child.name.startswith('.') and not include_hidden: 
                    continue
                # Откровенный системный мусор не показываем даже если include_hidden=True
                if child.name in {'.DS_Store', '__pycache__'}:
                    continue
                
                try:
                    prefix = "[DIR] " if child.is_dir() else "[FILE]"
                    name = child.name
                    results.append(f"{prefix} {name}")
                except OSError:
                    continue
            
            count = len(results)
            output = f"Directory '{path}':\n" + "\n".join(results) + f"\n\n(Total {count} items)"
            
            limit = self.safety_policy.max_tool_output if self.safety_policy else 5000
            return truncate_output(output, limit, source="filesystem")

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error listing directory: {e}")

# --- Tool Definitions ---

fs_manager = FilesystemManager(virtual_mode=True)

def set_safety_policy(policy: SafetyPolicy):
    fs_manager.set_policy(policy)

@tool("read_file")
def read_file_tool(path: str, offset: int = 0, limit: int = 2000) -> str:
    """Reads a file from the filesystem.
    Args:
        path: Relative path to file
        offset: Line number to start reading from (0-indexed, default 0)
        limit: Max lines to read (default 2000)
    """
    return fs_manager.read_file(path, offset, limit)

@tool("write_file")
def write_file_tool(path: str, content: str) -> str:
    """Writes content to a file. Creates directories if needed. Overwrites existing files.
    Args:
        path: Relative path to file
        content: Text content to write
    """
    return fs_manager.write_file(path, content)

@tool("edit_file")
def edit_file_tool(path: str, old_string: str, new_string: str) -> str:
    """Replaces exact text in a file. Returns a Unified Diff of changes.
    Args:
        path: Relative path to file
        old_string: Exact text block to replace. MUST be unique within the file.
        new_string: New text block to insert
    """
    return fs_manager.edit_file(path, old_string, new_string)

@tool("list_directory")
def list_directory_tool(path: str = ".", include_hidden: bool = False) -> str:
    """Lists files and directories in a given path.
    Args:
        path: Directory path (default ".")
        include_hidden: Include files starting with dot (e.g. .env, .gitignore)
    """
    return fs_manager.list_files(path, include_hidden)

@tool("search_in_file")
def search_in_file_tool(path: str, pattern: str, use_regex: bool = False, ignore_case: bool = False) -> str:
    """Searches for a text pattern (or regex) in a single file."""
    return fs_manager.search_in_file(path, pattern, use_regex, ignore_case)

@tool("search_in_directory")
def search_in_directory_tool(path: str, pattern: str, use_regex: bool = False,
                              ignore_case: bool = False, extensions: Optional[str] = None,
                              max_matches: int = 500, max_files: int = 200,
                              max_depth: Optional[int] = None) -> str:
    """Recursively searches for a text pattern (or regex) across all files in a directory."""
    return fs_manager.search_in_directory(path, pattern, use_regex, ignore_case,
                                          extensions, max_matches, max_files, max_depth)

@tool("tail_file")
def tail_file_tool(path: str, lines: int = 50) -> str:
    """Returns the last N lines of a file with line numbers (like Unix `tail`)."""
    return fs_manager.tail_file(path, lines)

@tool("download_file")
async def download_file(url: str, filename: Optional[str] = None) -> str:
    """Downloads a file from a URL to the current working directory."""
    try:
        if not filename:
            filename = url.split("/")[-1] or "downloaded_file"
        
        if os.path.sep in filename or (os.path.altsep and os.path.altsep in filename):
             return format_error(ErrorType.VALIDATION, f"Invalid filename '{filename}'.")
             
        destination = Path.cwd() / filename
        client = get_net_client() 
        logger.info(f"⬇️ Downloading {url} to {destination}")
        
        try:
            async with client.client.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                
                max_size = fs_manager.safety_policy.max_file_size if fs_manager.safety_policy else 10*1024*1024
                if content_length and int(content_length) > max_size:
                    return format_error(ErrorType.LIMIT_EXCEEDED, f"File too large (>{max_size} bytes). Download aborted.")

                # ✅ Исправлено: Асинхронная запись с помощью aiofiles (не блокирует event loop)
                async with aiofiles.open(destination, "wb") as f:
                    downloaded = 0
                    async for chunk in response.aiter_bytes():
                        downloaded += len(chunk)
                        if downloaded > max_size:
                            return format_error(ErrorType.LIMIT_EXCEEDED, f"File exceeded max size {max_size}. Aborted.")
                        await f.write(chunk)
                        
            return f"Success: File downloaded to {destination}"
        except httpx.HTTPStatusError as e:
            return format_error(ErrorType.NETWORK, f"HTTP {e.response.status_code} - {e.response.reason_phrase}")
        except httpx.RequestError as e:
            return format_error(ErrorType.NETWORK, f"Network request failed: {e}")
    except Exception as e:
        return format_error(ErrorType.EXECUTION, f"Error downloading file: {e}")