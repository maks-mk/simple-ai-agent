"""
Advanced Filesystem Tools Module.
Based on deepagents filesystem implementation.
Features:
- Virtual Mode (Security Sandbox)
- Unified Diff generation for edits
- Pagination for reading large files
- Safe Path Resolution
- Optimized os.scandir traversals
"""

import os
import re
import difflib
import aiofiles
import itertools
from functools import lru_cache
from pathlib import Path
from typing import Union, Optional
import logging
import httpx

from langchain_core.tools import tool
from tools.system_tools import get_net_client
from core.safety_policy import SafetyPolicy
from core.errors import format_error, ErrorType
from core.utils import truncate_output
from core.config import DEFAULT_MAX_FILE_SIZE, DEFAULT_READ_LIMIT

logger = logging.getLogger(__name__)

# --- Constants ---

IGNORED_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".next", ".nuxt",
    "venv", ".venv", "env", ".env",
    "dist", "build", "out", "target",
    ".idea", ".vscode",
}

# Быстрая проверка бинарности по расширению (skip disk I/O)
_KNOWN_TEXT_EXTS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss', '.less',
    '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.md', '.rst', '.txt', '.csv', '.log', '.sh', '.bat', '.ps1',
    '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.rb', '.php',
    '.sql', '.env', '.gitignore', '.dockerignore', '.editorconfig',
}
_KNOWN_BINARY_EXTS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.svg',
    '.mp3', '.mp4', '.avi', '.mkv', '.wav', '.flac', '.ogg',
    '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.xz',
    '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.ttf', '.otf', '.woff', '.woff2', '.eot',
    '.pyc', '.pyo', '.class', '.o', '.obj',
}

@lru_cache(maxsize=512)
def _is_binary_cached(path_str: str) -> bool:
    """Check if file is binary. Cached by path string, with fast extension shortcut."""
    p = Path(path_str)
    ext = p.suffix.lower()
    if ext in _KNOWN_TEXT_EXTS:
        return False
    if ext in _KNOWN_BINARY_EXTS:
        return True
    try:
        with open(path_str, 'rb') as f:
            chunk = f.read(8192)
            return b'\x00' in chunk
    except Exception:
        return True

class FilesystemManager:
    __slots__ = ('cwd', 'virtual_mode', 'safety_policy')

    def __init__(self, root_dir: Union[str, Path] = None, virtual_mode: bool = True):
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.safety_policy: Optional[SafetyPolicy] = None

    def set_policy(self, policy: SafetyPolicy):
        self.safety_policy = policy

    def _is_binary(self, path: Union[str, Path]) -> bool:
        """Check if file is binary. Uses cached function with extension shortcuts."""
        return _is_binary_cached(str(path))

    def _count_lines(self, path: Path) -> int:
        """Count lines in file efficiently using binary chunk reading."""
        try:
            if path.stat().st_size == 0:
                return 0
            
            count = 0
            last_chunk = b""
            with open(path, 'rb') as f:
                # Walrus operator for efficient chunk reading
                while chunk := f.read(65536):
                    count += chunk.count(b'\n')
                    last_chunk = chunk
            
            if last_chunk and not last_chunk.endswith(b'\n'):
                count += 1
            return count 
        except Exception:
             return 0

    def _resolve_path(self, path_str: str) -> Path:
        """Resolves and validates paths strictly within the sandbox if virtual_mode is on."""
        if not path_str:
            raise ValueError("Path cannot be empty")
            
        clean_path = str(path_str).replace("\\", "/")
        path_obj = Path(clean_path).expanduser()
        
        if self.virtual_mode:
            if path_obj.is_absolute():
                raise ValueError(f"ACCESS DENIED: Absolute paths not allowed in virtual mode: {path_str}")
            
            full_path = (self.cwd / path_obj).resolve()
            
            # Use native is_relative_to (Python 3.9+) for safe bounds checking
            if not full_path.is_relative_to(self.cwd):
                raise ValueError(f"ACCESS DENIED: Path traversal outside working directory: {full_path}")
                
            return full_path

        return path_obj.resolve() if path_obj.is_absolute() else (self.cwd / path_obj).resolve()

    def read_file(self, path: str, offset: int = 0, limit: int = DEFAULT_READ_LIMIT, show_line_numbers: bool = True) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            if not target.is_file(): return format_error(ErrorType.VALIDATION, f"'{path}' is not a file.")
            
            stats = target.stat()
            max_size = self.safety_policy.max_file_size if self.safety_policy else DEFAULT_MAX_FILE_SIZE
            if stats.st_size > max_size:
                return format_error(ErrorType.LIMIT_EXCEEDED, f"File is too large ({stats.st_size} bytes). Max: {max_size}.")

            if self._is_binary(target):
                return format_error(ErrorType.VALIDATION, "File binary or unknown encoding.")

            if stats.st_size == 0:
                return "System reminder: File exists but has empty contents."

            total_lines = self._count_lines(target)
            
            policy_limit = self.safety_policy.max_read_lines if self.safety_policy else DEFAULT_READ_LIMIT
            limit = min(limit, policy_limit)

            if offset >= total_lines and total_lines > 0:
                return format_error(ErrorType.VALIDATION, f"Line offset {offset} exceeds file length ({total_lines} lines).")
            
            result =[]
            try:
                with open(target, 'r', encoding='utf-8', errors='replace') as f:
                    for i, line in enumerate(itertools.islice(f, offset, offset + limit)):
                        clean_line = line.rstrip('\n').rstrip('\r')
                        if show_line_numbers:
                            result.append(f"{offset + i + 1:6}  {clean_line}")
                        else:
                            result.append(clean_line)
            except Exception as e:
                return format_error(ErrorType.EXECUTION, f"Error reading file stream: {e}")
                
            output = "\n".join(result)
            
            end_index = offset + len(result)
            if total_lines > end_index:
                output += f"\n\n... (Showing lines {offset+1}-{end_index} of {total_lines}. Use offset/limit to read more)"
                
            return output

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error reading file: {e}")

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self._resolve_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
            return f"Success: File '{path}' saved ({len(content)} chars)."
        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error writing file: {e}")

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        """
        Smart text replacement: attempts exact match first, 
        then falls back to fuzzy line-by-line match (ignoring ALL whitespace).
        """
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            
            content = target.read_text(encoding='utf-8')
            original_newline = '\r\n' if '\r\n' in content else '\n'
            
            content_norm = content.replace('\r\n', '\n')
            old_string_norm = old_string.replace('\r\n', '\n')
            new_string_norm = new_string.replace('\r\n', '\n')
            
            count = content_norm.count(old_string_norm)
            
            if count == 1:
                new_content = content_norm.replace(old_string_norm, new_string_norm, 1)
            elif count > 1:
                return format_error(
                    ErrorType.VALIDATION, 
                    f"Found {count} identical occurrences of the exact target text. "
                    "Please provide more context (e.g., surrounding lines) to uniquely identify the block."
                )
            else:
                # FUZZY MATCH (Бронебойная стратегия)
                file_lines = content_norm.split('\n')
                search_lines = old_string_norm.split('\n')
                
                while search_lines and not search_lines[0].strip():
                    search_lines.pop(0)
                while search_lines and not search_lines[-1].strip():
                    search_lines.pop()
                    
                if not search_lines:
                    return format_error(ErrorType.VALIDATION, "old_string is empty or only contains whitespaces.")
                
                # Функция для удаления ВСЕХ пробелов и табов для сравнения
                def normalize_line(s: str) -> str:
                    return "".join(s.split())
                    
                search_lines_normalized =[normalize_line(l) for l in search_lines]
                file_lines_normalized =[normalize_line(l) for l in file_lines]
                
                matches =[]
                search_len = len(search_lines)
                
                for i in range(len(file_lines) - search_len + 1):
                    if file_lines_normalized[i : i + search_len] == search_lines_normalized:
                        matches.append(i)
                        
                if len(matches) == 0:
                    snippet = old_string_norm[:50].replace('\n', '\\n')
                    return format_error(
                        ErrorType.VALIDATION, 
                        f"Could not find a match for 'old_string' (even when ignoring ALL spaces/indentation).\n"
                        f"Snippet: '{snippet}...'.\n"
                        "Make sure you are replacing existing lines and DID NOT include line numbers in old_string."
                    )
                elif len(matches) > 1:
                    return format_error(
                        ErrorType.VALIDATION, 
                        f"Found {len(matches)} occurrences of the text (ignoring spaces). "
                        "Please include more surrounding lines to uniquely identify the block."
                    )
                else:
                    match_idx = matches[0]
                    new_string_lines = new_string_norm.split('\n')
                    new_file_lines = (
                        file_lines[:match_idx] + 
                        new_string_lines + 
                        file_lines[match_idx + len(search_lines):]
                    )
                    new_content = '\n'.join(new_file_lines)
            
            final_content = new_content.replace('\n', original_newline)
            
            with open(target, 'w', encoding='utf-8', newline='') as f:
                f.write(final_content)
            
            # Generator expression inline diff creation
            diff = difflib.unified_diff(
                content_norm.splitlines(),
                new_content.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm=""
            )
            diff_text = "\n".join(diff)
            
            return f"Success: File edited.\n\nDiff:\n```diff\n{diff_text}\n```"

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error editing file: {e}")

    def search_in_file(self, path: str, pattern: str, use_regex: bool = False, ignore_case: bool = False) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            if not target.is_file(): return format_error(ErrorType.VALIDATION, f"'{path}' is not a file.")

            if self._is_binary(target):
                return format_error(ErrorType.VALIDATION, "File is binary or has unknown encoding.")

            flags = re.IGNORECASE if ignore_case else 0
            matches =[]

            if use_regex:
                try:
                    content = target.read_text(encoding='utf-8', errors='replace')
                except UnicodeDecodeError:
                    return format_error(ErrorType.VALIDATION, "File is binary or has unknown encoding.")

                flags |= re.MULTILINE
                try:
                    compiled = re.compile(pattern, flags)
                except re.error as e:
                    return format_error(ErrorType.VALIDATION, f"Invalid regex pattern: {e}")
                
                for m in compiled.finditer(content):
                    start_pos = m.start()
                    line_no = content.count('\n', 0, start_pos) + 1
                    match_str = m.group(0)
                    display_line = match_str.splitlines()[0] if match_str else ""
                    if not display_line.strip():
                        line_start = content.rfind('\n', 0, start_pos) + 1
                        line_end = content.find('\n', start_pos)
                        if line_end == -1: line_end = len(content)
                        display_line = content[line_start:line_end]
                    
                    matches.append((line_no, display_line))
            else:
                needle = pattern.lower() if ignore_case else pattern
                try:
                    with open(target, 'r', encoding='utf-8', errors='replace') as f:
                        for i, line in enumerate(f):
                            if needle in (line.lower() if ignore_case else line):
                                matches.append((i + 1, line.rstrip('\n').rstrip('\r')))
                except Exception as e:
                     return format_error(ErrorType.EXECUTION, f"Error reading file stream: {e}")

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
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"Path '{path}' not found.")
            if not target.is_dir(): return format_error(ErrorType.VALIDATION, f"'{path}' is not a directory.")

            ext_filter = None
            if extensions:
                ext_filter = {e.strip() if e.strip().startswith(".") else f".{e.strip()}"
                              for e in extensions.split(",")}

            flags = re.IGNORECASE if ignore_case else 0
            compiled_regex = None
            needle = None

            if use_regex:
                flags |= re.MULTILINE
                try:
                    compiled_regex = re.compile(pattern, flags)
                except re.error as e:
                    return format_error(ErrorType.VALIDATION, f"Invalid regex pattern: {e}")
            else:
                needle = pattern.lower() if ignore_case else pattern

            all_results: list[str] =[]
            files_scanned = 0
            dirs_skipped_ignored = 0
            files_skipped_binary = 0
            max_size = self.safety_policy.max_file_size if self.safety_policy else DEFAULT_MAX_FILE_SIZE
            truncated = False

            # Optimized tree traversal using Stack and os.scandir to avoid extra stat() calls
            dirs_to_scan = [(target, 0)]
            
            while dirs_to_scan and not truncated:
                current_dir, depth = dirs_to_scan.pop()
                
                if max_depth is not None and depth >= max_depth:
                    continue

                try:
                    with os.scandir(current_dir) as it:
                        for entry in it:
                            if truncated: break
                            
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name in IGNORED_DIRS:
                                    dirs_skipped_ignored += 1
                                else:
                                    dirs_to_scan.append((Path(entry.path), depth + 1))
                                continue
                                
                            if not entry.is_file(follow_symlinks=False):
                                continue

                            # Fast extension filter
                            if ext_filter and Path(entry.name).suffix not in ext_filter:
                                continue
                            
                            # Fast size filter (scandir caches stat)
                            try:
                                if entry.stat(follow_symlinks=False).st_size > max_size:
                                    continue
                            except OSError:
                                continue

                            if files_scanned >= max_files:
                                truncated = True
                                break

                            file_path = entry.path
                            
                            try:
                                if self._is_binary(file_path):
                                    files_skipped_binary += 1
                                    continue
                            except OSError:
                                files_skipped_binary += 1
                                continue

                            files_scanned += 1
                            try:
                                rel_path = Path(file_path).relative_to(self.cwd)
                            except ValueError:
                                rel_path = file_path

                            # Scanning content
                            try:
                                if use_regex:
                                    content = Path(file_path).read_text(encoding='utf-8', errors='replace')
                                    for m in compiled_regex.finditer(content):
                                        start_pos = m.start()
                                        line_no = content.count('\n', 0, start_pos) + 1
                                        match_str = m.group(0)
                                        display_line = match_str.splitlines()[0] if match_str else ""
                                        if not display_line.strip():
                                            line_start = content.rfind('\n', 0, start_pos) + 1
                                            line_end = content.find('\n', start_pos)
                                            if line_end == -1: line_end = len(content)
                                            display_line = content[line_start:line_end]
                                        
                                        all_results.append(f"{rel_path}:{line_no}  {display_line}")
                                        if len(all_results) >= max_matches:
                                            truncated = True
                                            break
                                else:
                                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                        for i, line in enumerate(f):
                                            target_line = line.lower() if ignore_case else line
                                            if needle in target_line:
                                                all_results.append(f"{rel_path}:{i + 1}  {line.rstrip()}")
                                                if len(all_results) >= max_matches:
                                                    truncated = True
                                                    break
                            except Exception:
                                continue

                except PermissionError:
                    continue

            if not all_results:
                return (f"No matches found for '{pattern}' in '{path}'. "
                        f"Scanned {files_scanned} file(s), "
                        f"skipped {dirs_skipped_ignored} ignored dirs, "
                        f"{files_skipped_binary} binary files.")

            header_lines =[f"Found {len(all_results)} match(es) across {files_scanned} file(s)."]
            if truncated:
                header_lines.append(f"⚠ Results truncated (max_matches={max_matches}, max_files={max_files}). Narrow search.")
            if dirs_skipped_ignored:
                header_lines.append(f"  Skipped {dirs_skipped_ignored} ignored directory branches.")

            output = "\n".join(header_lines) + "\n\n" + "\n".join(all_results)
            limit = self.safety_policy.max_tool_output if self.safety_policy else 5000
            return truncate_output(output, limit, source="filesystem")

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error searching in directory: {e}")

    def tail_file(self, path: str, lines: int = 50, show_line_numbers: bool = True) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            if not target.is_file(): return format_error(ErrorType.VALIDATION, f"'{path}' is not a file.")

            policy_limit = self.safety_policy.max_read_lines if self.safety_policy else DEFAULT_READ_LIMIT
            lines = min(lines, policy_limit)

            collected: list[bytes] =[]
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

            if show_line_numbers:
                result = "\n".join(f"tail-{i + 1:02}  {line}" for i, line in enumerate(selected))
            else:
                result = "\n".join(selected)
                
            header = f"Last {len(selected)} line(s) of '{path}':\n\n"
            return header + result

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error reading tail of file: {e}")

    def find_files(self, path: str, name_pattern: str, max_results: int = 200, max_depth: Optional[int] = None) -> str:
        import fnmatch
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"Path '{path}' not found.")
            if not target.is_dir(): return format_error(ErrorType.VALIDATION, f"'{path}' is not a directory.")

            all_results =[]
            files_scanned = 0
            dirs_skipped = 0
            truncated = False

            dirs_to_scan =[(target, 0)]
            
            while dirs_to_scan and not truncated:
                current_dir, depth = dirs_to_scan.pop()
                
                if max_depth is not None and depth >= max_depth:
                    continue

                try:
                    with os.scandir(current_dir) as it:
                        for entry in it:
                            if truncated: break
                            
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name in IGNORED_DIRS:
                                    dirs_skipped += 1
                                else:
                                    dirs_to_scan.append((Path(entry.path), depth + 1))
                                continue
                                
                            if not entry.is_file(follow_symlinks=False):
                                continue

                            files_scanned += 1
                            
                            # Поиск по имени с поддержкой масок (*, ?)
                            if fnmatch.fnmatch(entry.name, name_pattern) or fnmatch.fnmatch(entry.name.lower(), name_pattern.lower()):
                                try:
                                    rel_path = Path(entry.path).relative_to(self.cwd)
                                except ValueError:
                                    rel_path = Path(entry.path)
                                
                                all_results.append(str(rel_path).replace("\\", "/"))
                                if len(all_results) >= max_results:
                                    truncated = True
                                    break
                except PermissionError:
                    continue

            if not all_results:
                return (f"No files matching '{name_pattern}' found in '{path}'. "
                        f"Scanned {files_scanned} files, skipped {dirs_skipped} ignored directories.")

            header = f"Found {len(all_results)} match(es) for '{name_pattern}':"
            if truncated:
                header += f" (Truncated to {max_results} max results)"
                
            output = header + "\n\n" + "\n".join(all_results)
            limit = self.safety_policy.max_tool_output if self.safety_policy else 5000
            return truncate_output(output, limit, source="filesystem")

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error finding files: {e}")
    
    def list_files(self, path: str, include_hidden: bool = False) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"Path '{path}' not found.")
            
            if target.is_file():
                return f"[FILE] {target.name} ({target.stat().st_size} bytes)"
            
            results =[]
            try:
                # Optimized directory listing using os.scandir
                with os.scandir(target) as it:
                    entries = sorted(list(it), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
                    for entry in entries:
                        if entry.name.startswith('.') and not include_hidden:
                            continue
                        if entry.name in {'.DS_Store', '__pycache__'}:
                            continue
                        
                        prefix = "[DIR] " if entry.is_dir(follow_symlinks=False) else "[FILE]"
                        results.append(f"{prefix} {entry.name}")
            except PermissionError:
                return format_error(ErrorType.EXECUTION, "Permission denied while accessing directory.")
            
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
def read_file_tool(path: str, offset: int = 0, limit: int = 2000, show_line_numbers: bool = True) -> str:
    """Reads a file from the filesystem.
    Args:
        path: Relative path to file
        offset: Line number to start reading from (0-indexed, default 0)
        limit: Max lines to read (default 2000)
        show_line_numbers: Set to False to get raw text without line numbers. Use False if you plan to copy-paste exact text into edit_file!
    """
    return fs_manager.read_file(path, offset, limit, show_line_numbers)

@tool("write_file")
def write_file_tool(path: str, content: str) -> str:
    """Writes content to a file. Overwrites existing files completely.
    WARNING: NEVER use this tool to modify an existing file. It will destroy the file's formatting or minify it. ALWAYS use 'edit_file' to modify existing files.
    Args:
        path: Relative path to file
        content: Text content to write
    """
    return fs_manager.write_file(path, content)

@tool("edit_file")
def edit_file_tool(path: str, old_string: str, new_string: str) -> str:
    """Replaces text in a file. Very smart: if exact match fails, it ignores spaces/indentation and finds the right block automatically.
    IMPORTANT RULES FOR LLM:
    1. 'old_string' MUST contain the exact text from the file (do NOT include the line numbers from read_file output).
    2. Provide enough context (2-3 surrounding lines in old_string) so the tool can uniquely identify the block to replace.
    3. Do not use regex here.
    Args:
        path: Relative path to file
        old_string: The text block to replace.
        new_string: The new text block to insert.
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
    """Searches for a text pattern (or regex) in a single file.
    Supports multiline regex if use_regex=True (e.g. use '(?s)pattern' to match across lines)."""
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
def tail_file_tool(path: str, lines: int = 50, show_line_numbers: bool = True) -> str:
    """Returns the last N lines of a file (like Unix `tail`).
    Args:
        path: Relative path to file
        lines: Number of lines to return
        show_line_numbers: Whether to prepend line numbers
    """
    return fs_manager.tail_file(path, lines, show_line_numbers)

@tool("download_file")
async def download_file(url: str, filename: Optional[str] = None) -> str:
    """Downloads a file from a URL to the current working directory."""
    try:
        if not filename:
            filename = url.split("/")[-1] or "downloaded_file"
        
        try:
            destination = fs_manager._resolve_path(filename)
        except ValueError as e:
            return format_error(ErrorType.VALIDATION, str(e))
             
        client = get_net_client() 
        logger.info(f"⬇️ Downloading {url} to {destination}")
        
        try:
            async with client.client.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                
                max_size = fs_manager.safety_policy.max_file_size if fs_manager.safety_policy else DEFAULT_MAX_FILE_SIZE
                if content_length and int(content_length) > max_size:
                    return format_error(ErrorType.LIMIT_EXCEEDED, f"File too large (>{max_size} bytes). Download aborted.")

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
        
@tool("find_file")
def find_file_tool(path: str, name_pattern: str, max_results: int = 200, max_depth: Optional[int] = None) -> str:
    """Finds files by their name pattern in a directory (recursive).
    Supports glob patterns like '*.py', 'mcp.json', 'test_*.py'.
    Does NOT search inside files, only checks file names.
    Args:
        path: Directory path to search in (default ".")
        name_pattern: The filename or glob pattern to search for
        max_results: Max number of file paths to return
        max_depth: How deep to search (None for unlimited)
    """
    return fs_manager.find_files(path, name_pattern, max_results, max_depth)