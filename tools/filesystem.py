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
        
        # 2. Virtual Mode Checks (Strict Sandbox)
        if self.virtual_mode:
            # Treat all paths as relative to root, even if they start with /
            # e.g. /etc/passwd -> {root}/etc/passwd
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

        # 3. Legacy Mode (Less Secure, follows os_tools behavior)
        path_obj = Path(clean_path)
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
            
            # Try exact match
            if old_string not in content:
                # Try normalization (CRLF -> LF)
                content_norm = content.replace('\r\n', '\n')
                old_string_norm = old_string.replace('\r\n', '\n')
                
                if old_string_norm in content_norm:
                    # Proceed with normalized content
                    content = content_norm
                    old_string = old_string_norm
                else:
                    snippet = old_string[:50].replace('\n', '\\n')
                    return format_error(ErrorType.VALIDATION, f"Could not find target text starting with: '{snippet}...'")

            # Perform replacement
            new_content = content.replace(old_string, new_string)
            
            target.write_text(new_content, encoding='utf-8')
            
            # Generate Diff
            diff = difflib.unified_diff(
                content.splitlines(),
                new_content.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm=""
            )
            diff_text = "\n".join(list(diff))
            
            return f"Success: File edited.\n\nDiff:\n```diff\n{diff_text}\n```"

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error editing file: {e}")

    def list_files(self, path: str) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists(): return format_error(ErrorType.NOT_FOUND, f"Path '{path}' not found.")
            
            results = []
            if target.is_file():
                st = target.stat()
                return f"[FILE] {target.name} ({st.st_size} bytes)"
            
            # Directory listing
            for child in sorted(target.iterdir()):
                # Skip hidden
                if child.name.startswith('.'): continue
                
                try:
                    prefix = "[DIR] " if child.is_dir() else "[FILE]"
                    name = child.name
                    results.append(f"{prefix} {name}")
                except OSError:
                    continue
            
            count = len(results)
            output = f"Directory '{path}':\n" + "\n".join(results) + f"\n\n(Total {count} items)"
            
            # Truncate if too long (using SafetyPolicy from manager if set)
            limit = self.safety_policy.max_tool_output if self.safety_policy else 5000
            return truncate_output(output, limit, source="filesystem")

        except Exception as e:
            return format_error(ErrorType.EXECUTION, f"Error listing directory: {e}")

# --- Tool Definitions ---

# Global instance for tools
# We assume CWD is the project root. Virtual mode enabled by default for safety.
fs_manager = FilesystemManager(virtual_mode=True)

def set_safety_policy(policy: SafetyPolicy):
    fs_manager.set_policy(policy)

@tool("read_file")
def read_file_tool(path: str, offset: int = 0, limit: int = 2000) -> str:
    """
    Reads a file from the filesystem.
    Args:
        path: Relative path to file
        offset: Line number to start reading from (0-indexed, default 0)
        limit: Max lines to read (default 2000)
    """
    return fs_manager.read_file(path, offset, limit)

@tool("write_file")
def write_file_tool(path: str, content: str) -> str:
    """
    Writes content to a file. Creates directories if needed. Overwrites existing files.
    Args:
        path: Relative path to file
        content: Text content to write
    """
    return fs_manager.write_file(path, content)

@tool("edit_file")
def edit_file_tool(path: str, old_string: str, new_string: str) -> str:
    """
    Replaces exact text in a file. Returns a Unified Diff of changes.
    Args:
        path: Relative path to file
        old_string: Exact text block to replace (must match file content exactly)
        new_string: New text block to insert
    """
    return fs_manager.edit_file(path, old_string, new_string)

@tool("list_directory")
def list_directory_tool(path: str = ".") -> str:
    """
    Lists files and directories in a given path.
    Args:
        path: Directory path (default ".")
    """
    return fs_manager.list_files(path)

@tool("download_file")
async def download_file(url: str, filename: Optional[str] = None) -> str:
    """
    Downloads a file from a URL to the current working directory.
    Uses httpx for downloading.
    """
    try:
        if not filename:
            filename = url.split("/")[-1] or "downloaded_file"
        
        if os.path.sep in filename or (os.path.altsep and os.path.altsep in filename):
             return format_error(ErrorType.VALIDATION, f"Invalid filename '{filename}'.")
             
        destination = Path.cwd() / filename
        client = get_net_client() # Используем клиент из system_tools
        logger.info(f"⬇️ Downloading {url} to {destination}")
        
        try:
            async with client.client.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                
                # Check limit
                max_size = fs_manager.safety_policy.max_file_size if fs_manager.safety_policy else 10*1024*1024
                if content_length and int(content_length) > max_size:
                    return format_error(ErrorType.LIMIT_EXCEEDED, f"File too large (>{max_size} bytes). Download aborted.")

                with open(destination, "wb") as f:
                    downloaded = 0
                    async for chunk in response.aiter_bytes():
                        downloaded += len(chunk)
                        if downloaded > max_size:
                            return format_error(ErrorType.LIMIT_EXCEEDED, f"File exceeded max size {max_size}. Aborted.")
                        f.write(chunk)
            return f"Success: File downloaded to {destination}"
        except httpx.HTTPStatusError as e:
            return format_error(ErrorType.NETWORK, f"HTTP {e.response.status_code} - {e.response.reason_phrase}")
        except httpx.RequestError as e:
            return format_error(ErrorType.NETWORK, f"Network request failed: {e}")
    except Exception as e:
        return format_error(ErrorType.EXECUTION, f"Error downloading file: {e}")
