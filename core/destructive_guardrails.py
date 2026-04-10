from __future__ import annotations

import os
import re
from pathlib import Path

from core.errors import ErrorType, format_error

_SYSTEM_ENV_VARS = (
    "SystemRoot",
    "WINDIR",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramData",
)
_WINDOWS_SYSTEM_DIR_NAMES = {
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "system volume information",
    "$recycle.bin",
}
_SHELL_RECURSIVE_PATTERNS = (
    re.compile(r"(^|\s)del\s+/[sqrf]+\b", re.IGNORECASE),
    re.compile(r"(^|\s)rmdir\s+/[sq]+\b", re.IGNORECASE),
    re.compile(r"Remove-Item\b.*\s-Recurse\b", re.IGNORECASE),
    re.compile(r"(^|\s)rm\s+-[^\n\r]*r", re.IGNORECASE),
)


def _normalized_roots() -> set[Path]:
    roots: set[Path] = set()
    for env_name in _SYSTEM_ENV_VARS:
        value = os.environ.get(env_name)
        if not value:
            continue
        try:
            roots.add(Path(value).resolve())
        except OSError:
            continue
    return roots


def _is_windows_root(path: Path) -> bool:
    anchor = path.anchor.rstrip("\\/")
    if not anchor:
        return False
    return bool(re.fullmatch(r"[A-Za-z]:", anchor))


def _looks_like_system_dir(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts if part and part not in {path.anchor}}
    if lowered_parts & _WINDOWS_SYSTEM_DIR_NAMES:
        return True
    return any(root == path or root in path.parents for root in _normalized_roots())


def deny_recursive_destructive_path(path: Path, *, recursive: bool) -> str | None:
    if not recursive:
        return None

    resolved = path.resolve()
    if _is_windows_root(resolved):
        return format_error(
            ErrorType.ACCESS_DENIED,
            f"Recursive destructive operation is blocked for drive root '{resolved}'.",
        )
    if _looks_like_system_dir(resolved):
        return format_error(
            ErrorType.ACCESS_DENIED,
            f"Recursive destructive operation is blocked for protected system path '{resolved}'.",
        )
    return None


def deny_recursive_destructive_command(command: str) -> str | None:
    stripped = str(command or "").strip()
    if not stripped:
        return None
    for pattern in _SHELL_RECURSIVE_PATTERNS:
        if pattern.search(stripped):
            return format_error(
                ErrorType.ACCESS_DENIED,
                "Recursive destructive shell commands are blocked by runtime guardrails.",
            )
    return None
