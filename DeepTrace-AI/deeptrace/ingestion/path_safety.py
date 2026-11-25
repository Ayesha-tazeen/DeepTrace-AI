"""Filename and path safety checks (path traversal, Windows reserved names)."""

from __future__ import annotations

import re
from pathlib import Path, PurePath
from typing import List, Optional, Tuple

_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"}
_WINDOWS_RESERVED.update(f"COM{i}" for i in range(1, 10))
_WINDOWS_RESERVED.update(f"LPT{i}" for i in range(1, 10))
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def check_filename(name: str, max_length: int = 255) -> List[str]:
    """Return a list of problems with ``name`` (empty list means safe)."""
    errors: List[str] = []
    if not name or not name.strip():
        return ["Filename is empty."]
    if len(name) > max_length:
        errors.append(f"Filename is longer than {max_length} characters.")
    if name in {".", ".."} or ".." in name:
        errors.append("Filename must not contain '..'.")
    if _INVALID_CHARS.search(name):
        errors.append("Filename contains invalid or control characters (<>:\"/\\|?* etc.).")
    if name != name.strip() or name.endswith("."):
        errors.append("Filename must not start/end with spaces or end with a dot.")
    if name.split(".")[0].strip().upper() in _WINDOWS_RESERVED:
        errors.append("Filename uses a reserved Windows device name.")
    return errors


def check_path_string(raw: str) -> List[str]:
    """Check a user-supplied path string before touching the file system."""
    errors: List[str] = []
    if "\x00" in raw:
        errors.append("Path contains a null byte.")
        return errors
    if raw.startswith("\\\\"):
        errors.append("Network (UNC) paths are not accepted.")
    if ".." in PurePath(raw).parts:
        errors.append("Path traversal ('..') is not allowed.")
    return errors


def is_within_directory(path: Path, root: Path) -> bool:
    """True if ``path`` (after resolving symlinks) is inside ``root``."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_input_path(
    raw: str, allowed_root: Optional[Path] = None
) -> Tuple[Optional[Path], List[str]]:
    """Resolve a user path to a real file, returning ``(path, errors)``."""
    errors = check_path_string(raw)
    if errors:
        return None, errors
    candidate = Path(raw)
    try:
        if candidate.is_symlink():
            return None, ["Symbolic links are not accepted."]
        if not candidate.exists():
            return None, [f"File does not exist: {candidate.name}"]
        if not candidate.is_file():
            return None, ["Path is not a regular file."]
        resolved = candidate.resolve()
    except OSError as exc:
        return None, [f"Cannot access path: {exc}"]
    if allowed_root is not None and not is_within_directory(resolved, allowed_root):
        return None, ["File is outside the allowed directory."]
    return resolved, []


def sanitize_filename(name: str, max_length: int = 60) -> str:
    """Turn an arbitrary name into a safe ASCII stem for storage."""
    base = re.split(r"[\\/]", name)[-1]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    base = re.sub(r"_+", "_", base).strip("._ ")
    base = base[:max_length].strip("._ ")
    if not base or base.upper() in _WINDOWS_RESERVED:
        base = "file"
    return base
