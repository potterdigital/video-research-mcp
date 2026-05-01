"""Policy helpers for local filesystem access boundaries."""

from __future__ import annotations

from pathlib import Path

from .config import get_config


def resolve_path(path_value: str) -> Path:
    """Resolve a user-supplied path to an absolute filesystem path."""
    return Path(path_value).expanduser().resolve()


def enforce_local_access_root(path: Path) -> Path:
    """Enforce LOCAL_FILE_ACCESS_ROOTS boundary when configured.

    Accepts the path if it falls under ANY configured access root. When no
    roots are configured the path is returned unchanged (legacy behaviour).

    Args:
        path: Absolute filesystem path to validate.

    Returns:
        The original path when allowed.

    Raises:
        PermissionError: If the path falls outside every configured access root.
    """
    cfg = get_config()
    roots = cfg.local_file_access_roots
    if not roots:
        return path

    resolved_roots = [Path(r).expanduser().resolve() for r in roots]
    for root in resolved_roots:
        if path.is_relative_to(root):
            return path

    roots_display = ", ".join(str(r) for r in resolved_roots)
    raise PermissionError(
        f"Path '{path}' is outside LOCAL_FILE_ACCESS_ROOTS [{roots_display}]"
    )
