"""Policy helpers for local filesystem access boundaries."""

from __future__ import annotations

from pathlib import Path, PurePath

from .config import get_config


# --- Sensitive-path deny-list -------------------------------------------------
#
# These patterns are denied regardless of LOCAL_FILE_ACCESS_ROOTS. Defence in
# depth: even when a user reasonably configures a broad allowed root such as
# ``~/dev``, content that almost always represents secrets must not be readable
# by tools that exfiltrate to remote services.
#
# Directory parts are matched case-sensitively against any path component
# (``~/dev/foo/.ssh/id_rsa`` is denied because ``.ssh`` is in ``path.parts``).
# Filename patterns are evaluated via ``PurePath.match`` against the basename
# only.

SENSITIVE_PATH_DIR_PARTS: frozenset[str] = frozenset(
    {
        ".ssh",
        ".aws",
        ".gnupg",
        ".docker",
        ".kube",
    }
)

# Multi-part directory paths that are sensitive (e.g. ``.config/gh``).
SENSITIVE_PATH_DIR_SUFFIXES: tuple[tuple[str, ...], ...] = (
    (".config", "gh"),
    (".config", "git"),
)

SENSITIVE_PATH_FILENAME_PATTERNS: tuple[str, ...] = (
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "id_dsa*",
    "*.kdbx",
    "*.gpg",
    "*.p12",
    "*.pfx",
    "credentials*",
    ".env",
    ".env.*",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "htpasswd*",
    "*.keystore",
    "*.jks",
)


class SensitivePathError(PermissionError):
    """Raised when a path matches the sensitive-path deny-list.

    Subclass of ``PermissionError`` so existing handlers continue to catch it,
    while specific call sites and the error categoriser can identify it.
    """


def _matched_sensitive_pattern(path: Path) -> str | None:
    """Return the pattern that flags *path* as sensitive, or None if allowed."""
    parts = path.parts
    for part in parts:
        if part in SENSITIVE_PATH_DIR_PARTS:
            return f"directory '{part}'"

    for suffix in SENSITIVE_PATH_DIR_SUFFIXES:
        n = len(suffix)
        for i in range(len(parts) - n + 1):
            if tuple(parts[i : i + n]) == suffix:
                return f"directory '{'/'.join(suffix)}'"

    name = path.name
    pure = PurePath(name)
    for pattern in SENSITIVE_PATH_FILENAME_PATTERNS:
        if pure.match(pattern):
            return f"filename pattern '{pattern}'"

    return None


def resolve_path(path_value: str) -> Path:
    """Resolve a user-supplied path to an absolute filesystem path."""
    return Path(path_value).expanduser().resolve()


def enforce_local_access_root(path: Path) -> Path:
    """Enforce LOCAL_FILE_ACCESS_ROOTS boundary and the sensitive-path deny-list.

    Accepts the path if it falls under ANY configured access root and does not
    match the sensitive-path deny-list (``.ssh/``, ``.aws/``, ``*.pem``, ``.env*``,
    etc). When no roots are configured the root check is skipped (legacy
    behaviour) but the deny-list still applies.

    Args:
        path: Absolute filesystem path to validate.

    Returns:
        The original path when allowed.

    Raises:
        PermissionError: If the path falls outside every configured access root.
        SensitivePathError: If the path matches the sensitive-path deny-list.
    """
    cfg = get_config()
    roots = cfg.local_file_access_roots

    if roots:
        resolved_roots = [Path(r).expanduser().resolve() for r in roots]
        if not any(path.is_relative_to(root) for root in resolved_roots):
            roots_display = ", ".join(str(r) for r in resolved_roots)
            raise PermissionError(
                f"Path '{path}' is outside LOCAL_FILE_ACCESS_ROOTS [{roots_display}]"
            )

    matched = _matched_sensitive_pattern(path)
    if matched is not None:
        raise SensitivePathError(
            f"Path '{path}' matches sensitive-path deny-list ({matched}). "
            "These paths are blocked regardless of LOCAL_FILE_ACCESS_ROOTS to "
            "prevent accidental exfiltration of credentials."
        )

    return path
