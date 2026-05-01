"""Tests for local_path_policy — multi-root access boundaries."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from video_research_mcp.local_path_policy import (
    enforce_local_access_root,
    resolve_path,
)


class TestSingleRoot:
    """Backward-compat: LOCAL_FILE_ACCESS_ROOT (singular) keeps working."""

    def test_path_inside_singular_root_is_allowed(
        self, tmp_path, monkeypatch, clean_config
    ):
        root = tmp_path / "allowed"
        root.mkdir()
        f = root / "doc.txt"
        f.write_text("ok")
        monkeypatch.setenv("LOCAL_FILE_ACCESS_ROOT", str(root))

        result = enforce_local_access_root(resolve_path(str(f)))
        assert result == f.resolve()

    def test_path_outside_singular_root_is_rejected(
        self, tmp_path, monkeypatch, clean_config
    ):
        root = tmp_path / "allowed"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("nope")
        monkeypatch.setenv("LOCAL_FILE_ACCESS_ROOT", str(root))

        with pytest.raises(PermissionError, match="outside LOCAL_FILE_ACCESS_ROOTS"):
            enforce_local_access_root(resolve_path(str(outside)))


class TestMultiRoot:
    """LOCAL_FILE_ACCESS_ROOTS (plural) accepts paths under any listed root."""

    def test_path_under_first_root_is_allowed(
        self, tmp_path, monkeypatch, clean_config
    ):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        f = root_a / "doc.txt"
        f.write_text("ok")
        monkeypatch.setenv(
            "LOCAL_FILE_ACCESS_ROOTS", os.pathsep.join([str(root_a), str(root_b)])
        )

        result = enforce_local_access_root(resolve_path(str(f)))
        assert result == f.resolve()

    def test_path_under_second_root_is_allowed(
        self, tmp_path, monkeypatch, clean_config
    ):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        f = root_b / "doc.txt"
        f.write_text("ok")
        monkeypatch.setenv(
            "LOCAL_FILE_ACCESS_ROOTS", os.pathsep.join([str(root_a), str(root_b)])
        )

        result = enforce_local_access_root(resolve_path(str(f)))
        assert result == f.resolve()

    def test_path_outside_all_roots_is_rejected(
        self, tmp_path, monkeypatch, clean_config
    ):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("nope")
        monkeypatch.setenv(
            "LOCAL_FILE_ACCESS_ROOTS", os.pathsep.join([str(root_a), str(root_b)])
        )

        with pytest.raises(PermissionError, match="outside LOCAL_FILE_ACCESS_ROOTS"):
            enforce_local_access_root(resolve_path(str(outside)))

    def test_plural_takes_precedence_over_singular(
        self, tmp_path, monkeypatch, clean_config
    ):
        """When both env vars are set, the plural list wins."""
        ignored_singular = tmp_path / "ignored"
        plural_root = tmp_path / "plural"
        ignored_singular.mkdir()
        plural_root.mkdir()
        f_in_singular = ignored_singular / "doc.txt"
        f_in_singular.write_text("nope")

        monkeypatch.setenv("LOCAL_FILE_ACCESS_ROOT", str(ignored_singular))
        monkeypatch.setenv("LOCAL_FILE_ACCESS_ROOTS", str(plural_root))

        with pytest.raises(PermissionError, match="outside LOCAL_FILE_ACCESS_ROOTS"):
            enforce_local_access_root(resolve_path(str(f_in_singular)))

    def test_empty_entries_in_plural_list_are_dropped(
        self, tmp_path, monkeypatch, clean_config
    ):
        """A trailing or leading separator must not silently widen access."""
        root = tmp_path / "only"
        root.mkdir()
        f = root / "doc.txt"
        f.write_text("ok")
        monkeypatch.setenv(
            "LOCAL_FILE_ACCESS_ROOTS", f"{os.pathsep}{root}{os.pathsep}{os.pathsep}"
        )

        result = enforce_local_access_root(resolve_path(str(f)))
        assert result == f.resolve()


class TestUnconfigured:
    """When no roots are configured, behaviour matches the legacy singular default."""

    def test_unset_returns_path_unchanged(self, tmp_path, monkeypatch, clean_config):
        """Legacy behaviour: unset env vars allow any path (back-compat)."""
        monkeypatch.delenv("LOCAL_FILE_ACCESS_ROOT", raising=False)
        monkeypatch.delenv("LOCAL_FILE_ACCESS_ROOTS", raising=False)
        f = tmp_path / "anywhere.txt"
        f.write_text("ok")

        result = enforce_local_access_root(resolve_path(str(f)))
        assert result == f.resolve()
