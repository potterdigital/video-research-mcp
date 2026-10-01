"""Tests for local_path_policy — multi-root access boundaries."""

from __future__ import annotations

import os

import pytest

from video_research_mcp.local_path_policy import (
    SensitivePathError,
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


class TestSensitivePathDenyList:
    """Sensitive paths are denied even inside an allowed root (defence in depth).

    Each test names the specific bug it guards against so a future reader can
    verify the test still earns its keep.
    """

    def _allow_tmp(self, tmp_path, monkeypatch):
        """Configure the server so *tmp_path* is an allowed access root."""
        monkeypatch.setenv("LOCAL_FILE_ACCESS_ROOTS", str(tmp_path))

    @pytest.mark.parametrize(
        "subpath, basename",
        [
            (".ssh", "id_rsa"),
            (".ssh", "id_ed25519"),
            (".aws", "credentials"),
            (".gnupg", "secring.gpg"),
            (".docker", "config.json"),
            (".kube", "config"),
            (".config/gh", "hosts.yml"),
            (".config/git", "credentials"),
        ],
    )
    def test_sensitive_directory_part_denied(
        self, tmp_path, monkeypatch, clean_config, subpath, basename
    ):
        # Guards: a malicious or accidental tool call on common credential
        # directories must not succeed even when the parent dir is allowed.
        target_dir = tmp_path / subpath
        target_dir.mkdir(parents=True)
        f = target_dir / basename
        f.write_text("secret")
        self._allow_tmp(tmp_path, monkeypatch)

        with pytest.raises(SensitivePathError, match="sensitive-path deny-list"):
            enforce_local_access_root(resolve_path(str(f)))

    @pytest.mark.parametrize(
        "filename",
        [
            "server.pem",
            "private.key",
            "id_rsa",
            "id_ed25519.pub",
            "id_ecdsa",
            "id_dsa",
            "vault.kdbx",
            "secret.gpg",
            "cert.p12",
            "cert.pfx",
            "credentials",
            "credentials.json",
            ".env",
            ".env.production",
            ".netrc",
            ".npmrc",
            ".pypirc",
            "htpasswd",
            "release.keystore",
            "android.jks",
        ],
    )
    def test_sensitive_filename_pattern_denied(
        self, tmp_path, monkeypatch, clean_config, filename
    ):
        # Guards: even a sensitive filename in an otherwise-allowed directory
        # must trigger the deny-list (not just paths under ~/.ssh).
        f = tmp_path / filename
        f.write_text("secret")
        self._allow_tmp(tmp_path, monkeypatch)

        with pytest.raises(SensitivePathError):
            enforce_local_access_root(resolve_path(str(f)))

    def test_sensitive_path_denied_even_when_no_roots_configured(
        self, tmp_path, monkeypatch, clean_config
    ):
        # Guards: the deny-list is a hard floor — applies even when the
        # operator forgot to set LOCAL_FILE_ACCESS_ROOTS at all.
        monkeypatch.delenv("LOCAL_FILE_ACCESS_ROOT", raising=False)
        monkeypatch.delenv("LOCAL_FILE_ACCESS_ROOTS", raising=False)
        target_dir = tmp_path / ".ssh"
        target_dir.mkdir()
        f = target_dir / "id_rsa"
        f.write_text("secret")

        with pytest.raises(SensitivePathError):
            enforce_local_access_root(resolve_path(str(f)))

    @pytest.mark.parametrize(
        "filename",
        [
            "notes-about-ssh.txt",  # contains 'ssh' but not a sensitive part
            "my.envelope.txt",  # starts with '.env' substring but not pattern
            "keystore-notes.md",  # contains 'keystore' but no .keystore ext
            "credit-report.pdf",  # contains 'cred' but not 'credentials*'
            "readme.md",
            "video.mp4",
        ],
    )
    def test_benign_filenames_allowed(
        self, tmp_path, monkeypatch, clean_config, filename
    ):
        # Guards: deny-list must not over-match. A future change that uses
        # naive substring matching ("env" in name) would fail here.
        f = tmp_path / filename
        f.write_text("ok")
        self._allow_tmp(tmp_path, monkeypatch)

        result = enforce_local_access_root(resolve_path(str(f)))
        assert result == f.resolve()

    def test_sensitive_error_is_permissionerror_subclass(self):
        # Guards: existing call sites catch PermissionError; the deny-list
        # error must remain a subclass so generic handlers still work.
        assert issubclass(SensitivePathError, PermissionError)

    def test_sensitive_error_categorizes_distinctly(self):
        # Guards: SENSITIVE_PATH_DENIED maps to its own ErrorCategory rather
        # than collapsing into the generic PERMISSION_DENIED bucket.
        from video_research_mcp.errors import ErrorCategory, categorize_error

        cat, _ = categorize_error(SensitivePathError("test"))
        assert cat is ErrorCategory.SENSITIVE_PATH_DENIED
