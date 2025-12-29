"""Tests for cassava_leaf_disease.utils.git."""

from __future__ import annotations

import pytest


def test_get_git_commit_id_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.utils import git

    class R:
        stdout = "abc123\n"

    monkeypatch.setattr(git.subprocess, "run", lambda *a, **k: R())
    assert git.get_git_commit_id() == "abc123"


def test_get_git_commit_id_unknown_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.utils import git

    def boom(*_a, **_k):
        raise RuntimeError("no git")

    monkeypatch.setattr(git.subprocess, "run", boom)
    assert git.get_git_commit_id() == "unknown"


def test_get_git_commit_id_empty_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.utils import git

    class R:
        stdout = ""

    monkeypatch.setattr(git.subprocess, "run", lambda *a, **k: R())
    assert git.get_git_commit_id() == "unknown"


def test_get_git_commit_id_check_false(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.utils import git

    class R:
        stdout = "abc123\n"
        returncode = 1

    def fake_run(*args, **kwargs):
        r = R()
        raise subprocess.CalledProcessError(1, "git", r.stdout)

    import subprocess

    monkeypatch.setattr(git.subprocess, "run", fake_run)
    assert git.get_git_commit_id() == "unknown"
