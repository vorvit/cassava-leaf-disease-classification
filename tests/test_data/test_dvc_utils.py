"""Tests for cassava_leaf_disease.data.dvc_utils."""

from __future__ import annotations

import builtins
import types
from typing import Any

import pytest


def test_dvc_pull_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import dvc_utils

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("dvc"):
            raise ImportError("missing dvc")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = dvc_utils.dvc_pull(targets=["data/cassava"])
    assert result.success is False
    assert "Failed to import DVC" in result.message


def test_dvc_pull_success_with_fake_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import dvc_utils

    pulled: list[object] = []

    class Repo:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def pull(self, targets=None):
            pulled.append(targets)

    fake_dvc_repo: Any = types.ModuleType("dvc.repo")
    fake_dvc_repo.Repo = Repo
    fake_dvc: Any = types.ModuleType("dvc")

    monkeypatch.setitem(__import__("sys").modules, "dvc", fake_dvc)
    monkeypatch.setitem(__import__("sys").modules, "dvc.repo", fake_dvc_repo)

    result = dvc_utils.dvc_pull(targets=["x"])
    assert result.success is True
    assert pulled == [["x"]]


def test_dvc_pull_failure_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import dvc_utils

    class Repo:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def pull(self, targets=None):
            raise RuntimeError("pull failed")

    fake_dvc_repo: Any = types.ModuleType("dvc.repo")
    fake_dvc_repo.Repo = Repo
    fake_dvc: Any = types.ModuleType("dvc")

    monkeypatch.setitem(__import__("sys").modules, "dvc", fake_dvc)
    monkeypatch.setitem(__import__("sys").modules, "dvc.repo", fake_dvc_repo)

    result = dvc_utils.dvc_pull(targets=["x"])
    assert result.success is False
    assert "pull failed" in result.message


def test_data_init_exports() -> None:
    from cassava_leaf_disease import data

    assert hasattr(data, "dvc_pull")
    assert hasattr(data, "DvcPullResult")


def test_dvc_pull_none_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import dvc_utils

    pulled: list[object] = []

    class Repo:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def pull(self, targets=None):
            pulled.append(targets)

    fake_dvc_repo: Any = types.ModuleType("dvc.repo")
    fake_dvc_repo.Repo = Repo
    fake_dvc: Any = types.ModuleType("dvc")

    monkeypatch.setitem(__import__("sys").modules, "dvc", fake_dvc)
    monkeypatch.setitem(__import__("sys").modules, "dvc.repo", fake_dvc_repo)

    result = dvc_utils.dvc_pull(targets=None)
    assert result.success is True
    assert pulled == [None]
