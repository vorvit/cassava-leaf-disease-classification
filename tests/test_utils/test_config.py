"""Tests for cassava_leaf_disease.utils.config."""

from __future__ import annotations

import builtins

import pytest


def test_get_default_class_names_reads_repo_config() -> None:
    from cassava_leaf_disease.utils.config import get_default_class_names

    names = get_default_class_names()
    assert isinstance(names, list)
    assert len(names) == 5
    assert "Healthy" in names


def test_utils_config_fallback_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.utils.config as cfg_mod

    # config file missing -> default
    monkeypatch.setattr(cfg_mod.Path, "exists", lambda *_a, **_k: False)
    assert cfg_mod.get_default_class_names()[-1] == "Healthy"


def test_utils_config_omegaconf_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.utils.config as cfg_mod

    # make config_path exist, but make OmegaConf import fail
    monkeypatch.setattr(cfg_mod.Path, "exists", lambda *_a, **_k: True)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "omegaconf":
            raise ImportError("nope")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert cfg_mod.get_default_class_names()[0].startswith("Cassava")


def test_utils_config_empty_class_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from omegaconf import OmegaConf

    import cassava_leaf_disease.utils.config as cfg_mod

    monkeypatch.setattr(cfg_mod.Path, "exists", lambda *_a, **_k: True)
    monkeypatch.setattr(
        OmegaConf,
        "load",
        lambda *_a, **_k: OmegaConf.create({"dataset": {"class_names": []}}),
    )
    names = cfg_mod.get_default_class_names()
    assert len(names) == 5
    assert "Healthy" in names


def test_utils_config_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from omegaconf import OmegaConf

    import cassava_leaf_disease.utils.config as cfg_mod

    monkeypatch.setattr(cfg_mod.Path, "exists", lambda *_a, **_k: True)
    monkeypatch.setattr(
        OmegaConf, "load", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad"))
    )
    names = cfg_mod.get_default_class_names()
    assert len(names) == 5
