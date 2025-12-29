"""Tests for cassava_leaf_disease.fire_cli."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_normalize_fire_args_supports_aliases() -> None:
    from cassava_leaf_disease.fire_cli import _normalize_fire_args

    args = _normalize_fire_args(["download-data", "--no-mlflow", "--image-path", "x.jpg"])
    assert args[0] == "download_data"
    assert "--mlflow=false" in args
    assert "--image" in args


def test_fire_train_calls_compose_and_train(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.fire_cli import CassavaFireCLI

    called = SimpleNamespace(cfg=None)

    def fake_compose(name: str, overrides: list[str]):
        called.cfg = (name, overrides)
        return SimpleNamespace()

    def fake_train_impl(cfg):
        called.train_cfg = cfg

    monkeypatch.setattr("cassava_leaf_disease.commands.compose_cfg", fake_compose, raising=False)
    monkeypatch.setattr("cassava_leaf_disease.training.train.train", fake_train_impl, raising=False)

    cli = CassavaFireCLI()
    cli.train(epochs=1, synthetic=True, mlflow=False)
    assert called.cfg[0] == "train"
    assert "train.epochs=1" in called.cfg[1]
    assert "data.synthetic.enabled=true" in called.cfg[1]
    assert "logger.enabled=false" in called.cfg[1]


def test_fire_train_all_params(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.fire_cli import CassavaFireCLI

    got = SimpleNamespace(name=None, overrides=None)

    def fake_compose(name, overrides):
        got.name = name
        got.overrides = overrides
        return SimpleNamespace()

    monkeypatch.setattr("cassava_leaf_disease.commands.compose_cfg", fake_compose, raising=False)
    monkeypatch.setattr("cassava_leaf_disease.training.train.train", lambda *_a, **_k: None)

    cli = CassavaFireCLI()
    # Fire doesn't support mixing *args with keyword args in the same call
    # So we test overrides separately
    cli.train(
        epochs=2,
        batch_size=32,
        lr=0.001,
        precision="16-mixed",
        synthetic=False,
        mlflow=True,
        num_workers=4,
    )
    assert got.name == "train"
    assert "train.epochs=2" in got.overrides
    assert "train.batch_size=32" in got.overrides
    assert "train.lr=0.001" in got.overrides
    assert "train.precision=16-mixed" in got.overrides
    assert "data.synthetic.enabled=false" in got.overrides
    assert "logger.enabled=true" in got.overrides
    assert "train.num_workers=4" in got.overrides


def test_fire_infer_all_params(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.fire_cli import CassavaFireCLI

    got = SimpleNamespace(name=None, overrides=None)

    def fake_compose(name, overrides):
        got.name = name
        got.overrides = overrides
        return SimpleNamespace()

    monkeypatch.setattr("cassava_leaf_disease.commands.compose_cfg", fake_compose, raising=False)
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")
    monkeypatch.setattr(infer_mod, "infer", lambda *_a, **_k: None)
    cli = CassavaFireCLI()
    cli.infer(image="x.jpg", ckpt="c.ckpt", device="cuda", top_k=2)
    assert got.name == "infer"
    assert "infer.image_path=x.jpg" in got.overrides
    assert "infer.checkpoint_path=c.ckpt" in got.overrides
    assert "infer.device=cuda" in got.overrides
    assert "infer.top_k=2" in got.overrides


def test_fire_download_data(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from cassava_leaf_disease.fire_cli import CassavaFireCLI

    def fake_compose(name: str, overrides: list[str]):
        return SimpleNamespace()

    def fake_download(cfg):
        return SimpleNamespace(success=True, message="ok")

    monkeypatch.setattr("cassava_leaf_disease.commands.compose_cfg", fake_compose, raising=False)
    monkeypatch.setattr(
        "cassava_leaf_disease.data.download_data.download_data", fake_download, raising=False
    )

    cli = CassavaFireCLI()
    cli.download_data(force=True)
    out = capsys.readouterr().out
    assert "download-data" in out


def test_fire_download_data_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.fire_cli import CassavaFireCLI

    def fake_compose(name: str, overrides: list[str]):
        return SimpleNamespace()

    def fake_download(cfg):
        return SimpleNamespace(success=False, message="error")

    monkeypatch.setattr("cassava_leaf_disease.commands.compose_cfg", fake_compose, raising=False)
    monkeypatch.setattr(
        "cassava_leaf_disease.data.download_data.download_data", fake_download, raising=False
    )

    cli = CassavaFireCLI()
    with pytest.raises(SystemExit, match="download-data failed"):
        cli.download_data(force=True)


def test_fire_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.fire_cli import CassavaFireCLI

    called = []

    def fake_main(args):
        called.extend(args)

    monkeypatch.setattr("cassava_leaf_disease.commands.main", fake_main)
    cli = CassavaFireCLI()
    cli.raw("train", "train.epochs=1")
    assert called == ["train", "train.epochs=1"]


def test_parse_bool() -> None:
    from cassava_leaf_disease.fire_cli import _parse_bool

    assert _parse_bool(True) is True
    assert _parse_bool(False) is False
    assert _parse_bool(1) is True
    assert _parse_bool(0) is False
    assert _parse_bool("true") is True
    assert _parse_bool("false") is False
    assert _parse_bool("1") is True
    assert _parse_bool("0") is False
    assert _parse_bool("yes") is True
    assert _parse_bool("no") is False
    assert _parse_bool("on") is True
    assert _parse_bool("off") is False
    with pytest.raises(ValueError):
        _parse_bool(None)
    with pytest.raises(ValueError):
        _parse_bool(object())


def test_bool_override() -> None:
    from cassava_leaf_disease.fire_cli import _bool_override

    assert _bool_override(True) == "true"
    assert _bool_override(False) == "false"


def test_parse_bool_unsupported_string() -> None:
    from cassava_leaf_disease.fire_cli import _parse_bool

    with pytest.raises(ValueError, match="Unsupported boolean value"):
        _parse_bool("maybe")


def test_fire_train_none_params(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.fire_cli import CassavaFireCLI

    got = SimpleNamespace(overrides=None)

    def fake_compose(name, overrides):
        got.overrides = overrides
        return SimpleNamespace()

    monkeypatch.setattr("cassava_leaf_disease.commands.compose_cfg", fake_compose, raising=False)
    monkeypatch.setattr("cassava_leaf_disease.training.train.train", lambda *_a, **_k: None)

    cli = CassavaFireCLI()
    cli.train()
    assert got.overrides == []


def test_fire_infer_none_params(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.fire_cli import CassavaFireCLI

    got = SimpleNamespace(overrides=None)

    def fake_compose(name, overrides):
        got.overrides = overrides
        return SimpleNamespace()

    monkeypatch.setattr("cassava_leaf_disease.commands.compose_cfg", fake_compose, raising=False)
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")
    monkeypatch.setattr(infer_mod, "infer", lambda *_a, **_k: None)

    cli = CassavaFireCLI()
    cli.infer()
    assert got.overrides == []


def test_fire_download_data_none_force(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.fire_cli import CassavaFireCLI

    got = SimpleNamespace(overrides=None)

    def fake_compose(name, overrides):
        got.overrides = overrides
        return SimpleNamespace()

    def fake_download(cfg):
        return SimpleNamespace(success=True, message="ok")

    monkeypatch.setattr("cassava_leaf_disease.commands.compose_cfg", fake_compose, raising=False)
    monkeypatch.setattr(
        "cassava_leaf_disease.data.download_data.download_data", fake_download, raising=False
    )

    cli = CassavaFireCLI()
    cli.download_data()
    assert got.overrides == []


def test_fire_main(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.fire_cli import main

    called = []

    def fake_fire(*args, **kwargs):
        called.append((args, kwargs))

    monkeypatch.setattr("fire.Fire", fake_fire)
    main(["train", "--epochs", "1"])
    assert len(called) == 1
    from cassava_leaf_disease.fire_cli import CassavaFireCLI

    got = SimpleNamespace(overrides=None)

    def fake_compose(name, overrides):
        got.overrides = overrides
        return SimpleNamespace()

    def fake_download(cfg):
        return SimpleNamespace(success=True, message="ok")

    monkeypatch.setattr("cassava_leaf_disease.commands.compose_cfg", fake_compose, raising=False)
    monkeypatch.setattr(
        "cassava_leaf_disease.data.download_data.download_data", fake_download, raising=False
    )

    cli = CassavaFireCLI()
    cli.download_data()
    assert got.overrides == []
