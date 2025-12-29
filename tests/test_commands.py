from __future__ import annotations

import runpy
from types import SimpleNamespace

import pytest


def test_split_multirun_flag() -> None:
    from cassava_leaf_disease import commands

    assert commands._split_multirun_flag([]) == (False, [])
    assert commands._split_multirun_flag(["-m", "a=1"]) == (True, ["a=1"])
    assert commands._split_multirun_flag(["--multirun", "a=1"]) == (True, ["a=1"])
    assert commands._split_multirun_flag(["a=1"]) == (False, ["a=1"])


def test_expand_multirun_overrides_cartesian_product() -> None:
    from cassava_leaf_disease import commands

    expanded = commands._expand_multirun_overrides(
        ["train.lr=0.1,0.01", "train.batch_size=16,32", "logger.enabled=false"]
    )
    assert len(expanded) == 4
    assert all("logger.enabled=false" in combo for combo in expanded)
    assert any("train.lr=0.1" in combo and "train.batch_size=16" in combo for combo in expanded)
    assert any("train.lr=0.01" in combo and "train.batch_size=32" in combo for combo in expanded)


def test_sanitize_overrides_quotes_values_with_equals() -> None:
    from cassava_leaf_disease import commands

    overrides = [
        "infer.checkpoint_path=outputs/runs/version_9/checkpoints/epoch=0-step=16.ckpt",
        "logger.enabled=false",
    ]
    sanitized = commands._sanitize_overrides(overrides)
    assert sanitized[1] == "logger.enabled=false"
    assert sanitized[0].startswith("infer.checkpoint_path=")
    assert sanitized[0].count('"') >= 2


def test_main_dispatches_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    calls: list[str] = []

    def fake_train(args: list[str]) -> None:
        calls.append(f"train:{args!r}")

    def fake_infer(args: list[str]) -> None:
        calls.append(f"infer:{args!r}")

    def fake_download(args: list[str]) -> None:
        calls.append(f"download:{args!r}")

    def fake_download_model(args: list[str]) -> None:
        calls.append(f"download-model:{args!r}")

    monkeypatch.setattr(commands, "_run_train_or_multirun", fake_train)
    monkeypatch.setattr(commands, "_run_infer", fake_infer)
    monkeypatch.setattr(commands, "_run_download_data", fake_download)
    monkeypatch.setattr(commands, "_run_download_model", fake_download_model)

    commands.main(["train", "train.epochs=1"])
    commands.main(["infer", "infer.image_path=foo.jpg"])
    commands.main(["download-data"])
    commands.main(["download-model"])
    assert calls[0].startswith("train:")
    assert calls[1].startswith("infer:")
    assert calls[2].startswith("download:")
    assert calls[3].startswith("download-model:")


def test_run_download_model_success_prints_paths(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from cassava_leaf_disease import commands

    monkeypatch.setattr(commands, "compose_cfg", lambda *_a, **_k: SimpleNamespace())

    def fake_impl(_cfg):
        return SimpleNamespace(success=True, message="ok", checkpoint_path="artifacts/best.ckpt")

    monkeypatch.setattr(
        "cassava_leaf_disease.data.download_model.download_model_to_dvc",
        fake_impl,
        raising=False,
    )
    commands._run_download_model([])
    out = capsys.readouterr().out
    assert "[download-model] ok" in out
    assert "artifacts/best.ckpt" in out


def test_run_download_model_success_without_path(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from cassava_leaf_disease import commands

    monkeypatch.setattr(commands, "compose_cfg", lambda *_a, **_k: SimpleNamespace())

    def fake_impl(_cfg):
        return SimpleNamespace(success=True, message="ok", checkpoint_path=None)

    monkeypatch.setattr(
        "cassava_leaf_disease.data.download_model.download_model_to_dvc",
        fake_impl,
        raising=False,
    )
    commands._run_download_model([])
    out = capsys.readouterr().out
    assert "[download-model] ok" in out


def test_run_download_model_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    monkeypatch.setattr(commands, "compose_cfg", lambda *_a, **_k: SimpleNamespace())

    def fake_impl(_cfg):
        return SimpleNamespace(success=False, message="nope", checkpoint_path=None)

    monkeypatch.setattr(
        "cassava_leaf_disease.data.download_model.download_model_to_dvc",
        fake_impl,
        raising=False,
    )
    with pytest.raises(SystemExit, match="download-model failed"):
        commands._run_download_model([])


def test_compose_cfg_works_with_sanitized_override() -> None:
    from cassava_leaf_disease.commands import compose_cfg

    cfg = compose_cfg(
        "infer",
        [
            "infer.image_path=data/cassava/train_images/xxx.jpg",
            "infer.checkpoint_path=outputs/runs/version_9/checkpoints/epoch=0-step=16.ckpt",
            "logger.enabled=false",
        ],
    )
    # basic smoke: config is composed and contains expected keys
    assert getattr(cfg, "infer", None) is not None


def test_module_main_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make sure `python -m cassava_leaf_disease` path is covered.
    import cassava_leaf_disease.commands as commands

    called = SimpleNamespace(v=False)

    def fake_main(argv=None) -> None:
        called.v = True

    monkeypatch.setattr(commands, "main", fake_main)
    runpy.run_module("cassava_leaf_disease", run_name="__main__")
    assert called.v is True


def test_run_train_or_multirun_no_multirun(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    called = []

    def fake_run_train(overrides: list[str]) -> None:
        called.extend(overrides)

    monkeypatch.setattr(commands, "_run_train", fake_run_train)
    commands._run_train_or_multirun(["train.epochs=1"])
    assert called == ["train.epochs=1"]


def test_run_train_or_multirun_multirun(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    calls: list[list[str]] = []

    def fake_run_train(overrides: list[str]) -> None:
        calls.append(list(overrides))

    monkeypatch.setattr(commands, "_run_train", fake_run_train)
    commands._run_train_or_multirun(["-m", "train.lr=0.1,0.01"])

    assert len(calls) == 2
    assert all(any(o.startswith("hydra.run.dir=outputs/multirun/") for o in c) for c in calls)
    assert any("train.lr=0.1" in c for c in calls)
    assert any("train.lr=0.01" in c for c in calls)


def test_expand_multirun_overrides_no_equals(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    result = commands._expand_multirun_overrides(["+group=foo"])
    assert result == [["+group=foo"]]


def test_expand_multirun_overrides_literal_without_equals() -> None:
    from cassava_leaf_disease import commands

    result = commands._expand_multirun_overrides(["train"])
    assert result == [["train"]]


def test_expand_multirun_overrides_empty_values(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    result = commands._expand_multirun_overrides(["a=1,"])
    assert result == [["a=1"]]


def test_expand_multirun_overrides_only_empty_values() -> None:
    from cassava_leaf_disease import commands

    result = commands._expand_multirun_overrides(["a=,"])
    assert result == [["a=,"]]


def test_sanitize_overrides_no_equals(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    result = commands._sanitize_overrides(["+group=foo"])
    assert result == ["+group=foo"]


def test_sanitize_overrides_literal_no_equals() -> None:
    from cassava_leaf_disease import commands

    assert commands._sanitize_overrides(["train"]) == ["train"]


def test_ensure_utf8_stdio_no_reconfigure(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    class S:
        def __init__(self):
            self.called = False

    fake_out = S()
    fake_err = S()
    monkeypatch.setattr(commands.sys, "stdout", fake_out)
    monkeypatch.setattr(commands.sys, "stderr", fake_err)
    commands._ensure_utf8_stdio()
    # Should not crash even if reconfigure is missing


def test_main_empty_args(capsys) -> None:
    from cassava_leaf_disease import commands

    commands.main([])
    out = capsys.readouterr().out
    assert "Usage:" in out


def test_main_help_flag(capsys) -> None:
    from cassava_leaf_disease import commands

    commands.main(["--help"])
    out = capsys.readouterr().out
    assert "Commands:" in out

    commands.main(["-h"])
    out = capsys.readouterr().out
    assert "Commands:" in out

    commands.main(["help"])
    out = capsys.readouterr().out
    assert "Commands:" in out


def test_main_train_help(capsys) -> None:
    from cassava_leaf_disease import commands

    commands.main(["train", "--help"])
    out = capsys.readouterr().out
    assert "cassava_leaf_disease train" in out

    commands.main(["train", "-h"])
    out = capsys.readouterr().out
    assert "cassava_leaf_disease train" in out


def test_main_infer_help(capsys) -> None:
    from cassava_leaf_disease import commands

    commands.main(["infer", "help"])
    out = capsys.readouterr().out
    assert "cassava_leaf_disease infer" in out


def test_main_download_data_help(capsys) -> None:
    from cassava_leaf_disease import commands

    commands.main(["download-data", "-h"])
    out = capsys.readouterr().out
    assert "download-data" in out

    commands.main(["download_data", "--help"])
    out = capsys.readouterr().out
    assert "download-data" in out


def test_main_download_model_help(capsys) -> None:
    from cassava_leaf_disease import commands

    commands.main(["download-model", "--help"])
    out = capsys.readouterr().out
    assert "download-model" in out


def test_main_unknown_command() -> None:
    from cassava_leaf_disease import commands

    with pytest.raises(SystemExit):
        commands.main(["unknown"])


def test_wants_help() -> None:
    from cassava_leaf_disease import commands

    assert commands._wants_help(["-h"]) is True
    assert commands._wants_help(["--help"]) is True
    assert commands._wants_help(["help"]) is True
    assert commands._wants_help([]) is False
    assert commands._wants_help(["train"]) is False


def test_print_help_functions(capsys) -> None:
    from cassava_leaf_disease import commands

    commands._print_help()
    out = capsys.readouterr().out
    assert "Usage:" in out

    commands._print_train_help()
    out = capsys.readouterr().out
    assert "cassava_leaf_disease train" in out

    commands._print_infer_help()
    out = capsys.readouterr().out
    assert "cassava_leaf_disease infer" in out

    commands._print_download_data_help()
    out = capsys.readouterr().out
    assert "download-data" in out

    commands._print_download_model_help()
    out = capsys.readouterr().out
    assert "download-model" in out


def test_expand_multirun_overrides_with_comma(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    result = commands._expand_multirun_overrides(["train.lr=0.1,0.01"])
    assert len(result) == 2
    assert any("train.lr=0.1" in combo for combo in result)
    assert any("train.lr=0.01" in combo for combo in result)


def test_sanitize_overrides_already_quoted() -> None:
    from cassava_leaf_disease import commands

    result = commands._sanitize_overrides(['infer.checkpoint_path="a=b.ckpt"'])
    assert result == ['infer.checkpoint_path="a=b.ckpt"']


def test_sanitize_overrides_equals_in_value() -> None:
    from cassava_leaf_disease import commands

    result = commands._sanitize_overrides(["infer.checkpoint_path=epoch=0-step=16.ckpt"])
    assert result[0].startswith("infer.checkpoint_path=")
    assert '"' in result[0]


def test_compose_cfg_smoke() -> None:
    from cassava_leaf_disease.commands import compose_cfg

    cfg = compose_cfg("train", [])
    assert cfg is not None


def test_run_train_calls_train(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    called = []

    def fake_compose(name, overrides):
        called.append(("compose", name, overrides))
        return SimpleNamespace()

    def fake_train(cfg):
        called.append("train")

    monkeypatch.setattr(commands, "compose_cfg", fake_compose)
    monkeypatch.setattr("cassava_leaf_disease.training.train.train", fake_train)
    commands._run_train(["train.epochs=1"])
    assert called[0][0] == "compose"
    assert called[1] == "train"


def test_run_infer_calls_infer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    called = []

    def fake_compose(name, overrides):
        called.append(("compose", name, overrides))
        return SimpleNamespace()

    def fake_infer(cfg):
        called.append("infer")

    monkeypatch.setattr(commands, "compose_cfg", fake_compose)
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")
    monkeypatch.setattr(infer_mod, "infer", fake_infer)
    commands._run_infer(["infer.image_path=x.jpg"])
    assert called[0][0] == "compose"
    assert called[1] == "infer"


def test_run_download_data_calls_download(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from cassava_leaf_disease import commands

    def fake_compose(name, overrides):
        return SimpleNamespace()

    def fake_download(cfg):
        return SimpleNamespace(success=True, message="ok")

    monkeypatch.setattr(commands, "compose_cfg", fake_compose)
    monkeypatch.setattr("cassava_leaf_disease.data.download_data.download_data", fake_download)
    commands._run_download_data([])
    out = capsys.readouterr().out
    assert "[download-data]" in out


def test_run_download_data_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease import commands

    def fake_compose(name, overrides):
        return SimpleNamespace()

    def fake_download(cfg):
        return SimpleNamespace(success=False, message="error")

    monkeypatch.setattr(commands, "compose_cfg", fake_compose)
    monkeypatch.setattr("cassava_leaf_disease.data.download_data.download_data", fake_download)
    with pytest.raises(SystemExit, match="download-data failed"):
        commands._run_download_data([])
