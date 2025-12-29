"""CLI entrypoints.

Keep this module lightweight: no heavy ML imports at import time.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from contextlib import suppress


def main(argv: Sequence[str] | None = None) -> None:
    """Main CLI entrypoint.

    Real commands (train/infer/download-data) will be implemented in later steps.
    """
    _ensure_utf8_stdio()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    command = args[0]
    command_args = args[1:]

    if command == "train":
        if _wants_help(command_args):
            _print_train_help()
            return
        _run_train(command_args)
        return

    if command == "infer":
        if _wants_help(command_args):
            _print_infer_help()
            return
        _run_infer(command_args)
        return

    if command in {"download-data", "download_data"}:
        raise SystemExit("download-data is not implemented.")

    raise SystemExit(f"Unknown command: {command!r}. Try `python -m cassava_leaf_disease --help`.")


def _print_help() -> None:
    print(
        "\n".join(
            [
                "cassava_leaf_disease",
                "",
                "Usage:",
                "  python -m cassava_leaf_disease <command> [args...]",
                "",
                "Commands (will be implemented in Task2):",
                "  train          Train a model (runs DVC pull first)",
                "  infer          Run inference on an image (runs DVC pull first)",
            ]
        )
    )


def _wants_help(args: list[str]) -> bool:
    return any(a in {"-h", "--help", "help"} for a in args)


def _print_train_help() -> None:
    print(
        "\n".join(
            [
                "cassava_leaf_disease train",
                "",
                "Usage:",
                "  python -m cassava_leaf_disease train [hydra_overrides...]",
                "",
                "Examples:",
                "  python -m cassava_leaf_disease train",
                "  python -m cassava_leaf_disease train train.epochs=2 train.batch_size=32",
                "  python -m cassava_leaf_disease train data.synthetic.enabled=true "
                "logger.enabled=false",
            ]
        )
    )


def _print_infer_help() -> None:
    print(
        "\n".join(
            [
                "cassava_leaf_disease infer",
                "",
                "Usage:",
                "  python -m cassava_leaf_disease infer infer.image_path=... [hydra_overrides...]",
                "",
                "Examples:",
                "  python -m cassava_leaf_disease infer "
                "infer.image_path=data/cassava/train_images/xxx.jpg",
                "  python -m cassava_leaf_disease infer infer.image_path=... "
                "infer.checkpoint_path=outputs/.../best.ckpt",
                "  python -m cassava_leaf_disease infer infer.image_path=... logger.enabled=false",
            ]
        )
    )


def _run_train(overrides: list[str]) -> None:
    # Lazy imports to keep CLI startup fast.
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    # commands.py lives in `<repo>/cassava_leaf_disease/commands.py`
    # so `<repo>/configs` is `parents[1] / "configs"`.
    config_dir = (Path(__file__).resolve().parents[1] / "configs").resolve()

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=overrides)

    from cassava_leaf_disease.training.train import train

    train(cfg)


def _run_infer(overrides: list[str]) -> None:
    # Lazy imports to keep CLI startup fast.
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = (Path(__file__).resolve().parents[1] / "configs").resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="infer", overrides=overrides)

    from cassava_leaf_disease.serving.infer import infer

    infer(cfg)


def _ensure_utf8_stdio() -> None:
    """Ensure UTF-8 stdout/stderr on Windows consoles.

    Some third-party libs (e.g., MLflow) print unicode symbols to stdout.
    On certain Windows locales default encoding may be cp1251/cp866, causing crashes.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with suppress(Exception):
                reconfigure(encoding="utf-8")
