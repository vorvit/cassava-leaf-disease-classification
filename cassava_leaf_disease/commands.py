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
        _run_train(command_args)
        return

    if command == "infer":
        raise SystemExit("infer is not implemented yet.")

    if command in {"download-data", "download_data"}:
        raise SystemExit("download-data is not implemented yet.")

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
                "  download-data  Optional: download data from public sources",
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
