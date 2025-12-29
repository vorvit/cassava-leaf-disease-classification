"""CLI entrypoints.

Keep this module lightweight: no heavy ML imports at import time.
"""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Main CLI entrypoint.

    Real commands (train/infer/download-data) will be implemented in later steps.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    command = args[0]
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

"""CLI entrypoints.

This module intentionally avoids any heavy imports at module import time.
Executable logic should live inside functions and be called from `main()`.
"""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Main CLI entrypoint.

    This will be wired to real commands (train/infer/download-data) in later steps.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    command = args[0]
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


