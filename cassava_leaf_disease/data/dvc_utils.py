"""DVC integration helpers.

Task2 requirement: data must be managed via DVC, and `train`/`infer` should be able
to fetch required artifacts automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DvcPullResult:
    """Result of a DVC pull attempt."""

    success: bool
    message: str


def dvc_pull(targets: list[str] | None = None, repo_dir: str | Path = ".") -> DvcPullResult:
    """Run DVC pull via Python API.

    We intentionally use `subprocess`-free API where possible to stay portable.
    """
    try:
        from dvc.repo import Repo  # lazy import
    except Exception as exc:  # pragma: no cover
        return DvcPullResult(success=False, message=f"Failed to import DVC: {exc}")

    try:
        with Repo(str(repo_dir)) as repo:
            repo.pull(targets=targets)
    except Exception as exc:
        return DvcPullResult(success=False, message=str(exc))
    return DvcPullResult(success=True, message="ok")
