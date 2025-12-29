"""Git-related helpers."""

from __future__ import annotations

import subprocess


def get_git_commit_id() -> str:
    """Return current git commit hash or 'unknown' if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    commit_id = result.stdout.strip()
    return commit_id or "unknown"
