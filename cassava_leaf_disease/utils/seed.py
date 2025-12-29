"""Reproducibility helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed python-side RNGs.

    Torch/Lightning seeding will be handled in the training module later.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

"""Reproducibility helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed python-side RNGs.

    Torch/Lightning seeding will be handled in the training module later.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)


