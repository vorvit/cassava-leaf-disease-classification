"""Tests for cassava_leaf_disease.utils.seed."""

from __future__ import annotations

import os
import random

import pytest


def test_seed_everything_sets_env_and_random(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.utils.seed import seed_everything

    seed_everything(123)
    assert os.environ["PYTHONHASHSEED"] == "123"
    a = random.randint(0, 1_000_000)
    seed_everything(123)
    b = random.randint(0, 1_000_000)
    assert a == b
