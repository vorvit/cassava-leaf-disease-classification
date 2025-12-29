"""Tests for helper functions in cassava_leaf_disease.training.train."""

from __future__ import annotations

import os

import pytest

from cassava_leaf_disease.training.train import (
    _ensure_s3_env_from_dotenv,
    _load_dotenv_file,
    _normalize_max_time,
    _strip_quotes,
)


def test_train_helper_strip_quotes() -> None:
    assert _strip_quotes("'value'") == "value"
    assert _strip_quotes('"value"') == "value"
    assert _strip_quotes("value") == "value"
    assert _strip_quotes("  'value'  ") == "value"
    assert _strip_quotes("'value") == "'value"
    assert _strip_quotes("value'") == "value'"


def test_train_helper_load_dotenv_file(tmp_path) -> None:
    dotenv_path = tmp_path / ".env"
    assert _load_dotenv_file(dotenv_path) == {}

    dotenv_path.write_text(
        "AWS_ACCESS_KEY_ID='key123'\n"
        'AWS_SECRET_ACCESS_KEY="secret456"\n'
        "# comment\n"
        "EMPTY=\n"
        "NO_EQUALS\n"
        "   =\n",
        encoding="utf-8",
    )
    result = _load_dotenv_file(dotenv_path)
    assert result["AWS_ACCESS_KEY_ID"] == "key123"
    assert result["AWS_SECRET_ACCESS_KEY"] == "secret456"
    assert "EMPTY" in result
    assert "NO_EQUALS" not in result
    assert "   " not in result


def test_train_helper_ensure_s3_env_from_dotenv(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    # Test empty dotenv
    _ensure_s3_env_from_dotenv(tmp_path)
    assert os.getenv("AWS_ACCESS_KEY_ID") is None

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "YOUR_ACCESS_KEY=test_key\nYOUR_SECRET_KEY=test_secret\n", encoding="utf-8"
    )

    _ensure_s3_env_from_dotenv(tmp_path)
    assert os.getenv("AWS_ACCESS_KEY_ID") == "test_key"
    assert os.getenv("AWS_SECRET_ACCESS_KEY") == "test_secret"

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    dotenv_path.write_text(
        "YC_ACCESS_KEY_ID=yc_key\nYC_SECRET_ACCESS_KEY=yc_secret\n", encoding="utf-8"
    )
    _ensure_s3_env_from_dotenv(tmp_path)
    assert os.getenv("AWS_ACCESS_KEY_ID") == "yc_key"
    assert os.getenv("AWS_SECRET_ACCESS_KEY") == "yc_secret"


def test_train_helper_normalize_max_time() -> None:
    assert _normalize_max_time(None) is None
    assert _normalize_max_time("null") is None
    assert _normalize_max_time("00:25:00") == "00:25:00"
    assert _normalize_max_time(25) == "00:25:00"
    assert _normalize_max_time(0) == "00:00:00"
    assert _normalize_max_time(-5) == "00:00:00"
    assert _normalize_max_time(125) == "02:05:00"
    assert _normalize_max_time(object()) is None
