"""Tests for cassava_leaf_disease.data.download_data."""

from __future__ import annotations

import io
import tarfile
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest


def test_filename_from_content_disposition() -> None:
    from cassava_leaf_disease.data import download_data as mod

    assert mod._filename_from_content_disposition(None) is None
    assert mod._filename_from_content_disposition("attachment") is None
    assert (
        mod._filename_from_content_disposition('attachment; filename="dataset.rar"')
        == "dataset.rar"
    )
    assert mod._filename_from_content_disposition("attachment; filename='test.zip'") == "test.zip"


def test_infer_filename_from_url_fallbacks() -> None:
    from cassava_leaf_disease.data import download_data as mod

    assert mod._infer_filename_from_url("https://example.com/path/file.zip") == "file.zip"
    assert mod._infer_filename_from_url("https://example.com/") == "dataset.rar"
    assert mod._infer_filename_from_url("https://example.com/?filename=a.rar") == "a.rar"


def test_is_rarfile_detects_magic_bytes(tmp_path) -> None:
    from cassava_leaf_disease.data import download_data as mod

    p = tmp_path / "x.rar"
    p.write_bytes(b"Rar!\x1a\x07\x01\x00xxxx")
    assert mod._is_rarfile(p) is True

    p2 = tmp_path / "x.zip"
    p2.write_bytes(b"PK\x03\x04")
    assert mod._is_rarfile(p2) is False


def test_extract_archive_zip_moves_contents(tmp_path) -> None:
    from cassava_leaf_disease.data import download_data as mod

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    archive = tmp_path / "ds.zip"

    # Create zip with a single root folder to exercise _pick_extracted_root.
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("bundle/train.csv", "image_id,label\nx.jpg,0\n")

    mod._extract_archive(archive, data_dir=data_dir, force=True)
    assert (data_dir / "train.csv").exists()


def test_extract_archive_tar(tmp_path) -> None:
    from cassava_leaf_disease.data import download_data as mod

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    archive = tmp_path / "ds.tar"

    payload = b"image_id,label\nx.jpg,0\n"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo(name="bundle/train.csv")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    mod._extract_archive(archive, data_dir=data_dir, force=True)
    assert (data_dir / "train.csv").exists()


def test_extract_archive_rar_errors_without_backend(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cassava_leaf_disease.data import download_data as mod

    archive = tmp_path / "ds.rar"
    archive.write_bytes(b"Rar!\x1a\x07\x01\x00xxxx")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    monkeypatch.setattr(mod.shutil, "which", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="RAR archive detected"):
        mod._extract_archive(archive, data_dir=data_dir, force=True)


def test_extract_archive_rar_via_rarfile(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import cassava_leaf_disease.data.download_data as mod

    rar = tmp_path / "x.rar"
    rar.write_bytes(b"Rar!\x1a\x07\x01\x00xxxx")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    class RarFile:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def extractall(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "train.csv").write_text("x", encoding="utf-8")

    fake = types.SimpleNamespace(RarFile=RarFile)
    monkeypatch.setitem(__import__("sys").modules, "rarfile", fake)
    mod._extract_archive(rar, data_dir=data_dir, force=True)
    assert (data_dir / "train.csv").exists()


def test_extract_archive_rar_via_7z_failure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import cassava_leaf_disease.data.download_data as mod

    rar = tmp_path / "x.rar"
    rar.write_bytes(b"Rar!\x1a\x07\x01\x00xxxx")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    monkeypatch.setattr(mod.shutil, "which", lambda *_a, **_k: "7z")

    class R:
        returncode = 1
        stdout = "out"
        stderr = "err"

    monkeypatch.setattr(mod.subprocess, "run", lambda *_a, **_k: R())
    with pytest.raises(ValueError, match="Failed to extract RAR archive via 7-Zip"):
        mod._extract_archive(rar, data_dir=data_dir, force=True)


def test_download_data_happy_path_with_patched_network(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cassava_leaf_disease.data import download_data as mod
    from cassava_leaf_disease.data.download_data import download_data

    def fake_href(_public_url: str) -> str:
        return "https://example.com/dataset.zip"

    def fake_download(_url: str, dst):
        dst = tmp_path / "dataset.zip"
        with zipfile.ZipFile(dst, "w") as zf:
            zf.writestr("train.csv", "image_id,label\nx.jpg,0\n")
            zf.writestr("train_images/x.jpg", b"fake")
        return dst

    monkeypatch.setattr(mod, "_get_yadisk_download_href", fake_href)
    monkeypatch.setattr(mod, "_download_file", fake_download)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(tmp_path / "cassava")),
        download_data=SimpleNamespace(public_url="https://disk.yandex.ru/d/x", force=True),
    )
    result = download_data(cfg)
    assert result.success is True
    assert (tmp_path / "cassava" / "train.csv").exists()


def test_download_data_dataset_exists_short_circuit(tmp_path) -> None:
    from cassava_leaf_disease.data.download_data import download_data

    data_dir = tmp_path / "cassava"
    (data_dir / "train_images").mkdir(parents=True)
    (data_dir / "train.csv").write_text("x", encoding="utf-8")
    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(data_dir)),
        download_data=SimpleNamespace(public_url="https://disk.yandex.ru/d/x", force=False),
    )
    result = download_data(cfg)
    assert result.success is True
    assert "already exists" in result.message


def test_download_data_error_return(monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.data.download_data as mod
    from cassava_leaf_disease.data.download_data import download_data

    monkeypatch.setattr(
        mod,
        "_download_and_extract_yadisk_public",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir="x"),
        download_data=SimpleNamespace(public_url="https://disk.yandex.ru/d/x", force=True),
    )
    res = download_data(cfg)
    assert res.success is False
    assert "boom" in res.message


def test_download_data_get_href_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import download_data as mod

    class Resp:
        headers: ClassVar[dict[str, str]] = {}

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(mod, "urlopen", lambda *_a, **_k: Resp())
    with pytest.raises(ValueError, match="missing href"):
        mod._get_yadisk_download_href("https://disk.yandex.ru/d/x")


def test_download_file_uses_content_disposition_filename(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cassava_leaf_disease.data import download_data as mod

    class Resp:
        def __init__(self):
            self.headers = {"Content-Disposition": 'attachment; filename="x.rar"'}
            self._chunks = [b"abc", b""]

        def read(self, _n):
            return self._chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(mod, "urlopen", lambda *_a, **_k: Resp())
    dst = tmp_path / "fallback.rar"
    out = mod._download_file("https://example.com", dst)
    assert out.name == "x.rar"
    assert out.exists()


def test_move_tree_contents_force(tmp_path) -> None:
    from cassava_leaf_disease.data import download_data as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "file.txt").write_text("new")
    (dst / "file.txt").write_text("old")
    mod._move_tree_contents(src, dst, force=True)
    assert (dst / "file.txt").read_text() == "new"


def test_download_and_extract_archive_exists(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import download_data as mod

    downloads_dir = tmp_path / "_downloads"
    downloads_dir.mkdir()
    archive = downloads_dir / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train.csv", "image_id,label\nx.jpg,0\n")

    def fake_href(_url):
        return "https://example.com/dataset.zip"

    def fake_download(_url, dst):
        return archive

    monkeypatch.setattr(mod, "_get_yadisk_download_href", fake_href)
    monkeypatch.setattr(mod, "_download_file", fake_download)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    mod._download_and_extract_yadisk_public("https://disk.yandex.ru/d/x", data_dir, force=False)


def test_download_file_no_content_disposition(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import download_data as mod

    class Resp:
        def __init__(self):
            self.headers = {}
            self._chunks = [b"abc", b""]

        def read(self, _n):
            return self._chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(mod, "urlopen", lambda *_a, **_k: Resp())
    dst = tmp_path / "file.zip"
    out = mod._download_file("https://example.com", dst)
    assert out == dst
    assert out.exists()


def test_filename_from_content_disposition_single_quote() -> None:
    from cassava_leaf_disease.data import download_data as mod

    assert mod._filename_from_content_disposition("attachment; filename='test.zip'") == "test.zip"


def test_extract_archive_unsupported_format(tmp_path) -> None:
    from cassava_leaf_disease.data import download_data as mod

    archive = tmp_path / "bad.bin"
    archive.write_bytes(b"invalid")

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(ValueError, match="Unsupported archive format"):
        mod._extract_archive(archive, data_dir=data_dir, force=True)


def test_is_rarfile_exception(tmp_path) -> None:
    from cassava_leaf_disease.data import download_data as mod

    # Non-existent file
    p = tmp_path / "missing.rar"
    assert mod._is_rarfile(p) is False


def test_extract_rar_no_rarfile_no_7z(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import download_data as mod

    rar = tmp_path / "x.rar"
    rar.write_bytes(b"Rar!\x1a\x07\x01\x00xxxx")
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()

    def fake_rarfile(*_a, **_k):
        raise ImportError("rarfile missing")

    monkeypatch.setitem(__import__("sys").modules, "rarfile", None)
    monkeypatch.setattr(mod.shutil, "which", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="RAR archive detected"):
        mod._extract_rar(rar, dst_dir)


def test_download_and_extract_archive_force(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import download_data as mod

    downloads_dir = tmp_path / "_downloads"
    downloads_dir.mkdir()
    archive = downloads_dir / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train.csv", "image_id,label\nx.jpg,0\n")

    def fake_href(_url):
        return "https://example.com/dataset.zip"

    def fake_download(_url, dst):
        return archive

    monkeypatch.setattr(mod, "_get_yadisk_download_href", fake_href)
    monkeypatch.setattr(mod, "_download_file", fake_download)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    mod._download_and_extract_yadisk_public("https://disk.yandex.ru/d/x", data_dir, force=True)


def test_get_yadisk_download_href_invalid_href(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import download_data as mod

    class Resp:
        def read(self):
            return b'{"href": ""}'

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(mod, "urlopen", lambda *_a, **_k: Resp())
    with pytest.raises(ValueError, match="missing href"):
        mod._get_yadisk_download_href("https://disk.yandex.ru/d/x")


def test_download_file_dst_unlink(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import download_data as mod

    class Resp:
        def __init__(self):
            self.headers = {"Content-Disposition": 'attachment; filename="x.rar"'}
            self._chunks = [b"abc", b""]

        def read(self, _n):
            return self._chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(mod, "urlopen", lambda *_a, **_k: Resp())
    dst = tmp_path / "fallback.rar"
    dst.write_bytes(b"old")
    out = mod._download_file("https://example.com", dst)
    assert out.name == "x.rar"
    assert not dst.exists()


def test_extract_archive_tmp_dir_cleanup(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import download_data as mod

    archive = tmp_path / "ds.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train.csv", "image_id,label\nx.jpg,0\n")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tmp_extract = data_dir / "_tmp_extract"
    tmp_extract.mkdir()
    (tmp_extract / "old.txt").write_text("old")

    mod._extract_archive(archive, data_dir=data_dir, force=True)
    assert not tmp_extract.exists()
    assert (data_dir / "train.csv").exists()


def test_extract_rar_via_7z_success(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import cassava_leaf_disease.data.download_data as mod

    rar = tmp_path / "x.rar"
    rar.write_bytes(b"Rar!\x1a\x07\x01\x00xxxx")
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()

    def fake_rarfile(*_a, **_k):
        raise ImportError("rarfile missing")

    monkeypatch.setitem(__import__("sys").modules, "rarfile", None)
    monkeypatch.setattr(mod.shutil, "which", lambda *_a, **_k: "7z")

    class R:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *_a, **_k: R())
    # Should not raise
    mod._extract_rar(rar, dst_dir)


def test_move_tree_contents_no_force(tmp_path) -> None:
    from cassava_leaf_disease.data import download_data as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "file.txt").write_text("new")
    (dst / "file.txt").write_text("old")
    mod._move_tree_contents(src, dst, force=False)
    assert (dst / "file.txt").read_text() == "old"
