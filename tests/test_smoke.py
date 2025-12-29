from __future__ import annotations


def test_import_package() -> None:
    import cassava_leaf_disease  # noqa: F401


def test_import_training_entrypoint() -> None:
    from cassava_leaf_disease.training.train import train  # noqa: F401
