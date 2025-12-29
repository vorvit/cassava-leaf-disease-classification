"""Training entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def _load_dotenv_file(dotenv_path: Path) -> dict[str, str]:
    """Load key/value pairs from a local .env file (no shell expansion)."""
    if not dotenv_path.exists():
        return {}

    result: dict[str, str] = {}
    text = dotenv_path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        result[key] = _strip_quotes(value)
    return result


def _ensure_s3_env_from_dotenv(repo_root: Path) -> None:
    """Populate AWS/YC env vars from `.env` if present (without printing secrets)."""
    import os

    dotenv = _load_dotenv_file(repo_root / ".env")
    if not dotenv:
        return

    candidates = {
        "AWS_ACCESS_KEY_ID": [
            "AWS_ACCESS_KEY_ID",
            "YC_ACCESS_KEY_ID",
            "YOUR_ACCESS_KEY",
            "YOUR_ACCESS_KEY_ID",
        ],
        "AWS_SECRET_ACCESS_KEY": [
            "AWS_SECRET_ACCESS_KEY",
            "YC_SECRET_ACCESS_KEY",
            "YOUR_SECRET_KEY",
            "YOUR_SECRET_ACCESS_KEY",
        ],
    }
    for canonical_key, aliases in candidates.items():
        if os.getenv(canonical_key):
            continue
        for alias in aliases:
            value = dotenv.get(alias)
            if value:
                os.environ[canonical_key] = value
                break


def _normalize_max_time(value: object) -> str | None:
    """Normalize Trainer max_time from config values."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "null":
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        total_minutes = int(value)
        total_minutes = max(0, total_minutes)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}:00"
    return None


def train(cfg: Any) -> None:
    """Run training using Hydra-composed config."""
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.loggers import CSVLogger, Logger

    from cassava_leaf_disease.data import dvc_pull
    from cassava_leaf_disease.training.datamodule import CassavaDataModule
    from cassava_leaf_disease.training.lightning_module import CassavaClassifier
    from cassava_leaf_disease.utils.git import get_git_commit_id

    pl.seed_everything(int(cfg.train.seed), workers=True)

    # Try to pull data (if tracked by DVC). On failure we can still run synthetic fallback.
    data_dir = str(cfg.paths.data_dir)
    pull_result = dvc_pull(targets=[data_dir])
    if not pull_result.success:
        print(f"[dvc] pull failed (continuing): {pull_result.message}")

    datamodule = CassavaDataModule(cfg)
    setup = getattr(datamodule, "setup", None)
    if callable(setup):
        setup("fit")

    train_cfg = getattr(cfg, "train", None)
    imbalance_cfg = getattr(train_cfg, "imbalance", None) if train_cfg else None
    strategy = str(getattr(imbalance_cfg, "strategy", "none")).lower() if imbalance_cfg else "none"
    use_loss_weights = strategy in {"loss_weights", "both"}
    class_weights = datamodule.train_class_weights() if use_loss_weights else None

    model = CassavaClassifier(cfg, class_weights=class_weights)

    outputs_dir = Path(str(cfg.paths.outputs_dir))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    loggers: list[Logger] = [CSVLogger(save_dir=str(outputs_dir), name="runs")]

    git_commit_id = get_git_commit_id()
    hparams = {
        "git_commit_id": git_commit_id,
        "train": dict(cfg.train),
        "model": dict(cfg.model),
        "augment": dict(cfg.augment),
        "data": {
            "dataset": dict(cfg.data.dataset),
            "split": dict(cfg.data.split),
            "paths": dict(cfg.data.paths),
            "synthetic": dict(cfg.data.synthetic),
        },
        "logger": dict(cfg.logger),
    }

    if bool(getattr(cfg.logger, "enabled", False)):
        try:
            from pytorch_lightning.loggers import MLFlowLogger

            mlflow_logger = MLFlowLogger(
                tracking_uri=str(cfg.logger.tracking_uri),
                experiment_name=str(cfg.logger.experiment_name),
                run_name=None
                if cfg.logger.run_name in (None, "null")
                else str(cfg.logger.run_name),
            )
            mlflow_logger.log_hyperparams(hparams)
            loggers.append(mlflow_logger)
        except Exception as exc:
            print(f"[mlflow] logger disabled (continuing): {exc}")

    callbacks: list[Callback] = []
    if bool(getattr(cfg.train, "save_checkpoints", False)):
        from pytorch_lightning.callbacks import ModelCheckpoint

        callbacks.append(
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename="best",
                save_weights_only=True,
            )
        )

    repo_root = Path(__file__).resolve().parents[2]
    max_time = _normalize_max_time(getattr(cfg.train, "max_time", None))

    accelerator = str(getattr(cfg.train, "accelerator", "auto"))
    if accelerator.lower() == "gpu" and not torch.cuda.is_available():
        raise SystemExit(
            "Requested train.accelerator=gpu, but CUDA is not available in this environment.\n"
            "On Windows, install CUDA PyTorch into `.venv` with `scripts/install_torch_cuda.ps1`.\n"
            "If you use `uv run`, add `--no-sync` to avoid reverting CUDA torch back to CPU-only."
        )

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.epochs),
        accelerator=accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        fast_dev_run=bool(cfg.train.fast_dev_run),
        default_root_dir=str(outputs_dir),
        logger=loggers,
        callbacks=callbacks,
        # Lightning enables checkpointing by default (ModelCheckpoint callback).
        # We explicitly disable it unless the project config asks for checkpoints,
        # to keep the training run lightweight and avoid writing large artifacts.
        enable_checkpointing=bool(getattr(cfg.train, "save_checkpoints", False)),
        max_time=max_time,
        deterministic=True,
    )

    trainer.fit(model=model, datamodule=datamodule)

    # Optional artifact logging/upload.
    artifacts_cfg = getattr(cfg.train, "artifacts", None)
    if not artifacts_cfg:
        return

    # Find best checkpoint path if checkpointing was enabled.
    best_ckpt_path: str | None = None
    for cb in getattr(trainer, "callbacks", []):
        if cb.__class__.__name__ == "ModelCheckpoint":
            best_ckpt_path = getattr(cb, "best_model_path", None)
            break

    if not best_ckpt_path:
        return

    ckpt_path = Path(str(best_ckpt_path))
    if not ckpt_path.exists():
        return

    # 1) Log checkpoint to MLflow artifacts (if MLflow logger is enabled).
    if bool(getattr(artifacts_cfg, "log_checkpoint_to_mlflow", True)):
        for lg in loggers:
            if lg.__class__.__name__ == "MLFlowLogger":
                try:
                    # MLFlowLogger exposes the active run_id; use low-level client to log artifact.
                    run_id = getattr(lg, "run_id", None)
                    experiment = getattr(lg, "experiment", None)
                    if run_id and experiment:
                        experiment.log_artifact(run_id, str(ckpt_path), artifact_path="checkpoints")
                        experiment.log_param(run_id, "best_checkpoint_path", str(ckpt_path))
                except Exception as exc:
                    print(f"[mlflow] artifact logging failed (continuing): {exc}")

    # 2) Optional: upload checkpoint to S3 (Yandex Object Storage).
    if bool(getattr(artifacts_cfg, "upload_checkpoint_to_s3", False)):
        try:
            import json
            import os
            from datetime import datetime, timezone

            import boto3
            import torch
        except Exception as exc:
            print(f"[s3] upload skipped (missing deps): {exc}")
            return

        _ensure_s3_env_from_dotenv(repo_root=repo_root)

        access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("YC_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("YC_SECRET_ACCESS_KEY")
        if not access_key or not secret_key:
            print("[s3] upload skipped (missing credentials in env)")
            return

        bucket = str(getattr(artifacts_cfg, "s3_bucket", "mlops-cassava-project"))
        prefix = str(getattr(artifacts_cfg, "s3_prefix", "cassava/models")).strip("/")
        endpoint_url = str(
            getattr(artifacts_cfg, "s3_endpoint_url", "https://storage.yandexcloud.net")
        )
        region = str(getattr(artifacts_cfg, "s3_region", "ru-central1"))

        s3 = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            region_name=region,
        )

        run_id = None
        for lg in loggers:
            if lg.__class__.__name__ == "MLFlowLogger":
                run_id = getattr(lg, "run_id", None)
                break
        run_tag = str(run_id) if run_id else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        key_prefix = f"{prefix}/{git_commit_id}/{run_tag}".strip("/")
        ckpt_key = f"{key_prefix}/{ckpt_path.name}"
        try:
            s3.upload_file(str(ckpt_path), bucket, ckpt_key)
            print(f"[s3] uploaded: s3://{bucket}/{ckpt_key}")
        except Exception as exc:
            print(f"[s3] upload failed (continuing): {exc}")
            return

        if not bool(getattr(artifacts_cfg, "upload_metrics_to_s3", False)):
            return

        def _as_float(v: object) -> float | None:
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                return float(v.detach().cpu().item())
            return None

        callback_metrics = getattr(trainer, "callback_metrics", {}) or {}
        metrics: dict[str, float] = {}
        for k, v in dict(callback_metrics).items():
            value = _as_float(v)
            if value is not None:
                metrics[str(k)] = float(value)

        payload = {
            "git_commit_id": git_commit_id,
            "run_tag": run_tag,
            "checkpoint_s3_uri": f"s3://{bucket}/{ckpt_key}",
            "metrics": metrics,
        }
        metrics_key = f"{key_prefix}/metrics.json"
        try:
            s3.put_object(
                Bucket=bucket,
                Key=metrics_key,
                Body=json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
                ContentType="application/json",
            )
            print(f"[s3] metrics uploaded: s3://{bucket}/{metrics_key}")
        except Exception as exc:
            print(f"[s3] metrics upload failed (continuing): {exc}")
