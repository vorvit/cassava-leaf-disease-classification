"""Training entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train(cfg: Any) -> None:
    """Run training using Hydra-composed config."""
    import pytorch_lightning as pl
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
    model = CassavaClassifier(cfg)

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

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.epochs),
        accelerator=str(cfg.train.accelerator),
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        fast_dev_run=bool(cfg.train.fast_dev_run),
        default_root_dir=str(outputs_dir),
        logger=loggers,
        callbacks=callbacks,
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
            import os

            import boto3
        except Exception as exc:
            print(f"[s3] upload skipped (missing deps): {exc}")
            return

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

        key = f"{prefix}/{ckpt_path.name}"
        try:
            s3.upload_file(str(ckpt_path), bucket, key)
            print(f"[s3] uploaded: s3://{bucket}/{key}")
        except Exception as exc:
            print(f"[s3] upload failed (continuing): {exc}")
