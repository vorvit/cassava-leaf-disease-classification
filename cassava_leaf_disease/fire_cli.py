"""Thin wrapper CLI using python-fire.

This module does NOT replace Hydra. It forwards a small, user-friendly CLI to the
existing Hydra-based CLI (`cassava_leaf_disease.commands`), by generating Hydra overrides.

Examples:
  cassava-fire train --epochs 1 --synthetic --no-mlflow
  cassava-fire train --epochs 2 --batch_size 32 train.precision=16-mixed
  cassava-fire infer --image data/cassava/train_images/xxx.jpg --ckpt outputs/.../best.ckpt
"""

from __future__ import annotations

from collections.abc import Sequence


def _bool_override(value: bool) -> str:
    return "true" if value else "false"


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("bool value is None")
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def _normalize_fire_args(args: list[str]) -> list[str]:
    """Normalize user-friendly CLI flags into python-fire-friendly ones.

    Fire uses Python identifiers for flags (underscores). Users often type dashes.
    We also support a couple of common `--no-xxx` negations.
    """
    mapping = {
        # Command aliases
        "download-data": "download_data",
        "--no-mlflow": "--mlflow=false",
        "--no_mlflow": "--mlflow=false",
        "--mlflow": "--mlflow=true",
        "--no-synthetic": "--synthetic=false",
        "--no_synthetic": "--synthetic=false",
        "--synthetic": "--synthetic=true",
        "--image-path": "--image",
        "--checkpoint-path": "--ckpt",
    }
    return [mapping.get(a, a) for a in args]


class CassavaFireCLI:
    """Fire CLI that forwards to Hydra commands.

    The goal is shorter and safer day-to-day commands, while keeping Hydra as the single
    source of truth for configuration.
    """

    def train(
        self,
        *overrides: str,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        precision: str | None = None,
        synthetic: bool | None = None,
        mlflow: bool | None = None,
        num_workers: str | int | None = None,
    ) -> None:
        """Run training (forwards to Hydra CLI).

        All keyword args are converted into Hydra overrides.
        You can also pass extra raw Hydra overrides as positional args.

        Examples:
          cassava-fire train --epochs 1 --synthetic --no-mlflow
          cassava-fire train --epochs 2 --batch_size 32 train.precision=16-mixed
        """
        from cassava_leaf_disease.commands import compose_cfg
        from cassava_leaf_disease.training.train import train as train_impl

        hydra_overrides: list[str] = []
        if epochs is not None:
            hydra_overrides.append(f"train.epochs={int(epochs)}")
        if batch_size is not None:
            hydra_overrides.append(f"train.batch_size={int(batch_size)}")
        if lr is not None:
            hydra_overrides.append(f"train.lr={float(lr)}")
        if precision is not None:
            hydra_overrides.append(f"train.precision={precision!s}")
        if num_workers is not None:
            hydra_overrides.append(f"train.num_workers={num_workers}")
        if synthetic is not None:
            hydra_overrides.append(
                f"data.synthetic.enabled={_bool_override(_parse_bool(synthetic))}"
            )
        if mlflow is not None:
            hydra_overrides.append(f"logger.enabled={_bool_override(_parse_bool(mlflow))}")

        cfg = compose_cfg("train", [*hydra_overrides, *overrides])
        train_impl(cfg)

    def infer(
        self,
        *overrides: str,
        image: str | None = None,
        ckpt: str | None = None,
        ckpt_s3: str | None = None,
        device: str | None = None,
        top_k: int | None = None,
    ) -> None:
        """Run inference (forwards to Hydra CLI).

        Examples:
          cassava-fire infer --image data/cassava/test_image/2216849948.jpg
          cassava-fire infer --image ... --ckpt artifacts/best.ckpt
          cassava-fire infer --image ... --ckpt_s3 s3://bucket/key/best.ckpt
        """
        from cassava_leaf_disease.commands import compose_cfg
        from cassava_leaf_disease.serving.infer import infer as infer_impl

        hydra_overrides: list[str] = []
        if image is not None:
            hydra_overrides.append(f"infer.image_path={image}")
        if ckpt is not None:
            hydra_overrides.append(f"infer.checkpoint_path={ckpt}")
        if ckpt_s3 is not None:
            # If S3 URI is provided, set checkpoint_path to null and use checkpoint_s3_uri
            hydra_overrides.append("infer.checkpoint_path=null")
            hydra_overrides.append(f"infer.checkpoint_s3_uri={ckpt_s3}")
        if device is not None:
            hydra_overrides.append(f"infer.device={device}")
        if top_k is not None:
            hydra_overrides.append(f"infer.top_k={int(top_k)}")

        cfg = compose_cfg("infer", [*hydra_overrides, *overrides])
        infer_impl(cfg)

    def raw(self, command: str, *overrides: str) -> None:
        """Forward any Hydra command verbatim.

        Example:
          cassava-fire raw train train.epochs=1 logger.enabled=false
        """
        from cassava_leaf_disease.commands import main as hydra_main

        hydra_main([str(command), *overrides])

    def download_data(self, *overrides: str, force: bool | None = None) -> None:
        """Download and extract dataset from a public link.

        Examples:
          cassava-fire download_data
          cassava-fire download_data --force
        """
        from cassava_leaf_disease.commands import compose_cfg
        from cassava_leaf_disease.data.download_data import download_data as impl

        hydra_overrides: list[str] = []
        if force is not None:
            hydra_overrides.append(f"download_data.force={_bool_override(_parse_bool(force))}")

        cfg = compose_cfg("download_data", [*hydra_overrides, *overrides])
        result = impl(cfg)
        if not result.success:
            raise SystemExit(f"download-data failed: {result.message}")
        print(f"[download-data] {result.message}")

    def download_model(
        self,
        *overrides: str,
        s3_uri: str | None = None,
        dst_dir: str | None = None,
        overwrite: bool | None = None,
        push: bool | None = None,
    ) -> None:
        """Download checkpoint from S3 and add to DVC tracking.

        Examples:
          cassava-fire download_model
          cassava-fire download_model --s3_uri s3://bucket/key/best.ckpt
          cassava-fire download_model --push
        """
        from cassava_leaf_disease.commands import compose_cfg
        from cassava_leaf_disease.data.download_model import download_model_to_dvc as impl

        hydra_overrides: list[str] = []
        if s3_uri is not None:
            hydra_overrides.append(f"download_model.s3_uri={s3_uri}")
        if dst_dir is not None:
            hydra_overrides.append(f"download_model.dst_dir={dst_dir}")
        if overwrite is not None:
            hydra_overrides.append(
                f"download_model.overwrite={_bool_override(_parse_bool(overwrite))}"
            )
        if push is not None:
            hydra_overrides.append(f"download_model.push={_bool_override(_parse_bool(push))}")

        cfg = compose_cfg("download_model", [*hydra_overrides, *overrides])
        result = impl(cfg)
        if not result.success:
            raise SystemExit(f"download-model failed: {result.message}")
        print(f"[download-model] {result.message}")
        if result.checkpoint_path:
            print(f"[download-model] Checkpoint: {result.checkpoint_path}")
            print(f"[download-model] DVC file: {result.checkpoint_path}.dvc")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `cassava-leaf-disease-fire` and `python -m ...fire_cli`."""
    import sys

    import fire

    from cassava_leaf_disease.commands import _ensure_utf8_stdio

    _ensure_utf8_stdio()
    args = sys.argv[1:] if argv is None else list(argv)
    args = _normalize_fire_args(args)
    # Fire expects to receive argv without the program name.
    fire.Fire(CassavaFireCLI, command=args, name="cassava-fire")


if __name__ == "__main__":  # pragma: no cover
    main()
