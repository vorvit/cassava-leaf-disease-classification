# CassavaVision — cassava leaf disease classification

Русская версия: [`README.md`](README.md)

## Problem statement

We build a computer‑vision system for automated cassava (manioc) leaf disease diagnosis from photos,
based on the Kaggle competition: `https://www.kaggle.com/c/cassava-leaf-disease-classification`.

Cassava is a critical food source for millions of people in Africa, yet crops are often affected by viral
diseases. Manual diagnosis requires expert knowledge that is not always available to farmers.

Project goal: build an industrial‑grade MLOps pipeline for training and (optionally) deployment of an
image classifier for 5 classes (4 diseases + healthy).

This is a learning project to practice MLOps tooling on a real-world CV task, while keeping the training
cycle reproducible and the inference path ready for later optimization.

## Input / output format (target API)

### Input

- JPEG/PNG image (binary payload)
- Expected tensor size after preprocessing: `(B, 3, 512, 512)` or `(B, 3, 380, 380)` depending on the
  model backend (roadmap)

### Output

JSON:

```json
{
  "predicted_class_id": 0,
  "class_name": "Cassava Bacterial Blight",
  "confidence": 0.91,
  "probabilities": {
    "Cassava Bacterial Blight": 0.91,
    "Cassava Brown Streak Disease": 0.02,
    "Cassava Green Mottle": 0.01,
    "Cassava Mosaic Disease": 0.03,
    "Healthy": 0.03
  }
}
```

Classes:

- Cassava Bacterial Blight (0)
- Cassava Brown Streak Disease (1)
- Cassava Green Mottle (2)
- Cassava Mosaic Disease (3)
- Healthy (4)

## Metrics

- **Accuracy**: Kaggle primary metric. Target reference: > 0.88 (SOTA benchmark).
- **F1‑Macro**: for class imbalance (CMD dominates).

## Validation

- **Stratified K‑Fold (5 folds)**: `data.split.strategy=kfold`
- **Hold‑out**: default `data.split.strategy=holdout`
- **Reproducibility**: fixed seeds, Hydra configs, DVC for data

## Dataset

Dataset: **Cassava Leaf Disease Classification** (Kaggle, 2020).

- **Size**: 21,397 labeled images
- **Notes**:
  - noisy labels (future: label smoothing / specialized losses)
  - strong class imbalance (class 3 ≈ 61%)
  - field images: varying lighting, angles, backgrounds

## Quick dataset download (public link)

If DVC pull is not available, you can download a public archive and extract it into `data/cassava/`:

```bash
python -m uv run cassava download-data
```

Options:

```bash
python -m uv run cassava download-data download_data.force=true
```

Note: if the archive is **RAR**, Python stdlib can't extract it. In that case install **7-Zip** (`7z`)
or use `rarfile` with an unrar backend.

## Modeling

### Baseline (implemented)

- fine‑tuning **ResNet18** via `timm` (ImageNet pretrained)
- basic augmentations (resize/normalize)
- training with **PyTorch Lightning**
- baseline target reference: Accuracy ≈ 0.80

### Main model (planned / roadmap)

- **EfficientNet‑B4** (or B3)
- **Vision Transformer (ViT Base 384)**
- model ensemble
- inference optimization via **TensorRT (FP16)**

## Deployment (planned / roadmap)

- **NVIDIA Triton Inference Server**
- preprocessing model (Python backend or DALI)
- Triton Ensemble (averaging/voting)
- external REST API via **FastAPI** + client (e.g., Telegram bot)
- automation via GitHub Actions

## Data (DVC)

We do not store data in git. Only `.dvc` metadata is committed.

Expected layout:

- `data/cassava/train.csv`
- `data/cassava/train_images/*.jpg`

### For reviewers (no credentials, public HTTP pull)

Default remote is `public_http`:

- `https://storage.yandexcloud.net/mlops-cassava-project/cassava`

Pull:

```powershell
python -m uv run dvc pull
```

Note: if the bucket/prefix is not public-read for `cassava/files/md5/**`, `dvc pull` will fail with 403,
and training will use a synthetic fallback so Task2 checks can still run end‑to‑end.

### For maintainers (push to S3 with credentials)

Never commit secrets. Store them locally via `.dvc/config.local`:

```powershell
python -m uv run dvc remote modify --local yandex_s3 access_key_id "YOUR_ACCESS_KEY"
python -m uv run dvc remote modify --local yandex_s3 secret_access_key "YOUR_SECRET_KEY"
python -m uv run dvc remote modify --local yandex_s3 endpointurl "https://storage.yandexcloud.net"
python -m uv run dvc remote modify --local yandex_s3 region "ru-central1"

python -m uv run dvc push -r yandex_s3
```

## Setup (Task2)

This project uses **uv**.

```powershell
python -m uv sync --dev
python -m uv run pre-commit install
python -m uv run pre-commit run -a
```

## MLflow (local)

Task assumes MLflow is available at `http://127.0.0.1:8080`.
For local debugging start it via Docker:

```powershell
docker compose up -d --build mlflow
```

## Train

### Quick smoke‑train (data‑independent, always works)

```powershell
python -m uv run python -m cassava_leaf_disease train data.synthetic.enabled=true train.epochs=1 train.batch_size=32 train.num_workers=0 logger.enabled=false
```

### Real data training (short run on a subset)

```powershell
python -m uv run python -m cassava_leaf_disease train data.synthetic.enabled=false data.limits.max_train_samples=800 data.limits.max_val_samples=200 train.epochs=1 train.batch_size=32 train.num_workers=0 logger.enabled=true
```

### KFold mode

```powershell
python -m uv run python -m cassava_leaf_disease train data.split.strategy=kfold data.split.folds=5 data.split.fold_index=0
```

## Local GPU mode (Windows, optional)

`uv.lock` pins a CPU‑safe environment for portable Task2 checks. For local GPU usage (e.g. RTX 3060),
install CUDA PyTorch into `.venv` without changing `uv.lock`:

```powershell
.\scripts\install_torch_cuda.ps1
```

Run training via venv Python (and use `train.num_workers=0` on Windows):

```powershell
.\.venv\Scripts\python.exe -m cassava_leaf_disease train data.synthetic.enabled=false data.limits.max_train_samples=5000 data.limits.max_val_samples=1000 train.epochs=2 train.batch_size=64 train.num_workers=0 train.precision=16-mixed logger.enabled=true
```

Important: `uv run` syncs the environment by default and may revert CUDA torch back to CPU-only torch from `uv.lock`.
If you want to use `uv run`, add `--no-sync`:

```powershell
python -m uv run --no-sync cassava train train.accelerator=gpu train.devices=1 train.num_workers=0 train.precision=16-mixed
```

## Inference (FastAPI, minimal)

Run:

```powershell
python -m uv run uvicorn cassava_leaf_disease.serving.app:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `POST /predict` (JPEG/PNG → JSON format above)

Note: `/predict` is currently a minimal deterministic stub to document the API contract; exporting and
serving a real trained checkpoint is a follow‑up step.

## Changelog (SemVer + Conventional Commits)

- `cliff.toml`
- `CHANGELOG.md`

Generate via Docker (no local binary install):

```powershell
.\scripts\generate_changelog_docker.ps1
git add CHANGELOG.md
git commit -m "chore(release): update changelog"
```

## Roadmap (after Task2)

- 380/512 image size and stronger augmentations
- Label smoothing / methods for noisy labels
- Exporting the model and real inference (FastAPI)
- Triton/TensorRT/ensembles — separate, without impacting Task2
