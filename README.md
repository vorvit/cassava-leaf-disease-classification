# cassava-leaf-disease-classification

## Project

Computer vision system for automated diagnosis of cassava (manioc) leaf diseases from photos, based on Kaggle competition:
`https://www.kaggle.com/c/cassava-leaf-disease-classification`.

**Goal**: build an MLOps pipeline for training and (optionally) deployment of an image classifier for 5 categories:

- Cassava Bacterial Blight (0)
- Cassava Brown Streak Disease (1)
- Cassava Green Mottle (2)
- Cassava Mosaic Disease (3)
- Healthy (4)

## Setup (Task2)

This project uses **uv** for dependency management.

```powershell
python -m uv sync --dev
python -m uv run pre-commit install
python -m uv run pre-commit run -a
```

## Data (DVC)

Dataset is tracked via **DVC** (data is not stored in git).

### For reviewers (no credentials, public bucket)

If the dataset cache is hosted in a **public-read** bucket, you can pull it without any credentials:

```powershell
python -m uv run dvc pull
```

This repo is configured to use `public_http` remote by default:

- `https://storage.yandexcloud.net/mlops-cassava-project/cassava`

### For maintainers (push to S3 with credentials)

To upload/update cache to Yandex Object Storage, use the `yandex_s3` remote and store credentials
locally in `.dvc/config.local` (never commit secrets):

```powershell
python -m uv run dvc remote modify --local yandex_s3 access_key_id "YOUR_ACCESS_KEY"
python -m uv run dvc remote modify --local yandex_s3 secret_access_key "YOUR_SECRET_KEY"
python -m uv run dvc remote modify --local yandex_s3 endpointurl "https://storage.yandexcloud.net"
python -m uv run dvc remote modify --local yandex_s3 region "ru-central1"

python -m uv run dvc push -r yandex_s3
```

Expected local layout:

- `data/cassava/train.csv`
- `data/cassava/train_images/*.jpg`

If you already have the dataset locally, track it:

```powershell
python -m uv run dvc add data/cassava
git add data/cassava.dvc data/.gitignore
git commit -m "chore(data): track cassava dataset with dvc"
```

## MLflow (local)

Task2 assumes MLflow server is available at `http://127.0.0.1:8080`.
For local usage you can start it with Docker:

```powershell
docker compose up -d --build mlflow
```

## Train

CPU (default, portable for Task2 checks):

```powershell
python -m uv run python -m cassava_leaf_disease train
```

Useful overrides:

```powershell
python -m uv run python -m cassava_leaf_disease train train.epochs=2 train.batch_size=32
```

## Local GPU mode (optional)

Task2 checks are CPU-safe (pinned in `uv.lock`). For your local RTX 3060 you can install CUDA PyTorch
into the project venv **without changing `uv.lock`**:

```powershell
.\scripts\install_torch_cuda.ps1
```

Then run training **using venv Python directly** (so uv does not re-sync CPU torch):

```powershell
.\.venv\Scripts\python.exe -m cassava_leaf_disease train train.epochs=2 train.batch_size=32
```

## Changelog (SemVer + Conventional Commits)

- Config: `cliff.toml`
- Generated file: `CHANGELOG.md`

Generate via Docker (recommended, no local binary install):

```powershell
.\scripts\generate_changelog_docker.ps1
git add CHANGELOG.md
git commit -m "chore(release): update changelog"
```

## Roadmap (not required for Task2)

- Add `F1-macro` metric (class imbalance)
- Add `StratifiedKFold` (5 folds) training mode
- Add serving: FastAPI (JPEG/PNG -> JSON probabilities)
- Heavy models / ensemble / Triton / TensorRT (only after Task2, optional)
