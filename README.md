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
