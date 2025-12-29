# CassavaVision — cassava leaf disease classification

Русская версия: [`README.md`](README.md)

## About the project

This project is a computer vision system for automated cassava (manioc) leaf disease diagnosis from photos. The project is built as a full-fledged MLOps pipeline with a reproducible training cycle and optimized inference.

Cassava is a critical food source for millions of people in Africa, yet crops are often affected by viral diseases. Manual diagnosis requires expert knowledge that is not always available to farmers.

Project goal: build an industrial-grade MLOps pipeline for training and deployment of an image classifier for 5 classes (4 diseases + healthy). The project will be built as a full-fledged industrial service with a reproducible training cycle and optimized inference.

This is a learning project to practice MLOps tooling on a real-world CV task, while keeping the training cycle reproducible and the inference path ready for later optimization.

## Input / output format

### Input

- JPEG/PNG image (binary payload)
- Expected tensor size after preprocessing: `(B, 3, 512, 512)` or `(B, 3, 380, 380)` depending on the model backend

### Output

JSON object:

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

**Classes:**

- Cassava Bacterial Blight (0)
- Cassava Brown Streak Disease (1)
- Cassava Green Mottle (2)
- Cassava Mosaic Disease (3)
- Healthy (4)

## Metrics

- **Accuracy**: Primary metric for the Kaggle competition. Business-interpretable metric for end users ("how often the bot is right"). Target reference: >0.88 (based on SOTA solutions from previous years).
- **F1-Macro**: Technical metric necessary due to strong class imbalance in the dataset (CMD class dominates). It allows tracking model quality on rare disease types, preventing the model from collapsing to predicting the majority class.

## Validation and testing

- **Strategy**: Stratified K-Fold Cross-Validation (5 folds). This is critical for imbalanced data to guarantee proportional representation of all classes in each fold.
- **Reproducibility**: Fixed global seeds (`torch.manual_seed`, `np.random.seed`) and use of `dvc` for versioning specific data splits (train/val/test).
- **Validation modes**:
  - **Stratified K-Fold (5 folds)**: `data.split.strategy=kfold` mode
  - **Hold-out**: default `data.split.strategy=holdout` mode

## Dataset

Dataset: **Cassava Leaf Disease Classification** (Kaggle competition, 2020).

- **Link**: [https://www.kaggle.com/c/cassava-leaf-disease-classification/data](https://www.kaggle.com/c/cassava-leaf-disease-classification/data)
- **Size**: 21,397 labeled images
- **Notes**:
  - Noisy labels present, requiring techniques like Label Smoothing or specialized losses (Bi-Tempered Logistic Loss).
  - Strong class imbalance (class 3 ≈ 61% of data).
  - Field images: varying lighting, angles, backgrounds.

## Modeling

### Baseline (implemented)

Fine-tuning lightweight models **ResNet18** or **EfficientNet-B0**, pretrained on ImageNet.

- Basic augmentations (Resize, Normalize) and advanced augmentations (Albumentations)
- Training with PyTorch Lightning for rapid prototyping
- Baseline target reference: Accuracy ≈ 0.80

### Main model (planned / roadmap)

Ensemble of several heavy architectures to achieve maximum accuracy:

1. **EfficientNet-B4** (or B3) — de facto standard for image classification tasks. Semi-supervised approach possible for noisy data.
2. **Vision Transformer (ViT Base 384)** — for capturing global contexts on the leaf.

- **Optimization**: Converting trained models to **TensorRT** (FP16 precision) for accelerated inference on GPU.

## Deployment (planned / roadmap)

Model will be deployed using **NVIDIA Triton Inference Server**.

Service architecture:

1. **Preprocessing Model**: Custom Python backend or DALI pipeline for image decoding and normalization on GPU.
2. **Inference Ensemble**: Triton Ensemble that queries models (TensorRT engines) in parallel and averages their predictions (Voting).
3. **Interface**:
   - Triton will be wrapped by a REST API service on **FastAPI**.
   - Client application: Telegram bot for sending leaf photos.

- The entire pipeline (training and deployment) will be automated via GitHub Actions.

---

## Technical details

### Setup

This project uses **uv** for dependency management and virtual environment.

#### 1. Clone the repository

```powershell
git clone https://github.com/vorvit/cassava-leaf-disease-classification.git
cd cassava-leaf-disease-classification
```

#### 2. Install dependencies

```powershell
python -m uv sync --dev
```

This command will create a virtual environment `.venv` and install all dependencies from `pyproject.toml` and `uv.lock`.

#### 3. Setup pre-commit hooks

```powershell
python -m uv run pre-commit install
python -m uv run pre-commit run -a
```

Pre-commit hooks automatically check code before commits (formatting, linting, type checking).

#### 4. Get data

Data is managed via DVC (Data Version Control) and is not stored in Git.

**Option A: Via DVC (recommended)**

```powershell
python -m uv run dvc pull
```

If DVC remote is configured for public access, data will be downloaded automatically.

**Option B: Direct archive download**

If DVC pull is not available, you can download the archive directly:

```powershell
python -m uv run cassava download-data
```

Options:

```powershell
python -m uv run cassava download-data download_data.force=true
```

**Note:** If the archive is in **RAR** format, Python stdlib can't extract it. In that case install **7-Zip** (`7z`) or use `rarfile` with an unrar backend.

#### 5. Setup MLflow (optional, for experiment logging)

For local MLflow server:

```powershell
docker compose up -d --build mlflow
```

MLflow will be available at `http://127.0.0.1:8080`.

#### 6. Setup GPU (optional, for Windows)

`uv.lock` pins a CPU-safe environment for portability. For local GPU usage (e.g. RTX 3060), install CUDA PyTorch into `.venv` without changing `uv.lock`:

```powershell
.\scripts\install_torch_cuda.ps1
```

After installing CUDA PyTorch, run training via venv Python directly (see Train section).

### Train

Model training is launched via the `train` CLI command. The project supports several training modes.

#### Quick smoke test (data-independent, always works)

To verify pipeline functionality without real data:

```powershell
python -m uv run python -m cassava_leaf_disease train data.synthetic.enabled=true train.epochs=1 train.batch_size=32 train.num_workers=0 logger.enabled=false
```

#### Training on real data (short run on a subset)

For quick verification on real data (with checkpoint saving for subsequent inference):

```powershell
python -m uv run python -m cassava_leaf_disease train `
  data.synthetic.enabled=false `
  data.limits.max_train_samples=800 `
  data.limits.max_val_samples=200 `
  train.epochs=1 `
  train.batch_size=32 `
  train.num_workers=0 `
  train.save_checkpoints=true `
  logger.enabled=true
```

**Important:** The `train.save_checkpoints=true` parameter is required to save the model checkpoint, which will then be automatically discovered when running inference.

#### Full training on real data

```powershell
python -m uv run python -m cassava_leaf_disease train `
  data.synthetic.enabled=false `
  train.epochs=50 `
  train.batch_size=64 `
  train.num_workers=0 `
  train.accelerator=cpu `
  train.save_checkpoints=true `
  logger.enabled=true
```

#### Training on GPU (Windows, via venv)

After installing CUDA PyTorch (see Setup):

```powershell
.\.venv\Scripts\python.exe -m cassava_leaf_disease train `
  data.synthetic.enabled=false `
  train.epochs=50 `
  train.batch_size=64 `
  train.num_workers=0 `
  train.accelerator=gpu `
  train.devices=1 `
  train.precision=16-mixed `
  train.save_checkpoints=true `
  logger.enabled=true
```

**Important:** `uv run` syncs the environment by default and may revert CUDA torch back to CPU-only torch from `uv.lock`. If you want to use `uv run`, add `--no-sync`:

```powershell
python -m uv run --no-sync cassava train train.accelerator=gpu train.devices=1 train.num_workers=0 train.precision=16-mixed
```

#### KFold mode (stratified cross-validation)

To run training with K-Fold validation:

```powershell
python -m uv run python -m cassava_leaf_disease train data.split.strategy=kfold data.split.folds=5 data.split.fold_index=0
```

#### Training parameters

Main parameters that can be overridden via Hydra:

- `train.epochs` — number of epochs
- `train.batch_size` — batch size
- `train.lr` — learning rate
- `train.accelerator` — `cpu` or `gpu`
- `train.precision` — `32`, `16-mixed` (for GPU)
- `train.num_workers` — number of DataLoader workers (on Windows with CUDA, `0` is recommended)
- `train.save_checkpoints` — whether to save model checkpoints (default `false`, set to `true` for subsequent inference)
- `model` — model choice (`resnet18`, `efficientnet_b0`)
- `augment` — augmentation choice (`basic`, `strong`)
- `logger.enabled` — enable/disable MLflow logging

Full list of parameters is available in the `configs/` directory.

### Inference

#### CLI inference

To run prediction on a single image via CLI:

**Simplest way (automatic discovery of the latest trained model):**

```powershell
.\.venv\Scripts\python.exe -m cassava_leaf_disease infer `
  infer.image_path="data/cassava/test_image/2216849948.jpg" `
  infer.device=auto
```

Inference will automatically find the latest trained model in `outputs/` (by modification time).

**Alternative options:**

**Option A: Use DVC-tracked checkpoint (recommended for production):**

```powershell
# First, download model from S3 and add to DVC tracking
.\.venv\Scripts\python.exe -m cassava_leaf_disease download-model

# Then run inference
.\.venv\Scripts\python.exe -m cassava_leaf_disease infer `
  infer.checkpoint_path="artifacts/best.ckpt" `
  infer.image_path="data/cassava/test_image/2216849948.jpg" `
  infer.device=auto
```

**Option B: Direct download from S3 (no local cache, slower):**

```powershell
.\.venv\Scripts\python.exe -m cassava_leaf_disease infer `
  infer.checkpoint_path=null `
  infer.checkpoint_s3_uri="s3://mlops-cassava-project/cassava/models/.../best.ckpt" `
  infer.image_path="data/cassava/test_image/2216849948.jpg" `
  infer.device=auto
```

**Checkpoint discovery priority:**

1. **Explicit path** (`infer.checkpoint_path`) — if specified and exists
2. **Auto-discovery** — latest `best.ckpt` in `outputs/` (by modification time)
3. **S3 URI** (`infer.checkpoint_s3_uri`) — if specified, downloaded to a temporary file

**Notes:**

- By default `infer.checkpoint_path=null` — uses automatic discovery of the latest model from `outputs/`
- `infer()` automatically calls `dvc pull` for `checkpoint_path` if the model is in DVC tracking
- If `checkpoint_path` is not found, auto-discovery is used, then `checkpoint_s3_uri` (if specified)
- **Recommended:** for production, use DVC-tracked checkpoint (option A) — faster and more reliable

#### FastAPI server

To run the REST API server:

```powershell
python -m uv run uvicorn --factory cassava_leaf_disease.serving.app:create_app --host 127.0.0.1 --port 8000
```

**Endpoints:**

- `GET /health` — server health check
- `POST /predict` — prediction (JPEG/PNG → JSON)

**Environment variables:**

- `CASSAVA_CHECKPOINT_PATH=artifacts/best.ckpt` — path to checkpoint (DVC-tracked)
- `CASSAVA_CHECKPOINT_S3_URI=s3://...` — alternative: S3 URI for direct download
- `CASSAVA_DEVICE=auto` (or `cpu` / `cuda`)

### Data (DVC)

We do not store data in git. Only `.dvc` metadata is committed.

**Expected layout:**

- `data/cassava/train.csv`
- `data/cassava/train_images/*.jpg`

**For getting data (public HTTP):**

Default remote is `public_http`:

- `https://storage.yandexcloud.net/mlops-cassava-project/cassava`

Pull:

```powershell
python -m uv run dvc pull
```

**For maintainers (push to S3 with credentials):**

Never commit secrets. Store them locally via `.dvc/config.local`:

```powershell
python -m uv run dvc remote modify --local yandex_s3 access_key_id "YOUR_ACCESS_KEY"
python -m uv run dvc remote modify --local yandex_s3 secret_access_key "YOUR_SECRET_KEY"
python -m uv run dvc remote modify --local yandex_s3 endpointurl "https://storage.yandexcloud.net"
python -m uv run dvc remote modify --local yandex_s3 region "ru-central1"

python -m uv run dvc push -r yandex_s3
```

### Tests & Coverage

**Run tests:**

```powershell
python -m uv run pytest
```

**Code coverage (project is configured with `fail_under = 95`):**

```powershell
python -m uv run pytest --cov=cassava_leaf_disease --cov-report=term-missing
```

### Changelog (SemVer + Conventional Commits)

The project uses Semantic Versioning and Conventional Commits for automatic changelog generation.

**Generate changelog via Docker:**

```powershell
.\scripts\generate_changelog.ps1
git add CHANGELOG.md
git commit -m "chore(release): update changelog"
```

### Results (latest GPU run on real data)

Latest run (25 minute limit, stopped by `train.max_time`):

- **MLflow run**: `ac0e608a1a774095aabb0f0c87b4bbb8` — `http://127.0.0.1:8080/#/experiments/1/runs/ac0e608a1a774095aabb0f0c87b4bbb8`
- **val/acc**: `0.686`
- **val/f1_macro**: `0.584`
- **val/loss**: `1.302`
- **train/acc**: `0.917`
- **train/loss_epoch**: `0.516`
- **Best model (best.ckpt) in S3**:
  - `s3://mlops-cassava-project/cassava/models/2882636f460823a9839bce138ce9adbffe968d00/ac0e608a1a774095aabb0f0c87b4bbb8/best.ckpt`
  - metrics (JSON): `s3://mlops-cassava-project/cassava/models/2882636f460823a9839bce138ce9adbffe968d00/ac0e608a1a774095aabb0f0c87b4bbb8/metrics.json`

### Roadmap

- 380/512 image size and stronger augmentations
- Label smoothing / methods for noisy labels
- Exporting the model and real inference (FastAPI)
- Triton/TensorRT/ensembles — separate, without impacting the main pipeline

---

## Note

This project was developed and tested using [Cursor](https://cursor.sh) on Windows 10.
