# CassavaVision — классификация болезней листьев маниоки

English version: [`README_EN.md`](README_EN.md)

## Постановка задачи

Разработка системы компьютерного зрения для автоматической диагностики заболеваний маниоки (кассавы)
по фотографиям листьев на основе соревнования Kaggle: `https://www.kaggle.com/c/cassava-leaf-disease-classification`.

Маниока является критически важным источником углеводов для миллионов людей в Африке, но урожаи часто
страдают от вирусных заболеваний. Ручная диагностика требует экспертных знаний, которые не всегда
доступны фермерам.

Цель проекта — создать надежный MLOps‑пайплайн для обучения и (опционально) деплоя модели классификации
изображений, различающей 5 категорий (4 типа болезней + здоровые листья).

Проект будет построен как полноценный сервис с воспроизводимым циклом обучения и оптимизированным инференсом.
На данном учебном примере я планирую изучить и потренировать навыки работы с MLOps‑инструментами на задаче CV.

## Формат входных и выходных данных (целевой API)

### Вход

- Изображение в формате JPEG/PNG (binary payload)
- Ожидаемая размерность тензора после предобработки: `(B, 3, 512, 512)` или `(B, 3, 380, 380)` —
  зависит от бэкенда модели (roadmap)

### Выход

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

Классы:

- Cassava Bacterial Blight (0)
- Cassava Brown Streak Disease (1)
- Cassava Green Mottle (2)
- Cassava Mosaic Disease (3)
- Healthy (4)

## Метрики

- **Accuracy**: основная метрика соревнования Kaggle. Ожидаемый уровень: > 0.88 (ориентир по SOTA решениям).
- **F1‑Macro**: метрика для имбалансных классов (CMD доминирует), чтобы качество не деградировало до мажоритарного класса.

## Валидация и тест

- **Stratified K‑Fold (5 folds)**: режим `data.split.strategy=kfold`
- **Hold‑out**: режим по умолчанию `data.split.strategy=holdout`
- **Воспроизводимость**: фиксация seed (Lightning), конфиги Hydra, данные под DVC

## Датасет

Используется датасет **Cassava Leaf Disease Classification** (Kaggle, 2020).

- **Объем**: 21,397 аннотированных изображений
- **Особенности**:
  - noisy labels (в будущем можно добавить label smoothing / специальные лоссы)
  - сильный дисбаланс (класс 3 ≈ 61%)
  - “полевые” фото: разное освещение, ракурс, фон

## Быстро скачать датасет (легальная публичная ссылка)

Если у вас нет доступа к данным через DVC, можно скачать архив с публичной ссылки и распаковать в
`data/cassava/`:

```bash
python -m uv run cassava download-data
```

Опции:

```bash
python -m uv run cassava download-data download_data.force=true
```

Примечание: если архив окажется в формате **RAR**, стандартная библиотека Python его не распакует.
В этом случае нужен установленный **7-Zip** (`7z`) или `rarfile` + unrar-backend.

## Моделирование

### Бейзлайн (реализовано)

- fine‑tuning **ResNet18** (через `timm`, предобучение ImageNet)
- простые аугментации (resize/normalize)
- обучение через **PyTorch Lightning**
- целевой ориентир бейзлайна: Accuracy ≈ 0.80

### Основная модель (план / roadmap)

- **EfficientNet‑B4** (или B3)
- **Vision Transformer (ViT Base 384)**
- ансамбль нескольких моделей
- оптимизация инференса через **TensorRT (FP16)**

## Внедрение (план / roadmap)

- инференс на **NVIDIA Triton Inference Server**
- preprocessing модель (Python backend или DALI pipeline)
- Triton Ensemble (усреднение предсказаний)
- внешний REST API на **FastAPI** + клиент (например Telegram‑бот)
- автоматизация обучения/деплоя через GitHub Actions

## Данные (DVC)

Данные не храним в git. В git попадают только `.dvc` метафайлы.

Ожидаемая структура:

- `data/cassava/train.csv`
- `data/cassava/train_images/*.jpg`

### Для проверяющих (pull без кредов, public HTTP)

Репозиторий настроен так, что дефолтный remote — `public_http`:

- `https://storage.yandexcloud.net/mlops-cassava-project/cassava`

Команда:

```powershell
python -m uv run dvc pull
```

Важно: если бакет/префикс не public-read для `cassava/files/md5/**`, `dvc pull` вернёт 403, и обучение
перейдёт на синтетический fallback (чтобы проверка Task2 не падала).

### Для мейнтейнеров (push в S3 с кредами)

Креды НЕ коммитим. Храним локально через `.dvc/config.local`:

```powershell
python -m uv run dvc remote modify --local yandex_s3 access_key_id "YOUR_ACCESS_KEY"
python -m uv run dvc remote modify --local yandex_s3 secret_access_key "YOUR_SECRET_KEY"
python -m uv run dvc remote modify --local yandex_s3 endpointurl "https://storage.yandexcloud.net"
python -m uv run dvc remote modify --local yandex_s3 region "ru-central1"

python -m uv run dvc push -r yandex_s3
```

## Setup (Task2)

Проект использует **uv**.

```powershell
python -m uv sync --dev
python -m uv run pre-commit install
python -m uv run pre-commit run -a
```

## MLflow (локально)

В задании предполагается, что MLflow доступен на `http://127.0.0.1:8080`.
Для локальных тестов поднимаем через Docker:

```powershell
docker compose up -d --build mlflow
```

## Train

### Быстрый smoke‑train (не зависит от данных, проходит всегда)

```powershell
python -m uv run python -m cassava_leaf_disease train data.synthetic.enabled=true train.epochs=1 train.batch_size=32 train.num_workers=0 logger.enabled=false
```

### Обучение на реальных данных (короткий прогон на подмножестве)

```powershell
python -m uv run python -m cassava_leaf_disease train data.synthetic.enabled=false data.limits.max_train_samples=800 data.limits.max_val_samples=200 train.epochs=1 train.batch_size=32 train.num_workers=0 logger.enabled=true
```

### KFold режим

```powershell
python -m uv run python -m cassava_leaf_disease train data.split.strategy=kfold data.split.folds=5 data.split.fold_index=0
```

## Local GPU mode (Windows, опционально)

`uv.lock` фиксирует CPU‑зависимости для переносимой проверки Task2. Для локального GPU (RTX 3060)
есть скрипт, который ставит CUDA PyTorch в `.venv` без изменения `uv.lock`:

```powershell
.\scripts\install_torch_cuda.ps1
```

Запускать на GPU нужно через venv Python (и на Windows использовать `train.num_workers=0`).
Важно: `uv run` по умолчанию делает sync окружения и может вернуть CPU‑torch из `uv.lock`.
Если всё-таки хотите запускать через `uv run`, используйте `--no-sync`.

```powershell
.\.venv\Scripts\python.exe -m cassava_leaf_disease train data.synthetic.enabled=false data.limits.max_train_samples=5000 data.limits.max_val_samples=1000 train.epochs=2 train.batch_size=64 train.num_workers=0 train.precision=16-mixed logger.enabled=true
```

Эквивалент через `uv run` (без пересинхронизации окружения):

```powershell
python -m uv run --no-sync cassava train train.accelerator=gpu train.devices=1 train.num_workers=0 train.precision=16-mixed
```

## Inference (FastAPI, базовая версия)

Запуск сервера:

```powershell
python -m uv run uvicorn cassava_leaf_disease.serving.app:app --host 127.0.0.1 --port 8000
```

Эндпоинты:

- `GET /health`
- `POST /predict` (JPEG/PNG → JSON как выше)

Важно: сейчас `/predict` — минимальный “stub” (детерминированный), чтобы показать контракт API без тяжёлого
экспорта модели. Экспорт “реальной” модели — следующий этап.

## Changelog (SemVer + Conventional Commits)

- `cliff.toml`
- `CHANGELOG.md`

Генерация через Docker (без установки бинаря):

```powershell
.\scripts\generate_changelog_docker.ps1
git add CHANGELOG.md
git commit -m "chore(release): update changelog"
```

## Roadmap (после Task2)

- Image size 380/512 и более сильные аугментации
- Label smoothing / методы для noisy labels
- Экспорт модели и “реальный” инференс (FastAPI)
- Triton/TensorRT/ансамбли — отдельно, без влияния на Task2
