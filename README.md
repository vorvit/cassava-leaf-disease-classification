# CassavaVision — классификация болезней листьев маниоки

English version: [`README_EN.md`](README_EN.md)

## О проекте

Этот проект представляет собой систему компьютерного зрения для автоматической диагностики заболеваний маниоки (кассавы) по фотографиям листьев. Проект построен как полноценный MLOps-пайплайн с воспроизводимым циклом обучения и оптимизированным инференсом.

Маниока является критически важным источником углеводов для миллионов людей в Африке, но урожаи часто страдают от вирусных заболеваний. Ручная диагностика требует экспертных знаний, которые не всегда доступны фермерам.

Цель проекта — создать надежный MLOps-пайплайн для обучения и деплоя модели классификации изображений, способной различать 5 категорий (4 типа болезней + здоровые листья). Проект будет построен как полноценный индустриальный сервис с воспроизводимым циклом обучения и оптимизированным инференсом.

На данном учебном примере планируется изучить и отработать навыки использования MLOps-инструментов на задаче компьютерного зрения.

## Формат входных и выходных данных

### Входные данные

- Изображение в формате JPEG/PNG (binary payload)
- Ожидаемая размерность тензора после предобработки: `(B, 3, 512, 512)` или `(B, 3, 380, 380)` в зависимости от бэкенда модели

### Выходные данные

JSON объект:

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

**Классы:**

- Cassava Bacterial Blight (0)
- Cassava Brown Streak Disease (1)
- Cassava Green Mottle (2)
- Cassava Mosaic Disease (3)
- Healthy (4)

## Метрики

- **Accuracy**: Основная метрика соревнования Kaggle. Бизнес-интерпретируемая метрика для конечного пользователя ("как часто бот прав"). Ожидаемый уровень: >0.88 (на основе SOTA решений прошлых лет).
- **F1-Macro**: Техническая метрика, необходимая из-за сильного дисбаланса классов в датасете (класс CMD доминирует). Она позволит отследить качество работы модели на редких видах болезней, чтобы модель не скатывалась в предсказание мажоритарного класса.

## Валидация и тест

- **Стратегия**: Stratified K-Fold Cross-Validation (5 folds). Это критично для имбалансных данных, чтобы гарантировать пропорциональное представление всех классов в каждом фолде.
- **Воспроизводимость**: Фиксация глобальных сидов (`torch.manual_seed`, `np.random.seed`) и использование `dvc` для версионирования конкретных сплитов данных (train/val/test).
- **Режимы валидации**:
  - **Stratified K-Fold (5 folds)**: режим `data.split.strategy=kfold`
  - **Hold-out**: режим по умолчанию `data.split.strategy=holdout`

## Датасет

Используется датасет **Cassava Leaf Disease Classification** (с платформы Kaggle, соревнование 2020 года).

- **Ссылка**: [https://www.kaggle.com/c/cassava-leaf-disease-classification/data](https://www.kaggle.com/c/cassava-leaf-disease-classification/data)
- **Объем**: 21,397 аннотированных изображений
- **Особенности**:
  - Присутствует шум в разметке (noisy labels), что потребует использования техник вроде Label Smoothing или специфических лоссов (Bi-Tempered Logistic Loss).
  - Сильный дисбаланс (класс 3 составляет ~61% данных).
  - Изображения "в полевых условиях": разное освещение, ракурсы, фон.

## Моделирование

### Бейзлайн (реализовано)

Fine-tuning легковесной модели **ResNet18** или **EfficientNet-B0**, предобученной на ImageNet.

- Простая аугментация (Resize, Normalize) и продвинутые аугментации (Albumentations)
- Обучение с использованием PyTorch Lightning для быстрого прототипирования
- Целевая метрика бейзлайна: Accuracy ~0.80

### Основная модель (план / roadmap)

Ансамбль (Ensemble) из нескольких тяжелых архитектур для достижения максимальной точности:

1. **EfficientNet-B4** (или B3) — стандарт де-факто для задач классификации изображений. Возможен semi-supervised подход для шумных данных.
2. **Vision Transformer (ViT Base 384)** — для захвата глобальных контекстов на листе.

- **Оптимизация**: Конвертация обученных моделей в **TensorRT** (FP16 precision) для ускорения инференса на GPU.

## Внедрение (план / roadmap)

Модель будет развернута с использованием **NVIDIA Triton Inference Server**.

Архитектура сервиса:

1. **Preprocessing Model**: Кастомный Python-бэкенд или DALI pipeline для декодирования изображений и нормализации на GPU.
2. **Inference Ensemble**: Triton Ensemble, который параллельно опрашивает модели (TensorRT engines) и усредняет их предсказания (Voting).
3. **Интерфейс**:
   - Снаружи Triton будет закрыт REST API сервисом на **FastAPI**.
   - В качестве клиентского приложения — Telegram-бот для отправки фото листьев.

- Весь пайплайн (обучение и деплой) будет автоматизирован через GitHub Actions.

---

## Технические детали

### Setup

Проект использует **uv** для управления зависимостями и виртуальным окружением.

#### 1. Клонирование репозитория

```powershell
git clone https://github.com/vorvit/cassava-leaf-disease-classification.git
cd cassava-leaf-disease-classification
```

#### 2. Установка зависимостей

```powershell
python -m uv sync --dev
```

Эта команда создаст виртуальное окружение `.venv` и установит все зависимости из `pyproject.toml` и `uv.lock`.

#### 3. Настройка pre-commit hooks

```powershell
python -m uv run pre-commit install
python -m uv run pre-commit run -a
```

Pre-commit hooks автоматически проверяют код перед коммитом (форматирование, линтинг, проверка типов).

#### 4. Получение данных

Данные управляются через DVC (Data Version Control) и не хранятся в Git.

**Вариант A: Через DVC (рекомендуется)**

```powershell
python -m uv run dvc pull
```

Если DVC remote настроен на публичный доступ, данные будут скачаны автоматически.

**Вариант B: Прямая загрузка архива**

Если DVC pull недоступен, можно скачать архив напрямую:

```powershell
python -m uv run cassava download-data
```

Опции:

```powershell
python -m uv run cassava download-data download_data.force=true
```

**Примечание:** Если архив окажется в формате **RAR**, стандартная библиотека Python его не распакует. В этом случае нужен установленный **7-Zip** (`7z`) или `rarfile` + unrar-backend.

#### 5. Настройка MLflow (опционально, для логирования экспериментов)

Для локального запуска MLflow сервера:

```powershell
docker compose up -d --build mlflow
```

MLflow будет доступен на `http://127.0.0.1:8080`.

#### 6. Настройка GPU (опционально, для Windows)

`uv.lock` фиксирует CPU-зависимости для переносимой проверки. Для локального GPU (например, RTX 3060) есть скрипт, который устанавливает CUDA PyTorch в `.venv` без изменения `uv.lock`:

```powershell
.\scripts\install_torch_cuda.ps1
```

После установки CUDA PyTorch запускайте обучение через venv Python напрямую (см. раздел Train).

### Train

Обучение модели запускается через CLI. Проект поддерживает два способа запуска: через **Fire CLI** (удобный синтаксис) и через **Hydra CLI** (полный контроль).

#### Fire CLI (рекомендуется для ежедневной работы)

Fire CLI предоставляет удобный синтаксис с именованными параметрами. Все команды запускаются из корня проекта (относительные пути).

**Быстрый smoke-тест (синтетические данные):**

```powershell
python -m uv run cassava-fire train --epochs 1 --synthetic --no-mlflow --num_workers 0
```

**Обучение на реальных данных (короткий прогон с сохранением чекпоинта):**

```powershell
python -m uv run cassava-fire train `
  --epochs 1 `
  --batch_size 32 `
  --num_workers 0 `
  --mlflow `
  train.save_checkpoints=true `
  data.synthetic.enabled=false `
  data.limits.max_train_samples=800 `
  data.limits.max_val_samples=200
```

**Полное обучение на реальных данных:**

```powershell
python -m uv run cassava-fire train `
  --epochs 50 `
  --batch_size 64 `
  --num_workers 0 `
  --mlflow `
  train.save_checkpoints=true `
  train.accelerator=cpu `
  data.synthetic.enabled=false
```

**Обучение на GPU (через venv, после установки CUDA PyTorch):**

```powershell
.\.venv\Scripts\python.exe -m cassava_leaf_disease.fire_cli train `
  --epochs 50 `
  --batch_size 64 `
  --num_workers 0 `
  --precision 16-mixed `
  --mlflow `
  train.save_checkpoints=true `
  train.accelerator=gpu `
  train.devices=1 `
  data.synthetic.enabled=false
```

**KFold режим (стратифицированная кросс-валидация):**

```powershell
python -m uv run cassava-fire train `
  --epochs 10 `
  --mlflow `
  train.save_checkpoints=true `
  data.synthetic.enabled=false `
  data.split.strategy=kfold `
  data.split.folds=5 `
  data.split.fold_index=0
```

**Параметры Fire CLI для команды `train`:**

| Параметр        | Тип          | Описание                                                             | Пример                             |
| --------------- | ------------ | -------------------------------------------------------------------- | ---------------------------------- |
| `--epochs`      | `int`        | Количество эпох обучения                                             | `--epochs 50`                      |
| `--batch_size`  | `int`        | Размер батча                                                         | `--batch_size 64`                  |
| `--lr`          | `float`      | Learning rate                                                        | `--lr 0.0003`                      |
| `--precision`   | `str`        | Точность вычислений: `32`, `16-mixed`                                | `--precision 16-mixed`             |
| `--num_workers` | `int \| str` | Количество воркеров DataLoader (на Windows с CUDA рекомендуется `0`) | `--num_workers 0`                  |
| `--synthetic`   | `bool`       | Использовать синтетические данные (по умолчанию `false`)             | `--synthetic` или `--no-synthetic` |
| `--mlflow`      | `bool`       | Включить/выключить MLflow логирование                                | `--mlflow` или `--no-mlflow`       |
| `*overrides`    | `str`        | Дополнительные Hydra overrides (позиционные аргументы)               | `train.save_checkpoints=true`      |

**Примеры комбинаций параметров:**

```powershell
# Минимальная команда с синтетическими данными
python -m uv run cassava-fire train --epochs 1 --synthetic

# С дополнительными Hydra overrides
python -m uv run cassava-fire train `
  --epochs 10 `
  --batch_size 32 `
  train.save_checkpoints=true `
  model=efficientnet_b0 `
  augment=strong

# С GPU (через venv)
.\.venv\Scripts\python.exe -m cassava_leaf_disease.fire_cli train `
  --epochs 50 `
  --precision 16-mixed `
  train.accelerator=gpu `
  train.devices=1 `
  train.save_checkpoints=true
```

#### Hydra CLI (полный контроль над конфигурацией)

Для полного контроля над всеми параметрами используйте Hydra CLI напрямую. Все пути относительные к корню проекта.

**Быстрый smoke-тест:**

```powershell
python -m uv run python -m cassava_leaf_disease train `
  data.synthetic.enabled=true `
  train.epochs=1 `
  train.batch_size=32 `
  train.num_workers=0 `
  logger.enabled=false
```

**Обучение на реальных данных (короткий прогон с сохранением чекпоинта):**

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

**Полное обучение на реальных данных:**

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

**Обучение на GPU (через venv):**

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

**Важно:**

- Параметр `train.save_checkpoints=true` необходим для сохранения чекпоинта модели, который затем будет автоматически найден при запуске инференса.
- `uv run` по умолчанию делает sync окружения и может вернуть CPU-torch из `uv.lock`. Для GPU используйте venv напрямую или `uv run --no-sync`.

**Основные параметры Hydra для обучения:**

- `train.epochs` — количество эпох
- `train.batch_size` — размер батча
- `train.lr` — learning rate
- `train.accelerator` — `cpu` или `gpu`
- `train.devices` — количество устройств (для GPU: `1`)
- `train.precision` — `32`, `16-mixed` (для GPU)
- `train.num_workers` — количество воркеров для DataLoader (на Windows с CUDA рекомендуется `0`)
- `train.save_checkpoints` — сохранять ли чекпоинты модели (по умолчанию `false`, установите `true` для последующего инференса)
- `data.synthetic.enabled` — использовать синтетические данные (`true`/`false`)
- `data.limits.max_train_samples` — ограничение количества обучающих примеров
- `data.limits.max_val_samples` — ограничение количества валидационных примеров
- `data.split.strategy` — стратегия разбиения: `holdout`, `kfold`
- `data.split.folds` — количество фолдов для KFold
- `data.split.fold_index` — индекс текущего фолда (0..folds-1)
- `model` — выбор модели (`resnet18`, `efficientnet_b0`)
- `augment` — выбор аугментаций (`basic`, `strong`)
- `logger.enabled` — включить/выключить MLflow логирование

Полный список параметров доступен в `configs/` директории.

### Inference

Для запуска предсказания на одном изображении доступны два способа: через **Fire CLI** (удобный синтаксис) и через **Hydra CLI** (полный контроль). Все пути относительные к корню проекта.

#### Fire CLI (рекомендуется для ежедневной работы)

**Самый простой способ (автоматический поиск последней обученной модели):**

```powershell
python -m uv run cassava-fire infer --image data/cassava/test_image/2216849948.jpg
```

Инференс автоматически найдёт последнюю обученную модель в `outputs/` (по времени модификации).

**С явным указанием checkpoint (относительный путь):**

Для DVC-tracked модели (скачанной через `download-model`):

```powershell
python -m uv run cassava-fire infer `
  --image data/cassava/test_image/2216849948.jpg `
  --ckpt artifacts/best.ckpt
```

**Примечание:** Если не указать `--ckpt`, команда автоматически найдёт последний checkpoint в `outputs/runs/version_X/checkpoints/best.ckpt` (по времени модификации).

**С загрузкой из S3:**

```powershell
python -m uv run cassava-fire infer `
  --image data/cassava/test_image/2216849948.jpg `
  --ckpt_s3 s3://mlops-cassava-project/cassava/models/.../best.ckpt
```

**С дополнительными параметрами:**

```powershell
python -m uv run cassava-fire infer `
  --image data/cassava/test_image/2216849948.jpg `
  --device cuda `
  --top_k 3
```

**Параметры Fire CLI для команды `infer`:**

| Параметр     | Тип   | Описание                                                                                                         | Пример                                           |
| ------------ | ----- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `--image`    | `str` | Путь к изображению (относительно корня проекта)                                                                  | `--image data/cassava/test_image/2216849948.jpg` |
| `--ckpt`     | `str` | Путь к checkpoint (относительный или абсолютный). Если не указан, используется автоматический поиск в `outputs/` | `--ckpt artifacts/best.ckpt`                     |
| `--ckpt_s3`  | `str` | S3 URI для прямой загрузки checkpoint (скачивается во временный файл)                                            | `--ckpt_s3 s3://bucket/key/best.ckpt`            |
| `--device`   | `str` | Устройство для инференса: `auto`, `cpu`, `cuda`                                                                  | `--device auto`                                  |
| `--top_k`    | `int` | Количество топ-классов для вывода (по умолчанию 5)                                                               | `--top_k 3`                                      |
| `*overrides` | `str` | Дополнительные Hydra overrides (позиционные аргументы)                                                           | `infer.checkpoint_path=null`                     |

**Приоритет поиска checkpoint:**

1. **Явный путь** (`--ckpt`) — если указан и существует
2. **Автоматический поиск** — последний `best.ckpt` в `outputs/` (по времени модификации)
3. **S3 URI** (`--ckpt_s3`) — если указан, скачивается во временный файл

#### Hydra CLI (полный контроль над конфигурацией)

**Автоматический поиск последней модели:**

```powershell
python -m uv run python -m cassava_leaf_disease infer `
  infer.image_path=data/cassava/test_image/2216849948.jpg `
  infer.device=auto
```

**С явным указанием checkpoint (относительный путь):**

```powershell
python -m uv run python -m cassava_leaf_disease infer `
  infer.checkpoint_path=artifacts/best.ckpt `
  infer.image_path=data/cassava/test_image/2216849948.jpg `
  infer.device=auto
```

**С загрузкой из S3:**

```powershell
python -m uv run python -m cassava_leaf_disease infer `
  infer.checkpoint_path=null `
  infer.checkpoint_s3_uri=s3://mlops-cassava-project/cassava/models/.../best.ckpt `
  infer.image_path=data/cassava/test_image/2216849948.jpg `
  infer.device=auto
```

**Параметры Hydra для инференса:**

- `infer.image_path` — путь к изображению (относительно корня проекта)
- `infer.checkpoint_path` — путь к checkpoint (относительный или абсолютный). Если `null`, используется автоматический поиск в `outputs/`
- `infer.checkpoint_s3_uri` — S3 URI для прямой загрузки checkpoint (скачивается во временный файл)
- `infer.device` — устройство для инференса: `auto`, `cpu`, `cuda`
- `infer.top_k` — количество топ-классов для вывода (по умолчанию 5)

**Примечания:**

- По умолчанию `infer.checkpoint_path=null` — используется автоматический поиск последней модели из `outputs/`
- `infer()` автоматически вызывает `dvc pull` для `checkpoint_path`, если модель в DVC tracking
- Если `checkpoint_path` не найден, используется автоматический поиск, затем `checkpoint_s3_uri` (если указан)
- **Рекомендуется:** для продакшена использовать DVC-tracked checkpoint — быстрее и надёжнее

**Пример: Скачать модель из S3 и использовать для инференса:**

```powershell
# Шаг 1: Скачать модель из S3 и добавить в DVC tracking
python -m uv run cassava-fire download_model

# Шаг 2: Запустить инференс (автоматически найдёт модель в artifacts/)
python -m uv run cassava-fire infer --image data/cassava/test_image/2216849948.jpg
```

#### FastAPI сервер

Для запуска REST API сервера:

```powershell
python -m uv run uvicorn --factory cassava_leaf_disease.serving.app:create_app --host 127.0.0.1 --port 8000
```

**Эндпоинты:**

- `GET /health` — проверка работоспособности сервера
- `POST /predict` — предсказание (JPEG/PNG → JSON)

**Переменные окружения:**

- `CASSAVA_CHECKPOINT_PATH=artifacts/best.ckpt` — путь к чекпоинту (DVC-tracked)
- `CASSAVA_CHECKPOINT_S3_URI=s3://...` — альтернатива: S3 URI для прямой загрузки
- `CASSAVA_DEVICE=auto` (или `cpu` / `cuda`)

### Другие команды Fire CLI

#### download_data — скачать данные

Скачивает и распаковывает датасет из публичной ссылки.

```powershell
# Скачать данные
python -m uv run cassava-fire download_data

# Принудительно перескачать (даже если уже есть)
python -m uv run cassava-fire download_data --force
```

**Параметры:**

| Параметр     | Тип    | Описание                                                 | Пример                  |
| ------------ | ------ | -------------------------------------------------------- | ----------------------- |
| `--force`    | `bool` | Принудительно перескачать данные, даже если они уже есть | `--force`               |
| `*overrides` | `str`  | Дополнительные Hydra overrides                           | `download_data.url=...` |

#### download_model — скачать модель из S3

Скачивает checkpoint из S3 и добавляет в DVC tracking.

```powershell
# Скачать модель (использует URI из configs/download_model.yaml)
python -m uv run cassava-fire download_model

# С указанием S3 URI
python -m uv run cassava-fire download_model `
  --s3_uri s3://mlops-cassava-project/cassava/models/.../best.ckpt

# С автоматической отправкой в DVC remote
python -m uv run cassava-fire download_model --push
```

**Параметры:**

| Параметр      | Тип    | Описание                                                       | Пример                               |
| ------------- | ------ | -------------------------------------------------------------- | ------------------------------------ |
| `--s3_uri`    | `str`  | S3 URI checkpoint для скачивания                               | `--s3_uri s3://bucket/key/best.ckpt` |
| `--dst_dir`   | `str`  | Локальная директория для сохранения (по умолчанию `artifacts`) | `--dst_dir artifacts`                |
| `--overwrite` | `bool` | Перезаписать существующий файл                                 | `--overwrite`                        |
| `--push`      | `bool` | Отправить в DVC remote после добавления                        | `--push`                             |
| `*overrides`  | `str`  | Дополнительные Hydra overrides                                 | `download_model.remote=yandex_s3`    |

#### raw — прямой вызов Hydra команд

Для прямого вызова любых Hydra команд без обёртки Fire.

```powershell
# Прямой вызов Hydra команды
python -m uv run cassava-fire raw train train.epochs=1 logger.enabled=false

# С любыми параметрами
python -m uv run cassava-fire raw infer infer.image_path=data/cassava/test_image/2216849948.jpg
```

### Данные (DVC)

Данные не хранятся в Git. В Git попадают только `.dvc` метафайлы.

**Ожидаемая структура (относительно корня проекта):**

- `data/cassava/train.csv`
- `data/cassava/train_images/*.jpg`

**Для получения данных (public HTTP):**

Репозиторий настроен так, что дефолтный remote — `public_http`:

- `https://storage.yandexcloud.net/mlops-cassava-project/cassava`

Команда:

```powershell
python -m uv run dvc pull
```

**Для мейнтейнеров (push в S3 с кредами):**

Креды НЕ коммитим. Храним локально через `.dvc/config.local`:

```powershell
python -m uv run dvc remote modify --local yandex_s3 access_key_id "YOUR_ACCESS_KEY"
python -m uv run dvc remote modify --local yandex_s3 secret_access_key "YOUR_SECRET_KEY"
python -m uv run dvc remote modify --local yandex_s3 endpointurl "https://storage.yandexcloud.net"
python -m uv run dvc remote modify --local yandex_s3 region "ru-central1"

python -m uv run dvc push -r yandex_s3
```

### Tests & Coverage

**Запуск тестов:**

```powershell
python -m uv run pytest
```

**Покрытие кода (в проекте настроено `fail_under = 95`):**

```powershell
python -m uv run pytest --cov=cassava_leaf_disease --cov-report=term-missing
```

### Changelog (SemVer + Conventional Commits)

Проект использует Semantic Versioning и Conventional Commits для автоматической генерации changelog.

**Генерация changelog через Docker:**

```powershell
.\scripts\generate_changelog.ps1
git add CHANGELOG.md
git commit -m "chore(release): update changelog"
```

### Результаты (последний GPU-запуск на реальных данных)

Последний запуск (лимит 25 минут, остановка по `train.max_time`):

- **MLflow run**: `ac0e608a1a774095aabb0f0c87b4bbb8` — `http://127.0.0.1:8080/#/experiments/1/runs/ac0e608a1a774095aabb0f0c87b4bbb8`
- **val/acc**: `0.686`
- **val/f1_macro**: `0.584`
- **val/loss**: `1.302`
- **train/acc**: `0.917`
- **train/loss_epoch**: `0.516`
- **Лучшая модель (best.ckpt) в S3**:
  - `s3://mlops-cassava-project/cassava/models/2882636f460823a9839bce138ce9adbffe968d00/ac0e608a1a774095aabb0f0c87b4bbb8/best.ckpt`
  - метрики (JSON): `s3://mlops-cassava-project/cassava/models/2882636f460823a9839bce138ce9adbffe968d00/ac0e608a1a774095aabb0f0c87b4bbb8/metrics.json`

### Roadmap

- Image size 380/512 и более сильные аугментации
- Label smoothing / методы для noisy labels
- Экспорт модели и "реальный" инференс (FastAPI)
- Triton/TensorRT/ансамбли — отдельно, без влияния на основной пайплайн

---

## Примечание

Этот проект был разработан и протестирован с помощью [Cursor](https://cursor.sh) на Windows 10.
