# autograd-ml

Проект решает задачу табличной регрессии: по признакам объявления (год, пробег, параметры комплектации и прочее) предсказывается целевая переменная `price`

Помимо числовых и категориальных признаков используются текстовые признаки описания объявления. Для обработки описаний применяется модель cointegrated/rubert-tiny2 (`pytorch` + SentenceTransformer), с помощью которой генерируются эмбеддинги текста. Полученные векторные признаки добавляются в табличные данные и подаются на вход модели `CatBoost`.

Данные из продовой `MySQL` выгружаются заранее (чтобы при проверке не требовались .env и доступы к БД) и публикуются в Google Drive. Далее данные управляются через `DVC`, а модель и эксперименты — `MLflow`.

## Стек

- Python package `src/autograd_tabular`
- **UV** - управление зависимостями (`pyproject.toml` + `uv.lock`)
- **DVC** - управление данными (remote = S3/MinIO)
- **Hydra** - конфиги (гиперпараметры, пути, mlflow uri и т.д.)
- **CatBoost** - обучение модели табличной регрессии
- **rubert-tiny2** - генеририрование эмбеддингов описания объявлений
- **MLflow** - логирование экспериментов и хранения модели (artifacts + S3)
- **pre-commit** - линтеры/форматтеры (ruff/yapf/prettier)

---

## Структура репозитория

Ключевые файлы:

- `src/autograd_tabular/download.py` — стадия загрузки данных (из открытого источника)
- `src/autograd_tabular/train.py` — обучение + логирование в MLflow
- `src/autograd_tabular/infer.py` — обработка + предсказание целевой переменной на основе входных данных
- `configs/` — конфиги Hydra (данные / препроцессинг / модель / mlflow)
- `dvc.yaml`, `dvc.lock` — пайплайн DVC
- `docker-compose.yml` — локальная инфраструктура (MinIO + MLflow)
- `.pre-commit-config.yaml` — хуки проверки проекта
- `pyproject.toml`, `uv.lock` — зависимости и настройки линтеров/форматирования

---

## Setup

### 1) Клонировать репозиторий

### 2) Поднять инфраструктуру (MinIO + MLflow)

```bash
docker compose up -d
```

Поднимаются сервисы:

- **MinIO** — S3 хранилище (для DVC remote и MLflow artifacts)
- **MLflow** — tracking server (логирование)

> В проекте MLflow использует S3 как `default-artifact-root`, поэтому модель сохраняется в S3 (MinIO).

### 4) Подготовка Minio для работы

- Перейти по адресу: http://localhost:5000
- Ввести логин и пароль: minioadmin
- Создать в Minio бакет models

### 5) Создать окружение и установить зависимости (uv)

```bash
uv sync --frozen
```

> `--frozen` гарантирует, что зависимости берутся из `uv.lock`

### 6) Запустить pre-commit

```bash
pre-commit install
pre-commit run -a
```

### 7) Инициализация DVC

1. Стадия загрузки данных:

```bash
dvc repro dvc.yaml:download
```

2. Залить данные в DVC remote (S3/MinIO), чтобы `pull` работал на чистом клоне:

```bash
dvc push dvc.yaml:download
```

---

## Train & Infer

В проекте обучение встроено в DVC пайплайн и доступно как прямой запуск Python-скрипта.

### Вариант A: DVC pipeline

Запустить полный пайплайн (download → train -> infer):

```bash
dvc repro dvc.yaml
```

### Вариант B: запуск обучения напрямую

```bash
uv run python src/autograd_tabular/train.py
```

Внутри `train.py` и `infer.py` встроена проверка наличия данных через DVC (pull) и логирование в MLflow.

## Конфигурация через Hydra

Все основные параметры вынесены в `configs/`:

- пути к данным / target / drop_cols
- препроцессинг (split, random_state, список метрик, преобразования типов)
- параметры модели (CatBoost, pytorch(rubert-tiny2))
- настройки MLflow (tracking uri, experiment name и т.д.)

Запуск с переопределением параметров:

```bash
uv run python src/autograd_tabular/train.py preprocess.loss_function=RMSE model.learning_rate=0.05
```

---

## Data management через DVC

Данные **не хранятся в git**. Управление датасетом делается через DVC:

- стадия `download` формирует `artifacts/raw/dataset.csv`
- после `dvc push dvc.yaml:download` датасет доступен из DVC remote (S3-MinIO)
- в обучении используется DVC (pull) для получения данных на чистом окружении

## MLflow logging

Что логируется в MLflow:

- гиперпараметры модели (`mlflow.log_params(...)`)
- основные метрики (`MAE`, `RMSE`, `R2`) (можно передавать любые в параметрах hydra)
- **функции потерь/метрики по итерациям** (4 графика)
- модель как артефакт (в S3/MinIO) через `mlflow.catboost.log_model(...)`

---
