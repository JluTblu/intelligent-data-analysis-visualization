"""Загрузка baseline-модели и асинхронная оценка карточек без вердикта."""
import json
import threading
import time
from collections import OrderedDict, Counter
from pathlib import Path

import joblib
import pandas as pd

from features import RAW_COLUMNS, build_features

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "iavd_ml_dataset.parquet"
MODEL_PATH = PROJECT_DIR / "models" / "iavd_baseline.joblib"
METRICS_PATH = PROJECT_DIR / "models" / "iavd_baseline_metrics.json"
TARGET = "Вердикт"
PAGE_COLUMNS = ["EntityId", "Создан", TARGET, "Критичность", "Пиктограмма IRP", "Тип инцидента", "Источник инцидента", "Наименование"]
MAX_CACHED_SESSIONS = 8

_scores_by_state: OrderedDict[str, pd.DataFrame] = OrderedDict()
_scores_lock = threading.Lock()


def count_unlabeled() -> int:
    """Число карточек с пустым вердиктом, до выполнения модельного инференса."""
    labels = pd.read_parquet(DATA_PATH, columns=[TARGET])[TARGET]
    return int(labels.isna().sum())


def load_metadata() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def selected_threshold(k: str, strategy: str) -> float:
    metadata = load_metadata()
    options = metadata["time_split"]["cost_thresholds"][str(k)]
    key = "minimum_cost_strategy" if strategy == "cost" else "maximum_fbeta_strategy"
    return float(options[key]["threshold"])


def score_unlabeled(state_id: str) -> dict:
    """Строит признаки и оценивает все карточки без вердикта в фоновом потоке."""
    started = time.perf_counter()
    raw = pd.read_parquet(DATA_PATH, columns=list(dict.fromkeys([*PAGE_COLUMNS, *RAW_COLUMNS])))
    queue = raw[raw[TARGET].isna()].copy()
    if queue.empty:
        raise ValueError("В наборе не найдены карточки без вердикта")
    parse_errors: Counter[str] = Counter()
    X = build_features(queue, parse_errors)
    model = joblib.load(MODEL_PATH)
    queue["Оценка инцидента"] = model.predict_proba(X)[:, 1]

    result = queue[[*PAGE_COLUMNS, "Оценка инцидента"]].copy()
    result["Создан"] = pd.to_datetime(result["Создан"], errors="coerce", utc=True).dt.tz_convert(None)
    result["Оценка инцидента"] = result["Оценка инцидента"].astype(float)
    with _scores_lock:
        _scores_by_state[state_id] = result
        _scores_by_state.move_to_end(state_id)
        while len(_scores_by_state) > MAX_CACHED_SESSIONS:
            _scores_by_state.popitem(last=False)

    return {
        "rows": len(result),
        "seconds": round(time.perf_counter() - started, 2),
        "json_parse_errors": dict(parse_errors),
    }


def queue_for_threshold(state_id: str, threshold: float) -> tuple[pd.DataFrame, dict]:
    """Получает список приоритета по сохранённым оценкам без повторного инференса."""
    with _scores_lock:
        scores = _scores_by_state.get(state_id)
        if scores is None:
            return pd.DataFrame(), {"total": 0, "flagged": 0, "share": 0.0}
        _scores_by_state.move_to_end(state_id)
        scores = scores.copy()

    flagged = scores[scores["Оценка инцидента"] >= threshold].sort_values(
        "Оценка инцидента", ascending=False
    )
    flagged = flagged.rename(columns={"Оценка инцидента": "Оценка класса «Решён»"})
    table = flagged.rename(columns={
        "EntityId": "EntityId",
        "Создан": "Создан (UTC)",
        "Критичность": "Критичность",
        "Пиктограмма IRP": "Категория IRP",
        "Тип инцидента": "Тип инцидента",
        "Источник инцидента": "Источник SIEM",
        "Наименование": "Правило",
    })[
        ["EntityId", "Создан (UTC)", "Критичность", "Категория IRP", "Тип инцидента", "Источник SIEM", "Правило", "Оценка класса «Решён»"]
    ].head(500).reset_index(drop=True)
    return table, {
        "total": len(scores),
        "flagged": len(flagged),
        "share": len(flagged) / len(scores) if len(scores) else 0.0,
    }


def evaluation_for(k: str, strategy: str) -> dict:
    metadata = load_metadata()
    options = metadata["time_split"]["cost_thresholds"][str(k)]
    key = "minimum_cost_strategy" if strategy == "cost" else "maximum_fbeta_strategy"
    choice = options[key]
    metrics = choice["test_metrics"]
    matrix = metrics["confusion_matrix"]
    return {
        "threshold": float(choice["threshold"]),
        "beta": float(options["beta"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "review_rate": float(metrics["review_rate"]),
        "cost": int(metrics["cost_fp_plus_k_fn"]),
        "fp": int(matrix["fp"]),
        "fn": int(matrix["fn"]),
        "test_rows": int(metadata["time_split"]["test_rows"]),
        "all_review_cost": round(
            int(metadata["time_split"]["test_rows"])
            * (1 - float(metadata["time_split"]["metrics_at_threshold_0_5"]["positive_share"]))
        ),
    }
