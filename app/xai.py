"""Вычисления SHAP и PDP/ICE для страницы объяснимости."""
from functools import lru_cache
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

import charts
from advisor import METRICS_PATH, MODEL_PATH
from features import CATEGORICAL, FEATURES, NUMERIC

PROJECT_DIR = Path(__file__).resolve().parents[1]
SHAP_PATH = PROJECT_DIR / "models" / "iavd_baseline_shap.npz"
FEATURES_PATH = PROJECT_DIR / "data" / "processed" / "iavd_baseline_features_candidate.parquet"
_sample = np.load(SHAP_PATH, allow_pickle=False)
SHAP_VALUES = _sample["values"]
SHAP_DATA = _sample["data"]
SHAP_FEATURES = _sample["features"].astype(str).tolist()
SAMPLE_IDS = _sample["entity_id"]
SAMPLE_Y = _sample["y"].astype(int)
SAMPLE_SCORES = _sample["score"].astype(float)
BASE_VALUE = float(_sample["base"])

_metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
DEFAULT_THRESHOLD = float(
    _metadata["time_split"]["cost_thresholds"]["50"]["minimum_cost_strategy"]["threshold"]
)


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def pdp_sample() -> pd.DataFrame:
    data = pd.read_parquet(FEATURES_PATH, columns=["EntityId", *FEATURES])
    data = data.set_index("EntityId")
    return data.reindex(SAMPLE_IDS)[FEATURES].reset_index(drop=True)


def local_example(name: str, threshold: float = DEFAULT_THRESHOLD) -> tuple[object, str]:
    predicted = SAMPLE_SCORES >= threshold
    cases = {
        "TP": np.where((SAMPLE_Y == 1) & predicted)[0],
        "FN": np.where((SAMPLE_Y == 1) & ~predicted)[0],
        "FP": np.where((SAMPLE_Y == 0) & predicted)[0],
        "TN": np.where((SAMPLE_Y == 0) & ~predicted)[0],
    }
    prefix = name.split("—", 1)[0].strip()
    indexes = cases.get(prefix, np.array([], dtype=int))
    if not len(indexes):
        return None, "Для этого исхода нет карточки в SHAP-выборке."
    if name.startswith("TP"):
        index = indexes[np.argmax(SAMPLE_SCORES[indexes])]
    elif name.startswith("FN"):
        index = indexes[np.argmin(SAMPLE_SCORES[indexes])]
    elif name.startswith("FP"):
        index = indexes[np.argmax(SAMPLE_SCORES[indexes])]
    else:
        index = indexes[np.argmin(SAMPLE_SCORES[indexes])]

    fig = charts.shap_waterfall(
        SHAP_VALUES[index],
        SHAP_FEATURES,
        BASE_VALUE,
        SHAP_DATA[index],
    )
    y_label = "Решён (y=1)" if SAMPLE_Y[index] == 1 else "False Positive (y=0)"
    prediction = "выше порога — в очередь" if predicted[index] else "ниже порога"
    details = (
        f"**EntityId:** {int(SAMPLE_IDS[index])}  \n"
        f"Фактический вердикт: **{y_label}**  \n"
        f"Оценка модели: **{SAMPLE_SCORES[index]:.1%}**; {prediction} (τ={threshold:.3f})."
    ).replace(".", ",")
    return fig, details


def pdp_grid(data: pd.DataFrame, feature: str, points: int = 15, top: int = 10) -> list:
    values = data[feature]
    if feature in NUMERIC:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        return sorted(numeric.quantile(np.linspace(0.05, 0.95, points)).round(2).unique().tolist())
    return values.astype(str).value_counts().head(top).index.tolist()


def compute_pdp(feature: str):
    model = load_model()
    data = pdp_sample()
    grid = pdp_grid(data, feature)
    ice = np.empty((len(data), len(grid)), dtype=float)
    for index, value in enumerate(grid):
        modified = data.copy()
        modified[feature] = value
        ice[:, index] = model.predict_proba(modified)[:, 1]
    figure = charts.pdp_ice(
        feature,
        grid,
        ice.mean(axis=0),
        ice,
        data[feature],
        DEFAULT_THRESHOLD,
        feature in NUMERIC,
    )
    return figure
