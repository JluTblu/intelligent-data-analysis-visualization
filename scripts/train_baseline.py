"""Обучает базовую классификационную модель на подготовленных признаках."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA = PROJECT_DIR / "data" / "processed" / "iavd_baseline_features_candidate.parquet"
BINARY_DATA = PROJECT_DIR / "data" / "processed" / "iavd_binary_labeled.parquet"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "iavd_baseline.joblib"
METRICS_PATH = MODEL_DIR / "iavd_baseline_metrics.json"
RANDOM_PREDICTIONS_PATH = PROJECT_DIR / "data" / "processed" / "iavd_baseline_random_test_predictions.parquet"
TIME_PREDICTIONS_PATH = PROJECT_DIR / "data" / "processed" / "iavd_baseline_time_test_predictions.parquet"
SHAP_SAMPLE_PATH = MODEL_DIR / "iavd_baseline_shap.npz"
KEY = "EntityId"
TARGET = "y"
THRESHOLD = 0.5
RANDOM_STATE = 42
COST_RATIOS = (50, 100)


def make_model(categorical: list[str], numeric: list[str]) -> Pipeline:
    encoder = ColumnTransformer(
        transformers=[
            ("categorical", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            ), categorical),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("encode", encoder),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )),
    ])


def evaluate(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = THRESHOLD,
) -> tuple[dict, np.ndarray, np.ndarray]:
    y_true = y.to_numpy()
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    report = {
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "review_rate": float(y_pred.mean()),
        "positive_share": float(y.mean()),
        "cost_fp_plus_k_fn": None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return report, y_score, y_pred


def select_cost_threshold(y: pd.Series, scores: np.ndarray, k: int) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    positives = int(y.sum())
    true_positives = recall[:-1] * positives
    predicted_positives = np.divide(
        true_positives,
        precision[:-1],
        out=np.zeros_like(true_positives),
        where=precision[:-1] > 0,
    )
    false_negatives = positives - true_positives
    false_positives = predicted_positives - true_positives
    costs = false_positives + k * false_negatives
    best_index = int(np.argmin(costs))
    return float(thresholds[best_index]), float(costs[best_index])


def select_fbeta_threshold(y: pd.Series, scores: np.ndarray, beta: float) -> tuple[float, float]:
    """Порог максимума F_beta по курсовому правилу сравнения стратегий."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    beta_squared = beta**2
    denominator = beta_squared * precision[:-1] + recall[:-1]
    fbeta = np.divide(
        (1 + beta_squared) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(fbeta))
    return float(thresholds[best_index]), float(fbeta[best_index])


def save_shap_sample(model: Pipeline, X: pd.DataFrame, y: pd.Series, ids: pd.Series, scores: np.ndarray) -> int:
    """Предрассчитывает SHAP для глобальной страницы XAI; модель та же, что проверялась по времени."""
    import shap

    sample_ids = X.sample(n=min(1000, len(X)), random_state=RANDOM_STATE).index
    encoded = model.named_steps["encode"].transform(X.loc[sample_ids])
    estimator = model.named_steps["classifier"]
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(encoded)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    base = float(np.ravel(explainer.expected_value)[-1])
    np.savez_compressed(
        SHAP_SAMPLE_PATH,
        values=values,
        base=base,
        features=np.asarray(X.columns, dtype=str),
        data=X.loc[sample_ids].astype(str).to_numpy(dtype=str),
        entity_id=ids.loc[sample_ids].to_numpy(),
        y=y.loc[sample_ids].to_numpy(),
        score=scores[X.index.get_indexer(sample_ids)],
    )
    return len(sample_ids)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Не найден набор признаков: {DATA}")
    if not BINARY_DATA.exists():
        raise FileNotFoundError(f"Не найден бинарный набор: {BINARY_DATA}")

    data = pd.read_parquet(DATA)
    feature_columns = [name for name in data.columns if name not in {KEY, TARGET}]
    categorical = [name for name in feature_columns if data[name].dtype == object]
    numeric = [name for name in feature_columns if name not in categorical]

    X = data[feature_columns]
    y = data[TARGET].astype("int8")
    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_model(categorical, numeric)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    test = data.iloc[test_idx]
    random_metrics, y_score, y_pred = evaluate(model, X.iloc[test_idx], y.iloc[test_idx])
    random_y_true = y.iloc[test_idx].to_numpy()

    dates = pd.read_parquet(BINARY_DATA, columns=[KEY, "Создан"])
    if not dates[KEY].reset_index(drop=True).equals(data[KEY].reset_index(drop=True)):
        raise ValueError("Порядок карточек не совпадает между бинарным набором и матрицей признаков")
    created = pd.to_datetime(dates["Создан"], errors="coerce", utc=True)
    cutoff_train = created.quantile(0.7)
    cutoff_calibration = created.quantile(0.8)
    time_train = created < cutoff_train
    time_calibration = (created >= cutoff_train) & (created < cutoff_calibration)
    time_test = created >= cutoff_calibration
    time_model = make_model(categorical, numeric)
    time_model.fit(X.loc[time_train], y.loc[time_train])
    calibration_scores = time_model.predict_proba(X.loc[time_calibration])[:, 1]
    time_metrics, time_score, time_pred_05 = evaluate(time_model, X.loc[time_test], y.loc[time_test])
    time_test_rows = data.loc[time_test]
    time_y_true = y.loc[time_test].to_numpy()
    shap_sample_rows = save_shap_sample(
        time_model,
        X.loc[time_test],
        y.loc[time_test],
        time_test_rows[KEY],
        time_score,
    )
    thresholds = {}
    time_predictions = {
        KEY: time_test_rows[KEY].to_numpy(),
        "y_true": time_y_true,
        "y_score": time_score,
        "y_pred_0_5": time_pred_05,
    }
    for k in COST_RATIOS:
        cost_threshold, calibration_cost = select_cost_threshold(
            y.loc[time_calibration], calibration_scores, k
        )
        cost_metrics, _, cost_prediction = evaluate(
            time_model, X.loc[time_test], y.loc[time_test], cost_threshold
        )
        cost_confusion = cost_metrics["confusion_matrix"]
        cost_metrics["cost_fp_plus_k_fn"] = cost_confusion["fp"] + k * cost_confusion["fn"]
        cost_metrics["threshold"] = cost_threshold

        beta = float(np.sqrt(k))
        fbeta_threshold, calibration_fbeta = select_fbeta_threshold(
            y.loc[time_calibration], calibration_scores, beta
        )
        fbeta_metrics, _, fbeta_prediction = evaluate(
            time_model, X.loc[time_test], y.loc[time_test], fbeta_threshold
        )
        fbeta_confusion = fbeta_metrics["confusion_matrix"]
        fbeta_metrics["cost_fp_plus_k_fn"] = fbeta_confusion["fp"] + k * fbeta_confusion["fn"]
        fbeta_metrics["threshold"] = fbeta_threshold
        thresholds[str(k)] = {
            "beta": beta,
            "minimum_cost_strategy": {
                "threshold": cost_threshold,
                "calibration_cost": calibration_cost,
                "test_metrics": cost_metrics,
            },
            "maximum_fbeta_strategy": {
                "threshold": fbeta_threshold,
                "calibration_fbeta": calibration_fbeta,
                "test_metrics": fbeta_metrics,
            },
        }
        time_predictions[f"y_pred_cost_k{k}"] = cost_prediction
        time_predictions[f"y_pred_fbeta_k{k}"] = fbeta_prediction

    metrics = {
        "target": {"1": "Решён", "0": "False Positive"},
        "random_split": {
            "test_size": 0.2,
            "stratified": True,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "metrics_at_threshold": random_metrics,
        },
        "time_split": {
            "rule": "train on earlier 70%, choose threshold on next 10%, test on latest 20% by Создан",
            "train_cutoff_utc": cutoff_train.isoformat(),
            "calibration_cutoff_utc": cutoff_calibration.isoformat(),
            "train_rows": int(time_train.sum()),
            "calibration_rows": int(time_calibration.sum()),
            "test_rows": int(time_test.sum()),
            "train_start_utc": created[time_train].min().isoformat(),
            "train_end_utc": created[time_train].max().isoformat(),
            "calibration_start_utc": created[time_calibration].min().isoformat(),
            "calibration_end_utc": created[time_calibration].max().isoformat(),
            "test_start_utc": created[time_test].min().isoformat(),
            "test_end_utc": created[time_test].max().isoformat(),
            "metrics_at_threshold_0_5": time_metrics,
            "cost_thresholds": thresholds,
        },
        "model": {
            "class": "HistGradientBoostingClassifier",
            "max_iter": 200,
            "learning_rate": 0.08,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
        },
        "feature_columns": feature_columns,
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "threshold": THRESHOLD,
        "threshold_comparison": "On the calibration period, compare minimum Cost(tau)=k*FN+FP against maximum F_beta with beta=sqrt(k); evaluate both on the later test period.",
        "shap_sample": {"path": str(SHAP_SAMPLE_PATH.relative_to(PROJECT_DIR)), "rows": shap_sample_rows},
        "cost_ratio_definition": "cost = FP + k * FN; evaluate the course-specified k=50 and k=100 scenarios",
        "positive_share": float(y.mean()),
        "sklearn_version": sklearn.__version__,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(time_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({
        KEY: test[KEY].to_numpy(),
        "y_true": random_y_true,
        "y_score": y_score,
        "y_pred_0_5": y_pred,
    }).to_parquet(RANDOM_PREDICTIONS_PATH, index=False, compression="zstd")
    pd.DataFrame(time_predictions).to_parquet(
        TIME_PREDICTIONS_PATH, index=False, compression="zstd"
    )

    print(f"Модель: {MODEL_PATH}")
    print(f"Отчёт: {METRICS_PATH}")
    print(f"Случайные проверочные предсказания: {RANDOM_PREDICTIONS_PATH}")
    print(f"Временные предсказания: {TIME_PREDICTIONS_PATH}")
    print("Случайная проверка:")
    print(json.dumps(random_metrics, ensure_ascii=False, indent=2))
    print("Поздние даты при пороге 0,5:")
    print(json.dumps(time_metrics, ensure_ascii=False, indent=2))
    print("Сравнение порога минимальной стоимости и максимума F_beta:")
    print(json.dumps(thresholds, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
