"""Сводит временные метрики базовой модели и простых политик проверки."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS = PROJECT_DIR / "data" / "processed" / "iavd_baseline_time_test_predictions.parquet"
LABELS = PROJECT_DIR / "data" / "processed" / "iavd_binary_labeled.parquet"
MONTHLY_OUTPUT = PROJECT_DIR / "outputs" / "monthly_model_evaluation.csv"
POLICY_OUTPUT = PROJECT_DIR / "outputs" / "temporal_policy_comparison.csv"
KEY = "EntityId"


def confusion(y: pd.Series, prediction: pd.Series) -> dict[str, int]:
    return {
        "FP": int(((y == 0) & (prediction == 1)).sum()),
        "FN": int(((y == 1) & (prediction == 0)).sum()),
        "TP": int(((y == 1) & (prediction == 1)).sum()),
        "TN": int(((y == 0) & (prediction == 0)).sum()),
    }


def main() -> None:
    predictions = pd.read_parquet(PREDICTIONS)
    labels = pd.read_parquet(LABELS, columns=[KEY, "y", "Создан"])
    data = predictions.merge(labels, on=KEY, validate="one_to_one", suffixes=("_pred", ""))
    if not data["y_true"].equals(data["y"]):
        raise ValueError("Целевые метки в прогнозах не совпадают с подготовленным набором")

    data["Дата"] = pd.to_datetime(data["Создан"], errors="coerce", utc=True)
    data["Месяц"] = data["Дата"].dt.strftime("%Y-%m")
    rows = []
    for month, group in data.groupby("Месяц", sort=True):
        y = group["y_true"]
        counts = confusion(y, group["y_pred_0_5"])
        cost_counts = confusion(y, group["y_pred_k50"])
        rows.append({
            "Месяц": month,
            "Карточек": len(group),
            "Решён, %": round(100 * float(y.mean()), 2),
            "Средняя точность AP": round(float(average_precision_score(y, group["y_score"])), 4),
            "ROC-AUC": round(float(roc_auc_score(y, group["y_score"])), 4),
            "Проверено при пороге 0.5, %": round(100 * float(group["y_pred_0_5"].mean()), 2),
            "FP при пороге 0.5": counts["FP"],
            "FN при пороге 0.5": counts["FN"],
            "Проверено при пороге k=50, %": round(100 * float(group["y_pred_k50"].mean()), 2),
            "FP при пороге k=50": cost_counts["FP"],
            "FN при пороге k=50": cost_counts["FN"],
            "Цена k=50": cost_counts["FP"] + 50 * cost_counts["FN"],
        })

    y = data["y_true"]
    policies = {
        "Порог 0.5": data["y_pred_0_5"],
        "Порог из калибровки k=50": data["y_pred_k50"],
        "Проверять все карточки": pd.Series(1, index=data.index),
        "Не проверять карточки": pd.Series(0, index=data.index),
    }
    comparison = []
    for name, prediction in policies.items():
        counts = confusion(y, prediction)
        total = counts["FP"] + counts["FN"] + counts["TP"] + counts["TN"]
        comparison.append({
            "Правило": name,
            "Карточек на проверку": int(prediction.sum()),
            "Доля на проверку, %": round(100 * float(prediction.mean()), 2),
            "Точность проверок": round(counts["TP"] / max(counts["TP"] + counts["FP"], 1), 4),
            "Полнота по Решён": round(counts["TP"] / max(counts["TP"] + counts["FN"], 1), 4),
            "FP": counts["FP"],
            "FN": counts["FN"],
            "Цена k=50": counts["FP"] + 50 * counts["FN"],
            "Цена k=100": counts["FP"] + 100 * counts["FN"],
            "Всего карточек": total,
        })

    MONTHLY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(MONTHLY_OUTPUT, index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison).to_csv(POLICY_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Помесячные метрики: {MONTHLY_OUTPUT}")
    print(f"Сравнение правил: {POLICY_OUTPUT}")


if __name__ == "__main__":
    main()
