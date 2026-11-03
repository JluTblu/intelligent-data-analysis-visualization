"""Строит матрицу признаков для обучающей бинарной выборки."""
from collections import Counter
from pathlib import Path
import sys

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_DIR / "app"
sys.path.insert(0, str(APP_DIR))
from features import CATEGORICAL, FEATURES, KEY, NUMERIC, RAW_COLUMNS, build_features  # noqa: E402

INPUT = PROJECT_DIR / "data" / "processed" / "iavd_binary_labeled.parquet"
OUTPUT = PROJECT_DIR / "data" / "processed" / "iavd_baseline_features_candidate.parquet"
MANIFEST = PROJECT_DIR / "data" / "processed" / "iavd_baseline_feature_manifest.csv"
TARGET = "y"


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Не найден бинарный набор: {INPUT}")

    raw = pd.read_parquet(INPUT, columns=[KEY, TARGET, *RAW_COLUMNS])
    parse_errors: Counter[str] = Counter()
    features = build_features(raw, parse_errors)
    result = pd.concat([raw[[KEY, TARGET]], features], axis=1)
    result = result[[KEY, *FEATURES, TARGET]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False, compression="zstd")

    descriptions = {
        "categoryHigh": "Первое значение JSON-массива.",
        "categoryLow": "Первое значение JSON-массива.",
        "categoryMain": "Первое значение JSON-массива.",
        "processName": "Первое значение JSON-массива.",
        "Имя родителя процесса": "Первое значение JSON-массива.",
        "Теги": "Значения JSON-массива объединены через точку с запятой.",
        "Час": "Час UTC из «Создан».",
        "День недели": "День недели из «Создан»: понедельник = 0.",
        "Количество инцидентов": "Преобразовано из строки в число.",
        "Алертов в карточке": "Число элементов JSON-массива «Алерт».",
        "Значений инцидента": "Число элементов JSON-массива группировки.",
        "Длина команды": "Число символов в объединённых командах.",
        "Есть powershell": "1, если команда содержит powershell, иначе 0.",
    }
    manifest = pd.DataFrame([
        {
            "Признак": name,
            "Тип": "категориальный" if name in CATEGORICAL else "числовой",
            "Пропусков": int(result[name].isna().sum()),
            "Уникальных значений": int(result[name].nunique(dropna=True)),
            "Преобразование": descriptions.get(name, "Исходное поле, приведённое к строке."),
        }
        for name in FEATURES
    ])
    manifest.to_csv(MANIFEST, index=False, encoding="utf-8-sig")

    print(f"Карточек: {len(result):,}".replace(",", " "))
    print(f"Признаков: {len(FEATURES)} ({len(CATEGORICAL)} категориальных, {len(NUMERIC)} числовых)")
    print(f"Распределение y: {result[TARGET].value_counts().sort_index().to_dict()}")
    print(f"Преобразованный набор: {OUTPUT}")
    print(f"Описание признаков: {MANIFEST}")
    print(f"Ошибок разбора JSON-массивов: {dict(parse_errors)}")


if __name__ == "__main__":
    main()
