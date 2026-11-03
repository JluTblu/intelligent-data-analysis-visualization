"""Строит небольшой смешанный набор признаков для базовой ML-модели."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT = PROJECT_DIR / "data" / "processed" / "iavd_binary_labeled.parquet"
OUTPUT = PROJECT_DIR / "data" / "processed" / "iavd_baseline_features_candidate.parquet"
MANIFEST = PROJECT_DIR / "data" / "processed" / "iavd_baseline_feature_manifest.csv"
KEY = "EntityId"
TARGET = "y"
CATEGORICAL = {
    "Критичность": "Исходное категориальное поле.",
    "Критичность инцидента в источнике": "Исходное категориальное поле.",
    "Тип инцидента": "Исходное категориальное поле.",
    "Источник инцидента": "Исходное категориальное поле.",
    "Вендор из сием": "Исходное категориальное поле.",
    "Конвейер SIEM": "Исходное категориальное поле.",
    "Наименование": "Исходное название правила.",
    "Пиктограмма IRP": "Исходная категория карточки.",
    "categoryHigh": "Первое значение JSON-массива.",
    "categoryLow": "Первое значение JSON-массива.",
    "categoryMain": "Первое значение JSON-массива.",
    "processName": "Первое значение JSON-массива.",
    "Имя родителя процесса": "Первое значение JSON-массива.",
    "object": "Исходное категориальное поле.",
    "Привилегии аккаунта": "Исходное категориальное поле.",
    "Теги": "Значения JSON-массива объединены через точку с запятой.",
}
NUMERIC = {
    "Час": "Час UTC из «Создан».",
    "День недели": "День недели из «Создан»: понедельник = 0.",
    "Количество инцидентов": "Преобразовано из строки в число.",
    "Алертов в карточке": "Число элементов JSON-массива «Алерт».",
    "Значений инцидента": "Число элементов JSON-массива группировки.",
    "Длина команды": "Число символов в объединённых командах.",
    "Есть powershell": "1, если команда содержит powershell, иначе 0.",
}
RAW_COLUMNS = [
    KEY, TARGET, "Создан", "Количество инцидентов", "Алерт",
    "Значение Инцидента (группировка)", "Команда", *CATEGORICAL,
]


def json_items(value, column: str, errors: Counter) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value)
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        errors[column] += 1
        return [text]
    values = decoded if isinstance(decoded, list) else [decoded]
    return [str(item) for item in values if item is not None]


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Не найден бинарный набор: {INPUT}")

    raw = pd.read_parquet(INPUT, columns=RAW_COLUMNS)
    result = pd.DataFrame({KEY: raw[KEY], TARGET: raw[TARGET]})
    parse_errors: Counter[str] = Counter()

    for column in CATEGORICAL:
        if column in {"categoryHigh", "categoryLow", "categoryMain", "processName", "Имя родителя процесса", "Теги"}:
            items = raw[column].map(lambda value: json_items(value, column, parse_errors))
            if column == "Теги":
                result[column] = items.map(lambda values: "; ".join(dict.fromkeys(values)) or "—")
            else:
                result[column] = items.map(lambda values: values[0] if values else "—")
        else:
            result[column] = raw[column].fillna("—").astype(str)

    created = pd.to_datetime(raw["Создан"], errors="coerce", utc=True)
    result["Час"] = created.dt.hour.astype("Int8")
    result["День недели"] = created.dt.dayofweek.astype("Int8")
    result["Количество инцидентов"] = pd.to_numeric(
        raw["Количество инцидентов"], errors="coerce"
    )

    alerts = raw["Алерт"].map(lambda value: json_items(value, "Алерт", parse_errors))
    grouped_values = raw["Значение Инцидента (группировка)"].map(
        lambda value: json_items(value, "Значение Инцидента (группировка)", parse_errors)
    )
    commands = raw["Команда"].map(lambda value: json_items(value, "Команда", parse_errors))
    command_text = commands.map(lambda values: " ".join(values))

    result["Алертов в карточке"] = alerts.map(len).astype("int32")
    result["Значений инцидента"] = grouped_values.map(len).astype("int32")
    result["Длина команды"] = command_text.str.len().fillna(0).astype("int32")
    result["Есть powershell"] = command_text.str.contains(
        "powershell", case=False, regex=False
    ).fillna(False).astype("int8")

    ordered_features = [*CATEGORICAL, *NUMERIC]
    result = result[[KEY, *ordered_features, TARGET]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False, compression="zstd")

    manifest = pd.DataFrame([
        {
            "Признак": name,
            "Тип": "категориальный" if name in CATEGORICAL else "числовой",
            "Пропусков": int(result[name].isna().sum()),
            "Уникальных значений": int(result[name].nunique(dropna=True)),
            "Преобразование": CATEGORICAL.get(name, NUMERIC.get(name, "")),
        }
        for name in ordered_features
    ])
    manifest.to_csv(MANIFEST, index=False, encoding="utf-8-sig")

    print(f"Карточек: {len(result):,}".replace(",", " "))
    print(f"Признаков: {len(ordered_features)} ({len(CATEGORICAL)} категориальных, {len(NUMERIC)} числовых)")
    print(f"Распределение y: {result[TARGET].value_counts().sort_index().to_dict()}")
    print(f"Преобразованный набор: {OUTPUT}")
    print(f"Описание признаков: {MANIFEST}")
    print(f"Ошибок разбора JSON-массивов: {dict(parse_errors)}")


if __name__ == "__main__":
    main()
