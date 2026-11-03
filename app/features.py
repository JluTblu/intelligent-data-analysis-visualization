"""Единая сборка признаков для обучения и Taipy-инференса."""
import json
from collections import Counter

import pandas as pd

KEY = "EntityId"
CATEGORICAL = [
    "Критичность",
    "Критичность инцидента в источнике",
    "Тип инцидента",
    "Источник инцидента",
    "Вендор из сием",
    "Конвейер SIEM",
    "Наименование",
    "Пиктограмма IRP",
    "categoryHigh",
    "categoryLow",
    "categoryMain",
    "processName",
    "Имя родителя процесса",
    "object",
    "Привилегии аккаунта",
    "Теги",
]
NUMERIC = [
    "Час",
    "День недели",
    "Количество инцидентов",
    "Алертов в карточке",
    "Значений инцидента",
    "Длина команды",
    "Есть powershell",
]
FEATURES = CATEGORICAL + NUMERIC
JSON_CATEGORICAL = {
    "categoryHigh",
    "categoryLow",
    "categoryMain",
    "processName",
    "Имя родителя процесса",
    "Теги",
}
RAW_COLUMNS = [
    "Создан",
    "Количество инцидентов",
    "Алерт",
    "Значение Инцидента (группировка)",
    "Команда",
    *CATEGORICAL,
]


def _json_items(value, column: str, errors: Counter[str]) -> list[str]:
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


def build_features(raw: pd.DataFrame, errors: Counter[str] | None = None) -> pd.DataFrame:
    """Строит одинаковую таблицу признаков из сырых карточек любого целевого статуса."""
    parse_errors = errors if errors is not None else Counter()
    features = pd.DataFrame(index=raw.index)

    for column in CATEGORICAL:
        if column in JSON_CATEGORICAL:
            items = raw[column].map(lambda value: _json_items(value, column, parse_errors))
            if column == "Теги":
                features[column] = items.map(lambda values: "; ".join(dict.fromkeys(values)) or "—")
            else:
                features[column] = items.map(lambda values: values[0] if values else "—")
        else:
            features[column] = raw[column].fillna("—").astype(str)

    created = pd.to_datetime(raw["Создан"], errors="coerce", utc=True)
    features["Час"] = created.dt.hour.astype("Int8")
    features["День недели"] = created.dt.dayofweek.astype("Int8")
    features["Количество инцидентов"] = pd.to_numeric(raw["Количество инцидентов"], errors="coerce")

    alerts = raw["Алерт"].map(lambda value: _json_items(value, "Алерт", parse_errors))
    grouped_values = raw["Значение Инцидента (группировка)"].map(
        lambda value: _json_items(value, "Значение Инцидента (группировка)", parse_errors)
    )
    commands = raw["Команда"].map(lambda value: _json_items(value, "Команда", parse_errors))
    command_text = commands.map(lambda values: " ".join(values))
    features["Алертов в карточке"] = alerts.map(len).astype("int32")
    features["Значений инцидента"] = grouped_values.map(len).astype("int32")
    features["Длина команды"] = command_text.str.len().fillna(0).astype("int32")
    features["Есть powershell"] = command_text.str.contains(
        "powershell", case=False, regex=False
    ).fillna(False).astype("int8")
    return features[FEATURES]
