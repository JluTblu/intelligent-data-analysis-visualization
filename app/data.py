"""Чтение и фильтрация обновлённых карточек SOC для Taipy."""
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "iavd_ml_dataset.parquet"
DISPLAY_COLUMNS = [
    "EntityId",
    "Создан",
    "Вердикт",
    "Критичность",
    "Пиктограмма IRP",
    "Тип инцидента",
    "Источник инцидента",
    "Наименование",
]
TARGET_POSITIVE = "Решён"
TARGET_NEGATIVE = "False Positive"


def display_json_values(value) -> str:
    if pd.isna(value):
        return "Не указано"
    text = str(value)
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    values = decoded if isinstance(decoded, list) else [decoded]
    unique = list(dict.fromkeys(str(item) for item in values if item is not None))
    return " + ".join(unique) if unique else "Не указано"


@lru_cache(maxsize=1)
def load_cards() -> pd.DataFrame:
    """Загружает нужные приложению поля один раз на процесс; даты показывает в UTC."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден датасет: {DATA_PATH}")
    cards = pd.read_parquet(DATA_PATH, columns=DISPLAY_COLUMNS)
    created = pd.to_datetime(cards["Создан"], errors="coerce", utc=True)
    cards["Создан"] = created.dt.tz_convert(None)
    cards["Дата"] = cards["Создан"].dt.normalize()
    cards["Месяц"] = cards["Создан"].dt.strftime("%Y-%m")

    cards["Источник инцидента"] = cards["Источник инцидента"].map(display_json_values)
    cards["Тип инцидента"] = cards["Тип инцидента"].map(display_json_values)
    cards["Целевой класс"] = cards["Вердикт"].map(
        {TARGET_POSITIVE: "Инцидент (Решён)", TARGET_NEGATIVE: "False Positive"}
    )
    cards.loc[cards["Вердикт"].isna(), "Целевой класс"] = "Не размечено"
    cards["Целевой класс"] = cards["Целевой класс"].fillna("Другой вердикт")
    return cards.sort_values("Создан", ascending=False, ignore_index=True)


def kpis(cards: pd.DataFrame) -> dict[str, int | float]:
    """Разделяет бинарные метки, пустые вердикты и прочие известные вердикты."""
    labeled = cards[cards["Целевой класс"].isin(["Инцидент (Решён)", "False Positive"])]
    positive = int((labeled["Целевой класс"] == "Инцидент (Решён)").sum())
    negative = int((labeled["Целевой класс"] == "False Positive").sum())
    return {
        "total": len(cards),
        "labeled": len(labeled),
        "positive": positive,
        "negative": negative,
        "unlabeled": int(cards["Вердикт"].isna().sum()),
        "other_verdicts": int((cards["Целевой класс"] == "Другой вердикт").sum()),
    }


def options(cards: pd.DataFrame, column: str) -> list[str]:
    """Сортированные по частоте непустые значения для фильтра."""
    return cards[column].dropna().astype(str).value_counts().index.tolist()


def filter_cards(
    cards: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    criticalities: list[str],
    sources: list[str],
    verdicts: list[str],
) -> pd.DataFrame:
    mask = cards["Дата"].between(start, end)
    if criticalities:
        mask &= cards["Критичность"].isin(criticalities)
    if sources:
        mask &= cards["Источник инцидента"].isin(sources)
    if verdicts:
        mask &= cards["Целевой класс"].isin(verdicts)
    return cards.loc[mask]


def table_view(cards: pd.DataFrame) -> pd.DataFrame:
    columns = ["EntityId", "Создан", "Критичность", "Целевой класс", "Тип инцидента", "Источник инцидента", "Наименование"]
    table = cards[columns].reset_index(drop=True)
    table["Создан"] = table["Создан"].dt.strftime("%d.%m.%Y %H:%M:%S")
    return table.rename(columns={"Создан": "Создан (UTC)"})
