"""Интерактивный обзор потока карточек по материалам практик 2.5 — 2.6."""
import pandas as pd
from taipy.gui import Markdown, State, notify

import charts
from data import filter_cards, kpis, load_cards, options, table_view

_cards = load_cards()
_cards = _cards[_cards["Целевой класс"].isin(["Инцидент (Решён)", "False Positive"])].reset_index(drop=True)
_first_day, _last_day = _cards["Дата"].min(), _cards["Дата"].max()

date_from = _first_day.to_pydatetime()
date_to = _last_day.to_pydatetime()
criticality_lov = options(_cards, "Критичность")
criticality_sel: list[str] = []
source_lov = options(_cards, "Источник инцидента")
source_sel: list[str] = []
verdict_lov = ["Инцидент (Решён)", "False Positive"]
verdict_sel: list[str] = []

period = ""
kpi_total = kpi_positive = kpi_fp = ""
fig_monthly = fig_verdicts = fig_criticality = None
cards_table = pd.DataFrame()
empty_state = False


def _to_day(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _view(cards: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    stats = kpis(cards)
    return {
        "kpi_total": f"{stats['total']:,}".replace(",", " "),
        "kpi_positive": f"{stats['positive']:,}".replace(",", " "),
        "kpi_fp": f"{stats['negative']:,}".replace(",", " "),
        "fig_monthly": charts.monthly_classes(cards) if len(cards) else None,
        "fig_verdicts": charts.verdict_counts(cards) if len(cards) else None,
        "fig_criticality": charts.criticality_counts(cards) if len(cards) else None,
        "cards_table": table_view(cards),
        "period": (
            f"{start:%d.%m.%Y} — {end:%d.%m.%Y} · UTC · {len(cards):,} карточек"
        ).replace(",", " "),
        "empty_state": cards.empty,
    }


globals().update(_view(_cards, _first_day, _last_day))


def on_filter(state: State, var_name: str = "", value=None):
    start, end = _to_day(state.date_from), _to_day(state.date_to)
    if start > end:
        notify(state, "warning", "Дата начала позже даты окончания.")
        return
    subset = filter_cards(
        _cards,
        start,
        end,
        state.criticality_sel,
        state.source_sel,
        state.verdict_sel,
    )
    with state as s:
        for name, val in _view(subset, start, end).items():
            setattr(s, name, val)


def on_reset(state: State):
    with state as s:
        s.date_from, s.date_to = _first_day.to_pydatetime(), _last_day.to_pydatetime()
        s.criticality_sel, s.source_sel, s.verdict_sel = [], [], []
    on_filter(state)


page = Markdown("""
# Бинарная выборка карточек SOC

<|{period}|text|class_name=subtitle|>

<|layout|columns=280px 1|gap=16px|
<|part|class_name=card filters|
### Фильтры

<|{date_from}|date|label=С|format=dd.MM.yyyy|on_change=on_filter|class_name=fullwidth|>

<|{date_to}|date|label=По|format=dd.MM.yyyy|on_change=on_filter|class_name=fullwidth|>

<|{criticality_sel}|selector|lov={criticality_lov}|multiple|dropdown|label=Критичность|on_change=on_filter|class_name=fullwidth|>

<|{source_sel}|selector|lov={source_lov}|multiple|dropdown|label=Источник SIEM|on_change=on_filter|class_name=fullwidth|>

<|{verdict_sel}|selector|lov={verdict_lov}|multiple|dropdown|label=Класс карточки|on_change=on_filter|class_name=fullwidth|>

<|Сбросить фильтры|button|on_action=on_reset|>
|>

<|
<|layout|columns=1 1 1|gap=12px|
<|part|class_name=card|
**Карточек в бинарной выборке**
<|{kpi_total}|text|class_name=kpi|>
|>
<|part|class_name=card|
**Решён**
<|{kpi_positive}|text|class_name=kpi|>
|>
<|part|class_name=card|
**False Positive**
<|{kpi_fp}|text|class_name=kpi|>
|>
|>

<|part|render={empty_state}|class_name=card|
Нет карточек по выбранным фильтрам.
|>

<|layout|columns=1 1|gap=16px|
<|part|render={not empty_state}|class_name=card|
### Изменение целевых классов по месяцам
<|chart|figure={fig_monthly}|>
|>
<|part|render={not empty_state}|class_name=card|
### Наиболее частые вердикты
<|chart|figure={fig_verdicts}|>
|>
|>

<|part|render={not empty_state}|class_name=card|
### Карточки по критичности
<|chart|figure={fig_criticality}|>
|>

<|part|render={not empty_state}|class_name=card|
### Карточки в выбранном срезе
<|{cards_table}|table|page_size=20|>
|>
|>
|>
""")
