"""Страница объяснимости: глобальный SHAP, разбор исходов и PDP/ICE."""
import pandas as pd
import plotly.graph_objects as go
from taipy.gui import Markdown, State, invoke_long_callback, notify

import charts
import xai

_importance = pd.Series(abs(xai.SHAP_VALUES).mean(axis=0), index=xai.SHAP_FEATURES).sort_values(ascending=False)
feature_lov = _importance.index.tolist()
feature_sel = feature_lov[0]
fig_importance = charts.shap_importance(xai.SHAP_VALUES, xai.SHAP_FEATURES)
fig_by_value = charts.shap_by_value(xai.SHAP_VALUES, xai.SHAP_DATA, feature_sel, xai.SHAP_FEATURES)

_predicted = xai.SAMPLE_SCORES >= xai.DEFAULT_THRESHOLD
_example_masks = {
    "TP — найденный инцидент": (xai.SAMPLE_Y == 1) & _predicted,
    "FN — пропущенный инцидент": (xai.SAMPLE_Y == 1) & ~_predicted,
    "FP — ложная тревога": (xai.SAMPLE_Y == 0) & _predicted,
    "TN — верное понижение приоритета": (xai.SAMPLE_Y == 0) & ~_predicted,
}
example_lov = [name for name, mask in _example_masks.items() if mask.any()]
example_sel = example_lov[0] if example_lov else ""
fig_local, local_info = xai.local_example(example_sel) if example_sel else (None, "Нет примеров в SHAP-выборке.")

pdp_feature = next((name for name in ("Час", "Критичность", "Тип инцидента") if name in feature_lov), feature_lov[0])
fig_pdp = go.Figure()
fig_pdp.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    annotations=[dict(text="Нажми «Построить PDP/ICE»", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
    height=420,
)
pdp_busy = False
pdp_status = "Выбери признак и запусти расчёт PDP/ICE."


def on_feature(state: State, var_name: str, value):
    state.fig_by_value = charts.shap_by_value(xai.SHAP_VALUES, xai.SHAP_DATA, value, xai.SHAP_FEATURES)


def on_example(state: State, var_name: str, value):
    state.fig_local, state.local_info = xai.local_example(value)


def on_pdp(state: State):
    if state.pdp_busy:
        return
    state.pdp_busy = True
    state.pdp_status = f"Считаю PDP/ICE для признака «{state.pdp_feature}» в фоновом потоке."
    invoke_long_callback(state, xai.compute_pdp, [state.pdp_feature], on_pdp_done)


def on_pdp_done(state: State, status, figure=None):
    state.pdp_busy = False
    if status is True:
        state.fig_pdp = figure
        state.pdp_status = "Расчёт завершён. Серые линии показывают отдельные карточки, синяя — среднюю оценку."
    else:
        state.pdp_status = "Не удалось построить PDP/ICE; подробности доступны в журнале приложения."
        notify(state, "error", "Ошибка расчёта PDP/ICE.")


page = Markdown("""
# Дополнительная диагностика модели

Это необязательная диагностика экспериментальной модели класса **«Решён»** на случайной выборке из 1 000 карточек позднего временного периода. Она показывает, на какие признаки модель опирается, но не доказывает причины инцидента.

<|layout|columns=1|gap=16px|
<|part|class_name=card|
### Общая важность признаков
Показывает, какие признаки сильнее всего меняют оценку модели в среднем. Длина полосы — сила вклада, но не его направление.

<|chart|figure={fig_importance}|>
|>
<|part|class_name=card|
### Вклад частых значений

Показывает средний вклад для карточек с выбранным значением. Красный сдвигает модель к «Решён», синий — от него. Наведи курсор на полосу, чтобы увидеть точное значение и число карточек.

<|{feature_sel}|selector|lov={feature_lov}|dropdown|label=Признак|on_change=on_feature|class_name=fullwidth|>

<|chart|figure={fig_by_value}|>
|>
|>

## Частичная зависимость (PDP) и ICE

<|layout|columns=320px 1|gap=16px|
<|part|class_name=card filters|
<|{pdp_feature}|selector|lov={feature_lov}|dropdown|label=Признак|class_name=fullwidth|>

<|Построить PDP/ICE|button|on_action=on_pdp|active={not pdp_busy}|>

<|{pdp_status}|text|class_name=subtitle|>
|>
<|part|class_name=card|
<|chart|figure={fig_pdp}|>
|>
|>

## Разбор конкретного исхода

Выбери тип результата модели. TP/FN относятся к карточкам `y=1` («Решён»), FP/TN — к карточкам `y=0` (`False Positive`); здесь эти сокращения обозначают исход классификации, а не исходный вердикт.

<|layout|columns=320px 1|gap=16px|
<|part|class_name=card filters|
<|{example_sel}|selector|lov={example_lov}|label=Учебная карточка|on_change=on_example|class_name=fullwidth|>

<|{local_info}|text|mode=markdown|>
|>
<|part|class_name=card|
<|chart|figure={fig_local}|>
|>
|>

Ось локального графика показывает логарифм шансов: вклады складываются с базовым значением модели. SHAP объясняет расчёт модели для этой карточки, но не доказывает, почему инцидент произошёл.
""")
