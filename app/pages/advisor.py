"""Страница модели-советчика: порог по стоимости/Fβ и асинхронный инференс."""
import pandas as pd
from taipy.gui import Markdown, State, get_state_id, invoke_long_callback, notify

import advisor

k_lov = ["50", "100"]
k_sel = "50"
strategy_lov = ["Минимум стоимости", "Максимум Fβ"]
strategy_sel = strategy_lov[0]
_default_eval = advisor.evaluation_for(k_sel, "cost")
threshold = _default_eval["threshold"]
threshold_label = f"{threshold:.3f}".replace(".", ",")

running = False
has_result = False
progress = 0
run_info = ""
k_queue = f"{advisor.count_unlabeled():,}".replace(",", " ")
k_flagged = "—"
review_share = "—"
queue_table = pd.DataFrame(columns=[
    "EntityId",
    "Создан (UTC)",
    "Критичность",
    "Категория IRP",
    "Тип инцидента",
    "Источник SIEM",
    "Правило",
    "Оценка класса «Решён»",
])

test_precision = f"{_default_eval['precision']:.1%}".replace(".", ",")
test_recall = f"{_default_eval['recall']:.1%}".replace(".", ",")
test_cost = f"{_default_eval['cost']:,}".replace(",", " ")
test_cost_all = f"{_default_eval['all_review_cost']:,}".replace(",", " ")
test_fp = f"{_default_eval['fp']:,}".replace(",", " ")
test_fn = f"{_default_eval['fn']:,}".replace(",", " ")


def _strategy_code(value: str) -> str:
    return "fbeta" if value == "Максимум Fβ" else "cost"


def _refresh_queue(state: State, show_queue: bool = False):
    sid = get_state_id(state)
    table, summary = advisor.queue_for_threshold(sid, float(state.threshold))
    with state as s:
        s.queue_table = table
        s.k_queue = f"{summary['total']:,}".replace(",", " ") if summary["total"] else "0"
        s.k_flagged = f"{summary['flagged']:,}".replace(",", " ") if summary["total"] else "0"
        s.review_share = f"{summary['share']:.1%}".replace(".", ",") if summary["total"] else "0,0%"
        s.threshold_label = f"{float(state.threshold):.3f}".replace(".", ",")
        if show_queue:
            s.has_result = True


def _update_choice(state: State):
    code = _strategy_code(state.strategy_sel)
    result = advisor.evaluation_for(state.k_sel, code)
    threshold_value = result["threshold"]
    with state as s:
        s.threshold = threshold_value
        s.threshold_label = f"{threshold_value:.3f}".replace(".", ",")
        s.test_precision = f"{result['precision']:.1%}".replace(".", ",")
        s.test_recall = f"{result['recall']:.1%}".replace(".", ",")
        s.test_cost = f"{result['cost']:,}".replace(",", " ")
        s.test_cost_all = f"{result['all_review_cost']:,}".replace(",", " ")
        s.test_fp = f"{result['fp']:,}".replace(",", " ")
        s.test_fn = f"{result['fn']:,}".replace(",", " ")
    if state.has_result:
        _refresh_queue(state)


def on_k_change(state: State, var_name: str = "", value=None):
    _update_choice(state)


def on_strategy_change(state: State, var_name: str = "", value=None):
    _update_choice(state)


def on_threshold_change(state: State, var_name: str = "", value=None):
    if state.has_result:
        _refresh_queue(state)
    else:
        state.threshold_label = f"{float(state.threshold):.3f}".replace(".", ",")


def start_inference(state: State):
    if state.running:
        return
    sid = get_state_id(state)
    with state as s:
        s.running = True
        s.progress = 0
        s.run_info = "Оцениваю карточки без вердикта. Расчёт может занять несколько секунд; очередь появится автоматически."
    invoke_long_callback(
        state,
        advisor.score_unlabeled,
        [sid],
        on_inference_status,
        [],
        1000,
    )


def on_page_load(state: State, page_name: str):
    if page_name.strip("/") == "advisor" and not state.has_result and not state.running:
        start_inference(state)


def on_inference_status(state: State, status, result=None):
    if isinstance(status, bool):
        state.running = False
        if status:
            state.progress = 100
            _refresh_queue(state, show_queue=True)
            state.run_info = (
                f"Оценено карточек: {result['rows']:,} за "
                f"{result['seconds']:.2f}".replace(".", ",") + " с."
            ).replace(",", " ", 1)
            notify(state, "success", "Очередь рассчитана.")
        else:
            state.progress = 0
            state.run_info = "Не удалось рассчитать очередь."
            notify(state, "error", "Расчёт не выполнен.")
    else:
        state.progress = min(90, int(status) * 10)


page = Markdown("""
# Модель-советчик

Учебная модель ранжирует карточки без вердикта для ручного просмотра.

<|layout|columns=320px 1|gap=16px|
<|part|class_name=card filters|
### Настройки

<|{k_sel}|selector|lov={k_lov}|label=Цена пропуска (k)|dropdown|on_change=on_k_change|class_name=fullwidth|>

<|{strategy_sel}|selector|lov={strategy_lov}|label=Правило порога|dropdown|on_change=on_strategy_change|class_name=fullwidth|>

Чем выше k, тем дороже пропуск метки «Решён».

<|{threshold}|slider|min=0.01|max=0.99|step=0.01|continuous=False|on_change=on_threshold_change|>

Порог τ = <|{threshold_label}|text|>. Ниже порог — больше карточек в очереди.

<|{progress}|progress|linear|show_value|>

<|{run_info}|text|class_name=subtitle|>
|>

<|
<|layout|columns=1 1 1|gap=12px|
<|part|class_name=card|
Без вердикта
<|{k_queue}|text|class_name=kpi|>
|>
<|part|class_name=card|
Выше порога
<|{k_flagged}|text|class_name=kpi|>
|>
<|part|class_name=card|
Доля очереди
<|{review_share}|text|class_name=kpi|>
|>
|>

<|part|render={has_result}|class_name=card|
### Очередь на просмотр

До 500 карточек выше порога; самые высокие оценки — первыми.

<|{queue_table}|table|page_size=20|>
|>

<|part|class_name=card|
### Проверка на размеченных карточках

Поздний тест. Ползунок меняет очередь, но не эти метрики.

<|layout|columns=1 1 1|gap=12px|
<|part|class_name=metric|
Доля верных в очереди
<|{test_precision}|text|>
|>
<|part|class_name=metric|
Найдено от «Решён»
<|{test_recall}|text|>
|>
<|part|class_name=metric|
Cost модели
<|{test_cost}|text|>
|>
|>

<|layout|columns=1 1 1|gap=12px|
<|part|class_name=metric|
Cost, проверить все
<|{test_cost_all}|text|>
|>
<|part|class_name=metric|
Ложные тревоги (FP)
<|{test_fp}|text|>
|>
<|part|class_name=metric|
Пропуски «Решён» (FN)
<|{test_fn}|text|>
|>
|>
|>

|>
|>
""")
