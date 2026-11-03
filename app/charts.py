"""Графики обзора карточек для первой версии Taipy-приложения."""
import textwrap

import pandas as pd
import plotly.graph_objects as go

PALETTE = {
    "Инцидент (Решён)": "#C24E4E",
    "False Positive": "#547A9E",
    "Другой вердикт": "#A7AFB8",
    "Не размечено": "#D6DADE",
}


def monthly_classes(cards: pd.DataFrame) -> go.Figure:
    grouped = cards.groupby(["Месяц", "Целевой класс"], observed=False).size().unstack(fill_value=0)
    months = pd.to_datetime(grouped.index.astype(str), format="%Y-%m", errors="coerce").strftime("%m.%Y")
    fig = go.Figure()
    for label in ["Инцидент (Решён)", "False Positive", "Другой вердикт", "Не размечено"]:
        if label in grouped.columns:
            fig.add_bar(
                x=months,
                y=grouped[label],
                name=label,
                marker_color=PALETTE[label],
                hovertemplate="%{x}<br>%{y:,} карточек<extra>" + label + "</extra>",
            )
    fig.update_layout(
        barmode="stack",
        height=330,
        margin=dict(l=48, r=12, t=24, b=48),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis_title="Месяц создания (UTC)",
        yaxis_title="Карточки",
        template="plotly_white",
    )
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=months, tickangle=-30)
    return fig


def verdict_counts(cards: pd.DataFrame) -> go.Figure:
    values = cards["Вердикт"].fillna("Не размечен").value_counts().head(10).sort_values()
    labels = [str(x) for x in values.index]
    ticktext = ["<br>".join(textwrap.wrap(label, width=26)) for label in labels]
    colors = [PALETTE.get(cards.loc[cards["Вердикт"].fillna("Не размечен") == x, "Целевой класс"].iloc[0], "#8794A1")
              if not cards.loc[cards["Вердикт"].fillna("Не размечен") == x].empty else "#8794A1"
              for x in labels]
    fig = go.Figure(go.Bar(x=values.values, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(
        height=max(360, 42 * len(labels) + 80),
        margin=dict(l=220, r=18, t=24, b=42),
        xaxis_title="Карточки",
        yaxis_title="",
        template="plotly_white",
    )
    fig.update_yaxes(tickmode="array", tickvals=labels, ticktext=ticktext, automargin=True, tickfont=dict(size=12))
    return fig


def criticality_counts(cards: pd.DataFrame) -> go.Figure:
    values = cards["Критичность"].fillna("Не указана").value_counts().sort_values()
    fig = go.Figure(go.Bar(x=values.values, y=values.index.astype(str), orientation="h", marker_color="#8794A1"))
    fig.update_layout(
        height=300,
        margin=dict(l=12, r=12, t=24, b=42),
        xaxis_title="Карточки",
        yaxis_title="",
        template="plotly_white",
    )
    return fig


def shap_importance(values, features, top: int = 15) -> go.Figure:
    importance = pd.Series(abs(values).mean(axis=0), index=features).nlargest(top).sort_values()
    labels = importance.index.tolist()
    ticktext = ["<br>".join(textwrap.wrap(label, width=22)) for label in labels]
    fig = go.Figure(go.Bar(
        x=importance.values,
        y=labels,
        orientation="h",
        marker_color="#8794A1",
        hovertemplate="%{y}: средний |SHAP| = %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Глобальная важность: средний |SHAP| по {len(values):,} карточкам".replace(",", " "),
        height=max(480, 42 * len(importance) + 100),
        margin=dict(l=190, r=24, t=48, b=42),
        xaxis_title="Средний модуль вклада",
        yaxis_title="",
        template="plotly_white",
    )
    fig.update_yaxes(tickmode="array", tickvals=labels, ticktext=ticktext, automargin=True, tickfont=dict(size=12))
    return fig


def shap_by_value(values, data, feature: str, features, top: int = 12, min_count: int = 10) -> go.Figure:
    index = list(features).index(feature)
    summary = pd.DataFrame({"value": data[:, index], "shap": values[:, index]})
    grouped = summary.groupby("value")["shap"].agg(["mean", "size"])
    grouped = grouped[grouped["size"] >= min_count].nlargest(top, "size").sort_values("mean")
    colors = ["#C24E4E" if value > 0 else "#547A9E" for value in grouped["mean"]]
    labels = [str(value)[:70] for value in grouped.index]
    ticktext = ["<br>".join(textwrap.wrap(label, width=26)) for label in labels]
    fig = go.Figure(go.Bar(
        x=grouped["mean"],
        y=labels,
        orientation="h",
        marker_color=colors,
        customdata=[[count, value] for count, value in zip(grouped["size"], grouped.index)],
        hovertemplate="%{customdata[1]}<br>Средний вклад: %{x:+.3f}<br>Карточек: %{customdata[0]}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Вклад частых значений «{feature}»",
        height=max(420, 44 * len(grouped) + 110),
        margin=dict(l=220, r=24, t=48, b=42),
        xaxis_title="Средний SHAP-вклад в оценку класса «Решён»",
        yaxis_title="",
        template="plotly_white",
    )
    fig.update_yaxes(tickmode="array", tickvals=labels, ticktext=ticktext, automargin=True, tickfont=dict(size=12))
    fig.update_xaxes(zeroline=True, zerolinecolor="#9AA0A6")
    return fig


def shap_waterfall(values, features, base: float, row_values, top: int = 12) -> go.Figure:
    contributions = pd.DataFrame({"feature": features, "value": row_values, "shap": values})
    contributions["abs"] = contributions["shap"].abs()
    contributions = contributions.sort_values("abs", ascending=False)
    head, rest = contributions.head(top), contributions.iloc[top:]
    labels = [f"{row.feature} = {str(row.value)[:32]}" for row in head.itertuples()]
    deltas = head["shap"].tolist()
    if len(rest):
        labels.append(f"Остальные признаки ({len(rest)})")
        deltas.append(float(rest["shap"].sum()))
    labels, deltas = labels[::-1], deltas[::-1]
    final = base + sum(deltas)
    fig = go.Figure(go.Waterfall(
        orientation="h",
        y=["Базовая оценка"] + labels + ["Оценка карточки"],
        x=[base] + deltas + [0],
        measure=["absolute"] + ["relative"] * len(deltas) + ["total"],
        increasing=dict(marker=dict(color="#C24E4E")),
        decreasing=dict(marker=dict(color="#547A9E")),
        totals=dict(marker=dict(color="#334454")),
        text=[f"{base:+.2f}"] + [f"{value:+.2f}" for value in deltas] + [f"{final:+.2f}"],
        textposition="outside",
        connector=dict(line=dict(color="#DADCE0", width=1)),
    ))
    fig.update_layout(
        title="Вклад признаков в логарифм шансов класса «Решён»",
        height=120 + 30 * (len(deltas) + 2),
        margin=dict(l=20, r=55, t=48, b=50),
        xaxis_title="Вклад SHAP (лог-шансы)",
        template="plotly_white",
    )
    return fig


def pdp_ice(feature: str, grid: list, pdp, ice, observed: pd.Series, threshold: float, numeric: bool) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04)
    x = grid if numeric else [str(value)[:26] for value in grid]
    for line in ice[: min(60, len(ice))]:
        fig.add_scatter(x=x, y=line, mode="lines", line=dict(color="rgba(154,160,166,0.25)", width=1), hoverinfo="skip", row=1, col=1)
    fig.add_scatter(
        x=x,
        y=pdp,
        mode="lines+markers" if numeric else "markers",
        line=dict(color="#547A9E", width=3),
        marker=dict(color="#547A9E", size=7),
        hovertemplate="%{x}: средняя оценка %{y:.1%}<extra></extra>",
        row=1,
        col=1,
    )
    fig.add_hline(y=threshold, line=dict(color="#C24E4E", width=1, dash="dot"), row=1, col=1)
    if numeric:
        observed_numeric = pd.to_numeric(observed, errors="coerce").dropna()
        inside = observed_numeric[observed_numeric.between(min(grid), max(grid))]
        fig.add_histogram(x=inside, nbinsx=30, marker_color="#DADCE0", row=2, col=1)
    else:
        counts = observed.astype(str).value_counts().reindex(grid, fill_value=0)
        fig.add_bar(x=x, y=counts.values, marker_color="#DADCE0", row=2, col=1)
    fig.update_layout(
        title=f"Частичная зависимость от «{feature}» (PDP и ICE)",
        height=460,
        margin=dict(l=52, r=20, t=48, b=60),
        template="plotly_white",
        showlegend=False,
    )
    fig.update_yaxes(title="Оценка класса «Решён»", tickformat=".0%", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title="Карточки", row=2, col=1)
    return fig
