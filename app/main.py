"""Запуск Taipy-приложения обзора SOC и результатов ML-модели."""
from taipy.gui import Gui

from pages import advisor, overview, xai

root = """
<|navbar|lov={nav}|>
"""
nav = [("/overview", "Обзор потока"), ("/advisor", "Модель-советчик"), ("/xai", "Диагностика модели")]

pages = {
    "/": root,
    "overview": overview.page,
    "advisor": advisor.page,
    "xai": xai.page,
}


def on_page_load(state, page_name):
    advisor.on_page_load(state, page_name)


if __name__ == "__main__":
    Gui(pages=pages, css_file="style.css").run(
        title="SOC · анализ карточек", dark_mode=False, port=5050, use_reloader=False
    )
