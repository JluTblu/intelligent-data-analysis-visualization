"""Оставляет две целевые категории и добавляет числовую метку y."""
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_DIR / "data" / "raw" / "iavd_ml_dataset.parquet"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
PARQUET_OUTPUT = OUTPUT_DIR / "iavd_binary_labeled.parquet"
TARGET = "Вердикт"
LABELS = {"Решён": 1, "False Positive": 0}


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Не найден файл датасета: {SOURCE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(SOURCE)
    PARQUET_OUTPUT.unlink(missing_ok=True)
    output_schema = parquet_file.schema_arrow.append(pa.field("y", pa.int8()))
    output_schema = output_schema.remove_metadata()
    writer = None
    counts: Counter[str] = Counter()

    try:
        for group_number in range(parquet_file.num_row_groups):
            frame = parquet_file.read_row_group(group_number).to_pandas()
            counts.update(frame[TARGET].fillna("<пусто>").astype(str).value_counts().to_dict())
            selected = frame[frame[TARGET].isin(LABELS)].copy()
            selected["y"] = selected[TARGET].map(LABELS).astype("int8")
            table = pa.Table.from_pandas(selected, preserve_index=False)
            table = table.cast(output_schema, safe=False).replace_schema_metadata(None)

            if writer is None:
                writer = pq.ParquetWriter(PARQUET_OUTPUT, table.schema, compression="zstd")
            writer.write_table(table)

    finally:
        if writer is not None:
            writer.close()

    print(f"Строк исходного набора: {sum(counts.values()):,}".replace(",", " "))
    selected_count = sum(counts[label] for label in LABELS)
    excluded_count = sum(counts.values()) - selected_count
    print(f"Строк в бинарной задаче: {selected_count:,}".replace(",", " "))
    for verdict, y in LABELS.items():
        print(f"y={y} ({verdict}): {counts[verdict]:,}".replace(",", " "))
    print(f"Исключено других вердиктов и пустых меток: {excluded_count:,}".replace(",", " "))
    print(f"Полный промежуточный набор: {PARQUET_OUTPUT}")
    print("Важно: поля категории C пока сохранены; это набор для проверки, не признаки для обучения.")

if __name__ == "__main__":
    main()
