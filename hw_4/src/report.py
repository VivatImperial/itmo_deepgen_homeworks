from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


def build_labeled_grid(rows: list[list[tuple[str, str | Path]]], output: str | Path, *, cell_size: int = 256) -> None:
    if not rows or not any(rows):
        raise ValueError("Grid requires at least one image")
    label_height = 34
    padding = 8
    columns = max(len(row) for row in rows)
    canvas = Image.new(
        "RGB",
        (
            columns * cell_size + (columns - 1) * padding,
            len(rows) * (cell_size + label_height + padding),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        y = row_index * (cell_size + label_height + padding)
        for column_index, (label, image_path) in enumerate(row):
            x = column_index * (cell_size + padding)
            draw.text((x + 4, y + 6), label[:40], fill=(0, 0, 0))
            image = Image.open(image_path).convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y + label_height))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def write_metrics_summary(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
