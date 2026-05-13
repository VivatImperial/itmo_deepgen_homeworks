from __future__ import annotations

import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class PreparedImage:
    image_id: str
    image_path: Path
    caption_path: Path
    source: str
    caption: str


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Manifest must contain a non-empty images list")
    return payload


def _fetch_image(url: str, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(url)
    if source_path.exists():
        shutil.copyfile(source_path, cache_path)
        return cache_path
    with urllib.request.urlopen(url, timeout=30) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def _crop_box(entry: dict[str, Any]) -> tuple[int, int, int, int]:
    crop = entry.get("crop_box")
    if not isinstance(crop, list) or len(crop) != 4 or not all(isinstance(value, int) for value in crop):
        raise ValueError(f"Image entry must contain integer crop_box [left, top, right, bottom]: {entry.get('id')}")
    left, top, right, bottom = crop
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop_box geometry for {entry.get('id')}: {crop}")
    return left, top, right, bottom


def prepare_dataset(manifest_path: str | Path, output_dir: str | Path, *, image_size: int) -> list[PreparedImage]:
    payload = _load_manifest(manifest_path)
    output = Path(output_dir)
    raw_dir = output / "raw"
    instance_dir = output / "instance"
    instance_dir.mkdir(parents=True, exist_ok=True)
    records: list[PreparedImage] = []

    for entry in payload["images"]:
        if not isinstance(entry, dict):
            raise ValueError("Every manifest image entry must be an object")
        image_id = str(entry.get("id", "")).strip()
        url = str(entry.get("url", "")).strip()
        caption = str(entry.get("caption", "")).strip()
        source = str(entry.get("source", "")).strip()
        if not image_id or not url or not caption:
            raise ValueError(f"Image entry requires id, url and caption: {entry}")

        raw_path = _fetch_image(url, raw_dir / f"{image_id}.jpg")
        image = Image.open(raw_path).convert("RGB")
        cropped = image.crop(_crop_box(entry))
        resized = cropped.resize((image_size, image_size), Image.Resampling.LANCZOS)
        image_path = instance_dir / f"{image_id}.jpg"
        caption_path = instance_dir / f"{image_id}.txt"
        resized.save(image_path, quality=95)
        caption_path.write_text(caption + "\n", encoding="utf-8")
        records.append(
            PreparedImage(
                image_id=image_id,
                image_path=image_path,
                caption_path=caption_path,
                source=source,
                caption=caption,
            )
        )

    metadata = {
        "subject": payload.get("subject", ""),
        "image_count": len(records),
        "image_size": image_size,
        "images": [
            {
                **asdict(record),
                "image_path": str(record.image_path),
                "caption_path": str(record.caption_path),
            }
            for record in records
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records
