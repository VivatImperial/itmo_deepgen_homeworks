from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HW4Config:
    base_model: str
    controlnet_model: str
    subject_token: str
    class_name: str
    instance_prompt: str
    class_prompt: str
    resolution: int
    seed: int
    lora_ranks: list[int]
    comparison_prompts: list[str]


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field must be a non-empty string: {key}")
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Config field must be an integer: {key}")
    return value


def _require_int_list(payload: dict[str, Any], key: str) -> list[int]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, int) for item in value):
        raise ValueError(f"Config field must be a non-empty integer list: {key}")
    return value


def _require_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"Config field must be a non-empty string list: {key}")
    return value


def load_config(path: str | Path) -> HW4Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")

    return HW4Config(
        base_model=_require_string(payload, "base_model"),
        controlnet_model=_require_string(payload, "controlnet_model"),
        subject_token=_require_string(payload, "subject_token"),
        class_name=_require_string(payload, "class_name"),
        instance_prompt=_require_string(payload, "instance_prompt"),
        class_prompt=_require_string(payload, "class_prompt"),
        resolution=_require_int(payload, "resolution"),
        seed=_require_int(payload, "seed"),
        lora_ranks=_require_int_list(payload, "lora_ranks"),
        comparison_prompts=_require_string_list(payload, "comparison_prompts"),
    )
