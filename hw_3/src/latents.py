from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hw_3.src.config import ensure_file, ensure_parent_dir


def validate_w_plus(latent: torch.Tensor, *, label: str) -> torch.Tensor:
    if latent.ndim != 3:
        raise ValueError(
            f"Expected {label} to have shape [batch, layers, channels], got {tuple(latent.shape)}"
        )
    if latent.shape[1] < 1 or latent.shape[2] < 1:
        raise ValueError(f"Expected {label} to contain at least one layer and channel")
    return latent


def _extract_w_plus(payload: Any, path: Path) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("w_plus"), torch.Tensor):
        return payload["w_plus"]
    raise ValueError(f"Expected tensor or dict with 'w_plus' tensor in {path}")


def load_w_plus(path: str | Path, *, map_location: str | torch.device = "cpu") -> torch.Tensor:
    latent_path = ensure_file(path, "W+ latent")
    payload = torch.load(latent_path, map_location=map_location)
    latent = _extract_w_plus(payload, latent_path)
    return validate_w_plus(latent, label=str(latent_path))


def save_w_plus(latent: torch.Tensor, path: str | Path) -> Path:
    output_path = ensure_parent_dir(path)
    torch.save({"w_plus": validate_w_plus(latent, label="W+ latent").detach().cpu()}, output_path)
    return output_path
