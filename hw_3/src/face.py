from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hw_3.src.arcface_iresnet import iresnet100
from hw_3.src.config import ensure_file


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "backbone"):
            value = payload.get(key)
            if isinstance(value, dict):
                return _strip_module_prefix(value)
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return _strip_module_prefix(payload)
    return None


class ArcFaceIdentity:
    def __init__(self, model: Any, *, device: torch.device) -> None:
        self.model = model
        self.device = device

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "ArcFaceIdentity":
        path = ensure_file(checkpoint_path, "ArcFace checkpoint")
        target_device = torch.device(device)
        model = torch.load(path, map_location=target_device)
        if isinstance(model, dict) and isinstance(model.get("model"), torch.nn.Module):
            model = model["model"]
        else:
            state_dict = _extract_state_dict(model)
            if state_dict is not None:
                model = iresnet100()
                model.load_state_dict(state_dict, strict=False)
        if hasattr(model, "eval"):
            model = model.eval()
        if hasattr(model, "to"):
            model = model.to(target_device)
        if hasattr(model, "parameters"):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        return cls(model, device=target_device)

    def embed(self, image_batch: torch.Tensor, *, detach: bool = False) -> torch.Tensor:
        if detach:
            with torch.no_grad():
                embeddings = self.model(image_batch.to(self.device))
        else:
            embeddings = self.model(image_batch.to(self.device))
        return torch.nn.functional.normalize(embeddings, dim=1)

    def identity_loss(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        generated_embedding = self.embed(generated)
        reference_embedding = self.embed(reference, detach=True)
        cosine = torch.sum(generated_embedding * reference_embedding, dim=1)
        return 1.0 - cosine.mean()
