from __future__ import annotations

from collections.abc import Iterable

import torch

from hw_3.src.config import EXPRESSION_LAYER_RANGE, STYLE_LAYER_RANGE, LayerRange
from hw_3.src.latents import validate_w_plus


def mix_layers(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    layers: Iterable[int],
) -> torch.Tensor:
    source = validate_w_plus(source, label="source")
    target = validate_w_plus(target, label="target")
    if source.shape != target.shape:
        raise ValueError(f"source and target shapes must match, got {source.shape} and {target.shape}")

    mixed = target.clone()
    layer_count = target.shape[1]
    for layer in layers:
        if layer < 0 or layer >= layer_count:
            raise IndexError(f"Layer index {layer} is outside W+ layer count {layer_count}")
        mixed[:, layer, :] = source[:, layer, :]
    return mixed


def style_transfer(
    style_source: torch.Tensor,
    identity_target: torch.Tensor,
    layer_range: LayerRange = STYLE_LAYER_RANGE,
) -> torch.Tensor:
    return mix_layers(
        source=style_source,
        target=identity_target,
        layers=layer_range.indices(),
    )


def expression_transfer(
    expression_source: torch.Tensor,
    identity_target: torch.Tensor,
    layer_range: LayerRange = EXPRESSION_LAYER_RANGE,
) -> torch.Tensor:
    return mix_layers(
        source=expression_source,
        target=identity_target,
        layers=layer_range.indices(),
    )
