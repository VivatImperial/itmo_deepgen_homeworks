from pathlib import Path

import pytest
import torch

from hw_3.src.config import LayerRange
from hw_3.src.latents import load_w_plus, save_w_plus, validate_w_plus
from hw_3.src.transfers import expression_transfer, mix_layers, style_transfer


def test_validate_w_plus_accepts_three_dimensional_tensor() -> None:
    latent = torch.zeros(1, 18, 512)

    result = validate_w_plus(latent, label="latent")

    assert result is latent


def test_validate_w_plus_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Expected latent"):
        validate_w_plus(torch.zeros(18, 512), label="latent")


def test_save_and_load_w_plus_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "latent.pt"
    latent = torch.randn(1, 18, 512)

    save_w_plus(latent, path)

    assert torch.equal(load_w_plus(path), latent)


def test_load_w_plus_accepts_dictionary_payload(tmp_path: Path) -> None:
    path = tmp_path / "latent.pt"
    latent = torch.randn(1, 18, 512)
    torch.save({"w_plus": latent}, path)

    assert torch.equal(load_w_plus(path), latent)


def test_mix_layers_replaces_only_selected_layers() -> None:
    source = torch.ones(1, 6, 2)
    target = torch.zeros(1, 6, 2)

    mixed = mix_layers(source=source, target=target, layers=[1, 3, 5])

    assert torch.equal(mixed[:, 0], target[:, 0])
    assert torch.equal(mixed[:, 1], source[:, 1])
    assert torch.equal(mixed[:, 3], source[:, 3])
    assert torch.equal(mixed[:, 5], source[:, 5])


def test_style_and_expression_transfer_use_given_ranges() -> None:
    source = torch.full((1, 8, 2), 3.0)
    target = torch.zeros(1, 8, 2)

    styled = style_transfer(source, target, LayerRange(4, 7))
    expressed = expression_transfer(source, target, LayerRange(1, 2))

    assert torch.equal(styled[:, 4:8], source[:, 4:8])
    assert torch.equal(styled[:, :4], target[:, :4])
    assert torch.equal(expressed[:, 1:3], source[:, 1:3])
    assert torch.equal(expressed[:, 3:], target[:, 3:])
