import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from hw_3.src.face import ArcFaceIdentity
from hw_3.src.stylegan import ProjectionConfig, StyleGANGenerator, image_to_tensor


class _TinyIdentity(torch.nn.Module):
    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        return image_batch.flatten(start_dim=1)


class _TinyMapping:
    def __init__(self) -> None:
        self.w_avg = torch.arange(4, dtype=torch.float32)


class _TinyGenerator(torch.nn.Module):
    num_ws = 3

    def __init__(self) -> None:
        super().__init__()
        self.mapping = _TinyMapping()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        return torch.zeros(batch, 3, 4, 4)


class _FakeOpenUrl:
    def __enter__(self):
        return object()

    def __exit__(self, *args: object) -> None:
        return None


class _TinyVgg(torch.nn.Module):
    def forward(
        self,
        image_batch: torch.Tensor,
        *,
        resize_images: bool,
        return_lpips: bool,
    ) -> torch.Tensor:
        return image_batch[:, :, 0, 0]


class _LayerAwareMapping(torch.nn.Module):
    num_ws = 3

    def __init__(self) -> None:
        super().__init__()
        self.w_avg = torch.zeros(1)

    def forward(self, z: torch.Tensor, _: object) -> torch.Tensor:
        return torch.zeros(z.shape[0], self.num_ws, 1, device=z.device)


class _LayerAwareSynthesis(torch.nn.Module):
    def forward(self, ws: torch.Tensor, *, noise_mode: str) -> torch.Tensor:
        channels = torch.stack(
            [ws[:, 0, 0], -2 * ws[:, 1, 0], 3 * ws[:, 2, 0]],
            dim=1,
        )
        return channels[:, :, None, None].expand(-1, -1, 4, 4)


class _LayerAwareGenerator(torch.nn.Module):
    z_dim = 1
    img_resolution = 4
    num_ws = 3

    def __init__(self) -> None:
        super().__init__()
        self.mapping = _LayerAwareMapping()
        self.synthesis = _LayerAwareSynthesis()


def test_stylegan_load_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="StyleGAN checkpoint"):
        StyleGANGenerator.load(tmp_path / "stylegan2.pkl")


def test_arcface_load_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ArcFace checkpoint"):
        ArcFaceIdentity.load(tmp_path / "arcface.pth")


def test_projection_config_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        ProjectionConfig(steps=0)


def test_image_to_tensor_returns_normalized_batch() -> None:
    image = Image.new("RGB", (8, 8), color=(255, 127, 0))

    tensor = image_to_tensor(image, image_size=4)

    assert tensor.shape == (1, 3, 4, 4)
    assert torch.all(tensor <= 1.0)
    assert torch.all(tensor >= -1.0)


def test_arcface_identity_loss_backpropagates_to_generated_image() -> None:
    identity = ArcFaceIdentity(_TinyIdentity(), device=torch.device("cpu"))
    generated = torch.randn(1, 3, 4, 4, requires_grad=True)
    reference = torch.randn(1, 3, 4, 4)

    loss = identity.identity_loss(generated, reference)
    loss.backward()

    assert generated.grad is not None
    assert torch.any(generated.grad != 0)


def test_arcface_load_accepts_state_dict_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "arcface_state.pt"
    state_dict = {
        "module.conv1.weight": torch.randn(64, 3, 3, 3),
    }
    torch.save({"state_dict": state_dict}, checkpoint)

    identity = ArcFaceIdentity.load(checkpoint)

    assert identity.model.training is False


def test_stylegan_initial_w_plus_uses_generator_w_avg() -> None:
    generator = StyleGANGenerator(_TinyGenerator(), device=torch.device("cpu"))

    latent = generator.initial_w_plus(batch_size=2)

    assert latent.shape == (2, 3, 4)
    assert torch.equal(latent[0, 0], torch.arange(4, dtype=torch.float32))


def test_official_projector_requires_vendor_module(monkeypatch) -> None:
    generator = StyleGANGenerator(_TinyGenerator(), device=torch.device("cpu"))
    image = Image.new("RGB", (8, 8), color="white")
    original_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "dnnlib":
            raise ImportError(name)
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError, match="stylegan2-ada-pytorch"):
        generator.project_image_official(image, config=ProjectionConfig(steps=1))


def test_official_projector_optimizes_w_plus_layers(monkeypatch) -> None:
    generator = StyleGANGenerator(_LayerAwareGenerator(), device=torch.device("cpu"))
    image = Image.new("RGB", (4, 4), color="black")
    original_import_module = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *args, **kwargs: SimpleNamespace(
            util=SimpleNamespace(open_url=lambda _: _FakeOpenUrl())
        )
        if name == "dnnlib"
        else original_import_module(name, *args, **kwargs),
    )
    monkeypatch.setattr(torch.jit, "load", lambda _: _TinyVgg())

    latent = generator.project_image_official(
        image,
        config=ProjectionConfig(steps=2, learning_rate=0.01, w_avg_samples=2),
    )

    assert latent.shape == (1, 3, 1)
    assert not torch.allclose(latent[:, 0, :], latent[:, 1, :])
