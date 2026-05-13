from __future__ import annotations

import pickle
import importlib
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from hw_3.src.config import ensure_file
from hw_3.src.images import center_crop_square, resize_image
from hw_3.src.latents import validate_w_plus


@dataclass(frozen=True)
class ProjectionConfig:
    steps: int = 500
    learning_rate: float = 0.05
    image_size: int = 1024
    latent_layers: int | None = None
    latent_channels: int | None = None
    mse_weight: float = 1.0
    latent_l2_weight: float = 0.001
    w_avg_samples: int = 10000
    initial_noise_factor: float = 0.05
    regularize_noise_weight: float = 1e5

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        if self.latent_layers is not None and self.latent_layers < 1:
            raise ValueError("latent_layers must be positive")
        if self.latent_channels is not None and self.latent_channels < 1:
            raise ValueError("latent_channels must be positive")
        if self.w_avg_samples < 1:
            raise ValueError("w_avg_samples must be positive")


def image_to_tensor(image: Image.Image, *, image_size: int) -> torch.Tensor:
    prepared = resize_image(center_crop_square(image.convert("RGB")), image_size)
    array = np.asarray(prepared, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor / 127.5 - 1.0).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"Expected image tensor with shape [1, C, H, W], got {tuple(tensor.shape)}")
    image = tensor.detach().cpu().clamp(-1.0, 1.0)
    image = ((image.squeeze(0).permute(1, 2, 0) + 1.0) * 127.5).byte().numpy()
    return Image.fromarray(image)


class StyleGANGenerator:
    def __init__(self, generator: Any, *, device: torch.device) -> None:
        self.generator = generator
        self.device = device

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "StyleGANGenerator":
        path = ensure_file(checkpoint_path, "StyleGAN checkpoint")
        target_device = torch.device(device)
        try:
            payload = torch.load(path, map_location=target_device, weights_only=False)
        except Exception:
            with path.open("rb") as checkpoint_file:
                payload = pickle.load(checkpoint_file)
        generator = payload.get("G_ema") if isinstance(payload, dict) else payload
        if generator is None:
            raise ValueError(f"StyleGAN checkpoint does not contain a generator: {path}")
        if hasattr(generator, "eval"):
            generator = generator.eval()
        if hasattr(generator, "to"):
            generator = generator.to(target_device)
        return cls(generator, device=target_device)

    def synthesis(self, w_plus: torch.Tensor) -> torch.Tensor:
        latent = validate_w_plus(w_plus, label="W+ latent").to(self.device)
        if hasattr(self.generator, "synthesis"):
            return self.generator.synthesis(latent, noise_mode="const")
        return self.generator(latent)

    def initial_w_plus(
        self,
        *,
        batch_size: int = 1,
        latent_layers: int | None = None,
        latent_channels: int | None = None,
    ) -> torch.Tensor:
        if hasattr(self.generator, "mapping") and hasattr(self.generator.mapping, "w_avg"):
            w_avg = self.generator.mapping.w_avg.detach().to(self.device).float()
        else:
            channels = latent_channels or 512
            w_avg = torch.zeros(channels, device=self.device)

        layers = latent_layers or getattr(self.generator, "num_ws", None) or 18
        return w_avg.reshape(1, 1, -1).repeat(batch_size, layers, 1)

    def project_image(
        self,
        image: Image.Image,
        *,
        config: ProjectionConfig = ProjectionConfig(),
        progress: bool = True,
        description: str = "project",
    ) -> torch.Tensor:
        target = image_to_tensor(image, image_size=config.image_size).to(self.device)
        latent = self.initial_w_plus(
            latent_layers=config.latent_layers,
            latent_channels=config.latent_channels,
        ).clone()
        latent.requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=config.learning_rate)

        step_iterator = tqdm(
            range(config.steps),
            desc=description,
            leave=False,
            disable=not progress,
        )
        for _ in step_iterator:
            optimizer.zero_grad(set_to_none=True)
            generated = self.synthesis(latent)
            if generated.shape[-2:] != target.shape[-2:]:
                generated = torch.nn.functional.interpolate(
                    generated,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            loss = config.mse_weight * torch.nn.functional.mse_loss(generated, target)
            loss = loss + config.latent_l2_weight * latent.square().mean()
            loss.backward()
            optimizer.step()
            step_iterator.set_postfix(loss=f"{loss.item():.4f}")

        return latent.detach().cpu()

    def project_image_official(
        self,
        image: Image.Image,
        *,
        config: ProjectionConfig = ProjectionConfig(),
        verbose: bool = False,
    ) -> torch.Tensor:
        try:
            dnnlib = importlib.import_module("dnnlib")
        except ImportError as exc:
            raise ImportError(
                "NVIDIA projector requires stylegan2-ada-pytorch on PYTHONPATH. "
                "Run: export PYTHONPATH=\"hw_3/vendor/stylegan2-ada-pytorch:${PYTHONPATH}\""
            ) from exc

        generator = copy.deepcopy(self.generator).eval().requires_grad_(False).to(self.device)
        image_size = getattr(generator, "img_resolution", config.image_size)
        target_image = resize_image(center_crop_square(image.convert("RGB")), image_size)
        target_uint8 = np.asarray(target_image, dtype=np.uint8)
        target_tensor = torch.tensor(
            target_uint8.transpose([2, 0, 1]),
            device=self.device,
            dtype=torch.float32,
        )

        z_samples = np.random.RandomState(123).randn(config.w_avg_samples, generator.z_dim).astype(
            np.float32
        )
        w_samples = generator.mapping(torch.from_numpy(z_samples).to(self.device), None)
        w_samples = w_samples[:, :1, :].detach().cpu().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True).astype(np.float32)
        w_std = float((np.sum((w_samples - w_avg) ** 2) / config.w_avg_samples) ** 0.5)

        noise_buffers = {
            name: buffer
            for name, buffer in generator.synthesis.named_buffers()
            if "noise_const" in name
        }
        url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
        with dnnlib.util.open_url(url) as file:
            vgg16 = torch.jit.load(file).eval().to(self.device)

        target_features_image = target_tensor.unsqueeze(0)
        if target_features_image.shape[2] > 256:
            target_features_image = F.interpolate(target_features_image, size=(256, 256), mode="area")
        target_features = vgg16(target_features_image, resize_images=False, return_lpips=True)

        num_ws = getattr(generator.mapping, "num_ws", getattr(generator, "num_ws", 18))
        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=self.device)
        w_opt = w_opt.repeat([1, num_ws, 1]).requires_grad_(True)
        optimizer = torch.optim.Adam(
            [w_opt] + list(noise_buffers.values()),
            betas=(0.9, 0.999),
            lr=config.learning_rate,
        )

        for buffer in noise_buffers.values():
            buffer[:] = torch.randn_like(buffer)
            buffer.requires_grad = True

        step_iterator = tqdm(range(config.steps), desc="official projector", leave=False)
        for step in step_iterator:
            t = step / config.steps
            w_noise_scale = config.initial_noise_factor * w_std * max(0.0, 1.0 - t / 0.75) ** 2
            lr_ramp = min(1.0, (1.0 - t) / 0.25)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / 0.05)
            learning_rate = config.learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = w_opt + w_noise
            synth_images = generator.synthesis(ws, noise_mode="const")
            synth_images = (synth_images + 1) * (255 / 2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode="area")

            synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            distance = (target_features - synth_features).square().sum()

            regularization_loss = torch.zeros([], device=self.device)
            for value in noise_buffers.values():
                noise = value[None, None, :, :]
                while True:
                    regularization_loss = regularization_loss + (
                        noise * torch.roll(noise, shifts=1, dims=3)
                    ).mean().square()
                    regularization_loss = regularization_loss + (
                        noise * torch.roll(noise, shifts=1, dims=2)
                    ).mean().square()
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)

            loss = distance + regularization_loss * config.regularize_noise_weight
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step_iterator.set_postfix(dist=f"{distance.item():.2f}", loss=f"{loss.item():.2f}")
            if verbose and step % 50 == 0:
                print(
                    f"step {step + 1:>4d}/{config.steps}: "
                    f"dist {distance.item():.2f} loss {loss.item():.2f}"
                )

            with torch.no_grad():
                for buffer in noise_buffers.values():
                    buffer -= buffer.mean()
                    buffer *= buffer.square().mean().rsqrt()

        return w_opt.detach().cpu()
