from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hw_1.src.common import probability_from_log, scientific_notation_from_log


class AvatarGenerator:
    def __init__(self, alpha: float = 1.0, seed: int | None = None) -> None:
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.rgba_images: np.ndarray | None = None
        self.rgb_images: np.ndarray | None = None
        self.opaque_masks: np.ndarray | None = None
        self.image_shape: tuple[int, int, int] | None = None
        self.source_files: list[str] = []
        self.channel_priors: np.ndarray | None = None
        self.global_opaque_prior: float | None = None

    @staticmethod
    def _load_avatar(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            rgba_image = image.convert("RGBA")
        return np.asarray(rgba_image, dtype=np.uint8)

    def fit(self, avatar_dir: str | Path) -> None:
        avatar_dir = Path(avatar_dir)
        files = sorted(avatar_dir.glob("*.png"))
        if not files:
            raise ValueError(f"No PNG files found in {avatar_dir}")

        images = [self._load_avatar(path) for path in files]
        first_shape = images[0].shape
        if any(image.shape != first_shape for image in images):
            raise ValueError("All avatars must have the same image shape")

        self.rgba_images = np.stack(images, axis=0)
        self.rgb_images = self.rgba_images[..., :3]
        self.opaque_masks = self.rgba_images[..., 3] > 0
        self.image_shape = (first_shape[0], first_shape[1], 3)
        self.source_files = [path.name for path in files]
        self.channel_priors = self._estimate_channel_priors()
        self.global_opaque_prior = float(self.opaque_masks.mean())

    def _estimate_channel_priors(self) -> np.ndarray:
        if self.rgb_images is None or self.opaque_masks is None:
            raise RuntimeError("Images are not loaded")

        priors = np.zeros((3, 256), dtype=np.float64)
        for channel_index in range(3):
            values = self.rgb_images[..., channel_index][self.opaque_masks]
            histogram = np.bincount(values, minlength=256).astype(np.float64)
            priors[channel_index] = (histogram + self.alpha / 256.0) / (histogram.sum() + self.alpha)
        return priors

    def _mle_and_bayesian_alpha_probabilities(self, opaque_count: int, num_images: int) -> tuple[float, float]:
        if self.global_opaque_prior is None:
            raise RuntimeError("Call fit() before sample()")
        mle_probability = opaque_count / num_images
        bayesian_probability = (opaque_count + self.alpha * self.global_opaque_prior) / (num_images + self.alpha)
        return mle_probability, bayesian_probability

    def _sample_rgb_tuple(
        self,
        observed_rgb: np.ndarray,
    ) -> tuple[np.ndarray, list[int], list[float], list[float]]:
        if self.channel_priors is None:
            raise RuntimeError("Call fit() before sample()")

        channel_probability_tables: list[dict[int, tuple[float, float]]] = []
        for channel_index in range(3):
            observed_values = observed_rgb[:, channel_index]
            unique_values, counts = np.unique(observed_values, return_counts=True)
            mle_probabilities = counts / counts.sum()
            prior_probabilities = self.channel_priors[channel_index, unique_values]
            bayesian_probabilities = (counts + self.alpha * prior_probabilities) / (counts.sum() + self.alpha)
            bayesian_probabilities = bayesian_probabilities / bayesian_probabilities.sum()
            channel_probability_tables.append(
                {
                    int(value): (float(mle_probability), float(bayesian_probability))
                    for value, mle_probability, bayesian_probability in zip(
                        unique_values,
                        mle_probabilities,
                        bayesian_probabilities,
                        strict=True,
                    )
                }
            )

        unique_rgb = np.unique(observed_rgb, axis=0)
        bayesian_weights = np.ones(len(unique_rgb), dtype=np.float64)
        for candidate_index, candidate_rgb in enumerate(unique_rgb):
            for channel_index in range(3):
                bayesian_weights[candidate_index] *= channel_probability_tables[channel_index][int(candidate_rgb[channel_index])][1]

        bayesian_weights = bayesian_weights / bayesian_weights.sum()
        sampled_index = int(self.rng.choice(len(unique_rgb), p=bayesian_weights))
        sampled_rgb = unique_rgb[sampled_index]

        channel_supports: list[int] = []
        mle_per_channel: list[float] = []
        bayesian_per_channel: list[float] = []
        for channel_index in range(3):
            channel_value = int(sampled_rgb[channel_index])
            channel_supports.append(int(np.sum(observed_rgb[:, channel_index] == channel_value)))
            mle_probability, bayesian_probability = channel_probability_tables[channel_index][channel_value]
            mle_per_channel.append(mle_probability)
            bayesian_per_channel.append(bayesian_probability)

        return sampled_rgb, channel_supports, mle_per_channel, bayesian_per_channel

    def sample(self) -> tuple[Image.Image, dict[str, Any]]:
        if (
            self.rgb_images is None
            or self.opaque_masks is None
            or self.image_shape is None
            or self.channel_priors is None
        ):
            raise RuntimeError("Call fit() before sample()")

        num_images, height, width, _ = self.rgb_images.shape
        sampled_image = np.full((height, width, 3), 255, dtype=np.uint8)
        opaque_pixel_count = 0
        mle_log_probability = 0.0
        bayesian_log_probability = 0.0
        average_support_counts: list[float] = []

        for y in range(height):
            for x in range(width):
                opaque_observations = self.opaque_masks[:, y, x]
                opaque_count = int(opaque_observations.sum())
                mle_alpha_probability, bayesian_alpha_probability = self._mle_and_bayesian_alpha_probabilities(
                    opaque_count=opaque_count,
                    num_images=num_images,
                )

                sampled_opaque = self.rng.random() < bayesian_alpha_probability

                if sampled_opaque:
                    opaque_pixel_count += 1
                    bayesian_log_probability += math.log(bayesian_alpha_probability)
                    if opaque_count == 0:
                        mle_log_probability = float("-inf")
                        for channel_index in range(3):
                            global_prior = self.channel_priors[channel_index]
                            sampled_value = int(self.rng.choice(256, p=global_prior))
                            sampled_image[y, x, channel_index] = sampled_value
                            bayesian_log_probability += math.log(float(global_prior[sampled_value]))
                        average_support_counts.append(0.0)
                        continue

                    mle_log_probability += math.log(mle_alpha_probability)

                    observed_rgb = self.rgb_images[opaque_observations, y, x, :]
                    sampled_rgb, channel_supports, mle_per_channel, bayesian_per_channel = self._sample_rgb_tuple(
                        observed_rgb=observed_rgb
                    )
                    sampled_image[y, x, :] = sampled_rgb
                    for mle_probability, bayesian_probability in zip(
                        mle_per_channel,
                        bayesian_per_channel,
                        strict=True,
                    ):
                        mle_log_probability += math.log(mle_probability)
                        bayesian_log_probability += math.log(bayesian_probability)
                    average_support_counts.append(float(np.mean(channel_supports)))
                else:
                    mle_transparent_probability = 1.0 - mle_alpha_probability
                    bayesian_transparent_probability = 1.0 - bayesian_alpha_probability
                    if mle_transparent_probability == 0.0:
                        mle_log_probability = float("-inf")
                    elif not math.isinf(mle_log_probability):
                        mle_log_probability += math.log(mle_transparent_probability)
                    if bayesian_transparent_probability == 0.0:
                        bayesian_log_probability = float("-inf")
                    elif not math.isinf(bayesian_log_probability):
                        bayesian_log_probability += math.log(bayesian_transparent_probability)

        image = Image.fromarray(sampled_image, mode="RGB")

        metadata = {
            "image_shape": list(self.image_shape),
            "num_source_avatars": num_images,
            "source_files": self.source_files,
            "opaque_pixel_count": opaque_pixel_count,
            "opaque_ratio": opaque_pixel_count / (height * width),
            "average_channel_support_count": float(np.mean(average_support_counts)) if average_support_counts else 0.0,
            "global_channel_prior_min": float(self.channel_priors.min()),
            "global_channel_prior_max": float(self.channel_priors.max()),
            "global_opaque_prior": self.global_opaque_prior,
            "mle_probability": probability_from_log(mle_log_probability),
            "mle_probability_scientific": scientific_notation_from_log(mle_log_probability),
            "mle_log_probability": mle_log_probability,
            "bayesian_probability": probability_from_log(bayesian_log_probability),
            "bayesian_probability_scientific": scientific_notation_from_log(bayesian_log_probability),
            "bayesian_log_probability": bayesian_log_probability,
            "bayesian_model": "pixelwise_posterior_predictive_with_opacity_prior",
            "alpha": self.alpha,
        }
        return image, metadata

    def sample_many(self, count: int) -> list[tuple[Image.Image, dict[str, Any]]]:
        return [self.sample() for _ in range(count)]
