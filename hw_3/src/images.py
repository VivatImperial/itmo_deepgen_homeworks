from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from hw_3.src.config import ensure_dir, ensure_file, ensure_parent_dir


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class FaceLandmarks:
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    left_mouth: tuple[float, float]
    right_mouth: tuple[float, float]

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.left_eye,
                self.right_eye,
                self.nose,
                self.left_mouth,
                self.right_mouth,
            ],
            dtype=np.float32,
        )


def _point(payload: dict[str, object], key: str) -> tuple[float, float]:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"Landmarks JSON must contain a two-value '{key}' point")
    return float(value[0]), float(value[1])


def load_landmarks(path: str | Path) -> FaceLandmarks:
    landmark_path = ensure_file(path, "landmarks JSON")
    with landmark_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected landmarks object in {landmark_path}")
    return FaceLandmarks(
        left_eye=_point(payload, "left_eye"),
        right_eye=_point(payload, "right_eye"),
        nose=_point(payload, "nose"),
        left_mouth=_point(payload, "left_mouth"),
        right_mouth=_point(payload, "right_mouth"),
    )


def _canonical_landmarks(output_size: int) -> np.ndarray:
    points = np.asarray(
        [
            (0.35, 0.40),
            (0.65, 0.40),
            (0.50, 0.55),
            (0.38, 0.72),
            (0.62, 0.72),
        ],
        dtype=np.float32,
    )
    return points * float(output_size)


def _similarity_transform_matrix(
    landmarks: FaceLandmarks,
    *,
    output_size: int,
) -> tuple[float, float, float, float, float, float]:
    source = landmarks.as_array().astype(np.float64)
    target = _canonical_landmarks(output_size).astype(np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center

    covariance = target_zero.T @ source_zero
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T

    scale = singular_values.sum() / np.square(target_zero).sum()
    linear = scale * rotation
    offset = source_center - linear @ target_center
    return (
        float(linear[0, 0]),
        float(linear[0, 1]),
        float(offset[0]),
        float(linear[1, 0]),
        float(linear[1, 1]),
        float(offset[1]),
    )


def list_images(directory: str | Path) -> list[Path]:
    root = ensure_dir(directory, "image directory")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def resize_image(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def align_face(image: Image.Image, landmarks: FaceLandmarks, *, output_size: int) -> Image.Image:
    if output_size < 1:
        raise ValueError("output_size must be positive")

    matrix = _similarity_transform_matrix(landmarks, output_size=output_size)
    return image.convert("RGB").transform(
        (output_size, output_size),
        Image.Transform.AFFINE,
        matrix,
        resample=Image.Resampling.BICUBIC,
    )


def save_image(image: Image.Image, path: str | Path) -> Path:
    output_path = ensure_parent_dir(path)
    image.save(output_path)
    return output_path


def make_image_grid(
    images: Iterable[Image.Image],
    *,
    columns: int,
    cell_size: int,
    padding: int = 0,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    image_list = list(images)
    if not image_list:
        raise ValueError("Cannot create a grid without images")
    if columns < 1:
        raise ValueError("columns must be positive")
    if cell_size < 1:
        raise ValueError("cell_size must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")

    rows = ceil(len(image_list) / columns)
    width = columns * cell_size + (columns - 1) * padding
    height = rows * cell_size + (rows - 1) * padding
    grid = Image.new("RGB", (width, height), color=background)

    for index, image in enumerate(image_list):
        row, column = divmod(index, columns)
        resized = resize_image(image.convert("RGB"), cell_size)
        x = column * (cell_size + padding)
        y = row * (cell_size + padding)
        grid.paste(resized, (x, y))

    return grid


def save_grid(
    images: Iterable[Image.Image],
    path: str | Path,
    *,
    columns: int,
    cell_size: int,
    padding: int = 0,
) -> Path:
    grid = make_image_grid(images, columns=columns, cell_size=cell_size, padding=padding)
    return save_image(grid, path)
