from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def center_crop_image(image: Image.Image, crop_ratio: float) -> Image.Image:
    if not 0.0 < crop_ratio <= 1.0:
        raise ValueError("crop_ratio must be in (0, 1].")
    if crop_ratio >= 1.0:
        return image.copy()

    width, height = image.size
    crop_width = max(1, int(round(width * crop_ratio)))
    crop_height = max(1, int(round(height * crop_ratio)))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height))


def resize_and_pad(image: Image.Image, target_size: int) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), Image.BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), color=(0, 0, 0))
    left = (target_size - resized_width) // 2
    top = (target_size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return TF.to_tensor(canvas)


def preprocess_image(
    image: Image.Image,
    target_size: int,
    input_mode: str = "full_image",
    roi_ratio: float = 1.0,
) -> torch.Tensor:
    if input_mode == "full_image":
        prepared = image
    elif input_mode == "central_roi":
        prepared = center_crop_image(image, roi_ratio)
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    return resize_and_pad(prepared, target_size)


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


class ImageDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(
        self,
        paths: list[Path],
        image_size: int,
        augment: bool = False,
        input_mode: str = "full_image",
        roi_ratio: float = 1.0,
    ) -> None:
        self.paths = paths
        self.image_size = image_size
        self.augment = augment
        self.input_mode = input_mode
        self.roi_ratio = roi_ratio
        self.augment_transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            resized = preprocess_image(
                image,
                self.image_size,
                input_mode=self.input_mode,
                roi_ratio=self.roi_ratio,
            )
        tensor = self.augment_transform(resized) if self.augment else resized
        return tensor, str(path)


class LabeledImageDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        image_size: int,
        input_mode: str = "full_image",
        roi_ratio: float = 1.0,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.input_mode = input_mode
        self.roi_ratio = roi_ratio

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = preprocess_image(
                image,
                self.image_size,
                input_mode=self.input_mode,
                roi_ratio=self.roi_ratio,
            )
        return tensor, label, str(path)


@dataclass(slots=True)
class DatasetSplit:
    train_normal: list[Path]
    val_normal: list[Path]
    proliv: list[Path]
    test_samples: list[tuple[Path, int]]


def load_task2_splits(dataset_root: Path, val_fraction: float = 0.1, seed: int = 42) -> DatasetSplit:
    train_paths = list_images(dataset_root / "train")
    proliv_paths = list_images(dataset_root / "proliv")
    test_dir = dataset_root / "test" / "imgs"
    annotation_path = dataset_root / "test" / "test_annotation.txt"

    lines = annotation_path.read_text(encoding="utf-8").strip().splitlines()
    test_labels = {}
    for line in lines:
        filename, label = line.split()
        test_labels[filename] = int(label)

    test_paths = list_images(test_dir)
    test_samples = [(path, test_labels[path.name]) for path in test_paths]

    shuffled_train_paths = train_paths[:]
    random.Random(seed).shuffle(shuffled_train_paths)
    split_index = int(len(shuffled_train_paths) * (1.0 - val_fraction))
    train_normal = shuffled_train_paths[:split_index]
    val_normal = shuffled_train_paths[split_index:]

    return DatasetSplit(
        train_normal=train_normal,
        val_normal=val_normal,
        proliv=proliv_paths,
        test_samples=test_samples,
    )
