from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def create_image_dataloader(
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    workers: int,
    max_items: int | None = None,
    shuffle: bool = True,
) -> DataLoader:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"data_root is not a directory: {root}")

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(root=root, transform=transform)
    if max_items is not None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        dataset = Subset(dataset, range(min(max_items, len(dataset))))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
    )
