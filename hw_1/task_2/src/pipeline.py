from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hw_1.task_2.src.data import DatasetSplit, ImageDataset, LabeledImageDataset, load_task2_splits
from hw_1.task_2.src.model import MobileNetV3Autoencoder


def configure_logging(*, verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)


@dataclass(slots=True)
class TrainConfig:
    image_size: int = 64
    batch_size: int = 128
    epochs: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    num_workers: int = 0
    seed: int = 42
    train_augment: bool = True
    input_mode: str = "full_image"
    train_roi: float = 1.0
    train_patch_size: int = 64
    tile_size: int = 64
    tile_stride: int = 32
    tile_agg_mode: str = "max"
    score_roi: float = 1.0
    score_mode: str = "mean"
    score_param: float = 0.2
    anomaly_score_mode: str = "reconstruction"
    latent_score_weight: float = 0.5
    latent_covariance_eps: float = 1e-3
    threshold_mode: str = "balanced"
    proliv_quantile: float = 0.05
    encoder_name: str = "mobilevitv2_150.cvnets_in1k"
    encoder_out_index: int = 4
    bottleneck_channels: int = 128
    decoder_mode: str = "plain"
    freeze_encoder: bool = False
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    denoising_noise_std: float = 0.0
    denoising_mask_ratio: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("hw_1/task_2/data/unpacked/dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hw_1/task_2/artifacts"),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-mode",
        choices=("full_image", "central_roi", "random_patch"),
        default="full_image",
    )
    parser.add_argument("--train-roi", type=float, default=1.0)
    parser.add_argument("--train-patch-size", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--tile-stride", type=int, default=32)
    parser.add_argument(
        "--tile-agg-mode",
        choices=("mean", "topk_mean", "pixel_quantile", "max"),
        default="max",
    )
    parser.add_argument("--score-roi", type=float, default=1.0)
    parser.add_argument(
        "--score-mode",
        choices=("mean", "topk_mean", "pixel_quantile", "max"),
        default="mean",
    )
    parser.add_argument("--score-param", type=float, default=0.2)
    parser.add_argument(
        "--anomaly-score-mode",
        choices=("reconstruction", "latent_mahalanobis", "combined"),
        default="reconstruction",
    )
    parser.add_argument("--latent-score-weight", type=float, default=0.5)
    parser.add_argument("--latent-covariance-eps", type=float, default=1e-3)
    parser.add_argument("--encoder-name", type=str, default="mobilevitv2_150.cvnets_in1k")
    parser.add_argument("--encoder-out-index", type=int, default=4)
    parser.add_argument("--bottleneck-channels", type=int, default=128)
    parser.add_argument("--decoder-mode", choices=("plain", "skip"), default="plain")
    parser.add_argument("--train-encoder", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--denoising-noise-std", type=float, default=0.0)
    parser.add_argument("--denoising-mask-ratio", type=float, default=0.0)
    parser.add_argument(
        "--threshold-mode",
        choices=("balanced", "proliv_quantile"),
        default="balanced",
    )
    parser.add_argument("--proliv-quantile", type=float, default=0.05)
    parser.add_argument("--disable-train-augment", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Show only warnings and errors.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def apply_roi(images: torch.Tensor, roi: float) -> torch.Tensor:
    if roi >= 1.0:
        return images
    _, _, height, width = images.shape
    delta_h = int(height * (1.0 - roi) / 2.0)
    delta_w = int(width * (1.0 - roi) / 2.0)
    return images[:, :, delta_h : height - delta_h, delta_w : width - delta_w]


def apply_denoising_corruption(images: torch.Tensor, noise_std: float, mask_ratio: float) -> torch.Tensor:
    if noise_std <= 0.0 and mask_ratio <= 0.0:
        return images

    corrupted = images.clone()
    if noise_std > 0.0:
        corrupted = (corrupted + torch.randn_like(corrupted) * noise_std).clamp(0.0, 1.0)

    if mask_ratio > 0.0:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("denoising_mask_ratio must be in (0, 1).")
        batch_size, _channels, height, width = corrupted.shape
        mask_height = max(1, int(round(height * mask_ratio**0.5)))
        mask_width = max(1, int(round(width * mask_ratio**0.5)))
        fill_values = torch.mean(corrupted, dim=(2, 3), keepdim=True)
        for index in range(batch_size):
            top = torch.randint(0, max(1, height - mask_height + 1), (1,), device=corrupted.device).item()
            left = torch.randint(0, max(1, width - mask_width + 1), (1,), device=corrupted.device).item()
            corrupted[index, :, top : top + mask_height, left : left + mask_width] = fill_values[index]

    return corrupted


def center_crop_tensor(images: torch.Tensor, crop_ratio: float) -> torch.Tensor:
    if not 0.0 < crop_ratio <= 1.0:
        raise ValueError("train_roi must be in (0, 1].")
    return apply_roi(images, crop_ratio)


def center_crop_to_size(images: torch.Tensor, crop_size: int) -> torch.Tensor:
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    _, _, height, width = images.shape
    if crop_size > min(height, width):
        raise ValueError("crop_size cannot exceed input spatial size.")
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return images[:, :, top : top + crop_size, left : left + crop_size]


def sample_random_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size <= 0:
        raise ValueError("train_patch_size must be positive.")
    batch_size, _channels, height, width = images.shape
    if patch_size > min(height, width):
        raise ValueError("train_patch_size cannot exceed input spatial size.")

    patches = []
    max_top = height - patch_size
    max_left = width - patch_size
    for index in range(batch_size):
        top = 0 if max_top == 0 else torch.randint(0, max_top + 1, (1,), device=images.device).item()
        left = 0 if max_left == 0 else torch.randint(0, max_left + 1, (1,), device=images.device).item()
        patches.append(images[index, :, top : top + patch_size, left : left + patch_size])
    return torch.stack(patches, dim=0)


def prepare_model_inputs(images: torch.Tensor, config: TrainConfig, training: bool) -> torch.Tensor:
    if config.input_mode == "full_image":
        return images
    if config.input_mode == "central_roi":
        return images
    if config.input_mode == "random_patch":
        if training:
            return sample_random_patches(images, config.train_patch_size)
        return center_crop_to_size(images, config.train_patch_size)
    raise ValueError(f"Unsupported input_mode: {config.input_mode}")


def aggregate_error_map(error_map: torch.Tensor, score_mode: str, score_param: float) -> torch.Tensor:
    flat = error_map.reshape(error_map.shape[0], -1)
    if score_mode == "mean":
        return torch.mean(flat, dim=1)
    if score_mode == "max":
        return torch.max(flat, dim=1).values
    if score_mode == "topk_mean":
        if not 0.0 < score_param <= 1.0:
            raise ValueError("score_param must be in (0, 1] for topk_mean mode.")
        topk_count = max(1, int(round(flat.shape[1] * score_param)))
        return torch.topk(flat, k=topk_count, dim=1).values.mean(dim=1)
    if score_mode == "pixel_quantile":
        if not 0.0 < score_param < 1.0:
            raise ValueError("score_param must be in (0, 1) for pixel_quantile mode.")
        return torch.quantile(flat, q=score_param, dim=1)
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def aggregate_score_matrix(score_matrix: torch.Tensor, score_mode: str, score_param: float) -> torch.Tensor:
    if score_mode == "mean":
        return torch.mean(score_matrix, dim=1)
    if score_mode == "max":
        return torch.max(score_matrix, dim=1).values
    if score_mode == "topk_mean":
        if not 0.0 < score_param <= 1.0:
            raise ValueError("score_param must be in (0, 1] for topk_mean mode.")
        topk_count = max(1, int(round(score_matrix.shape[1] * score_param)))
        return torch.topk(score_matrix, k=topk_count, dim=1).values.mean(dim=1)
    if score_mode == "pixel_quantile":
        if not 0.0 < score_param < 1.0:
            raise ValueError("score_param must be in (0, 1) for pixel_quantile mode.")
        return torch.quantile(score_matrix, q=score_param, dim=1)
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def sliding_window_positions(spatial_size: int, tile_size: int, tile_stride: int) -> list[int]:
    if tile_size <= 0 or tile_stride <= 0:
        raise ValueError("tile_size and tile_stride must be positive.")
    if tile_size > spatial_size:
        raise ValueError("tile_size cannot exceed input spatial size.")
    positions = list(range(0, spatial_size - tile_size + 1, tile_stride))
    last_position = spatial_size - tile_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def reconstruction_tile_scores(
    model: nn.Module,
    images: torch.Tensor,
    score_mode: str,
    score_param: float,
    tile_size: int,
    tile_stride: int,
    tile_agg_mode: str,
) -> torch.Tensor:
    batch_size, channels, height, width = images.shape
    tile_height = min(tile_size, height)
    tile_width = min(tile_size, width)
    tops = sliding_window_positions(height, tile_height, tile_stride)
    lefts = sliding_window_positions(width, tile_width, tile_stride)
    tiles = [
        images[:, :, top : top + tile_height, left : left + tile_width]
        for top in tops
        for left in lefts
    ]
    tile_batch = torch.stack(tiles, dim=1).reshape(-1, channels, tile_height, tile_width)
    tile_reconstructions = model(tile_batch)
    tile_error_map = torch.mean((tile_reconstructions - tile_batch) ** 2, dim=1)
    tile_scores = aggregate_error_map(tile_error_map, score_mode, score_param).reshape(batch_size, -1)
    return aggregate_score_matrix(tile_scores, tile_agg_mode, score_param)


def reconstruction_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_roi: float,
    score_mode: str,
    score_param: float,
    config: TrainConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    model.eval()
    scores: list[float] = []
    names: list[str] = []
    input_mode = "full_image" if config is None else config.input_mode
    tile_size = 64 if config is None else config.tile_size
    tile_stride = 32 if config is None else config.tile_stride
    tile_agg_mode = "max" if config is None else config.tile_agg_mode
    with torch.inference_mode():
        for images, paths in loader:
            images = images.to(device)
            if input_mode == "random_patch":
                batch_scores = reconstruction_tile_scores(
                    model,
                    images,
                    score_mode,
                    score_param,
                    tile_size,
                    tile_stride,
                    tile_agg_mode,
                )
            else:
                reconstructions = model(images)
                roi_images = apply_roi(images, score_roi)
                roi_reconstructions = apply_roi(reconstructions, score_roi)
                error_map = torch.mean((roi_reconstructions - roi_images) ** 2, dim=1)
                batch_scores = aggregate_error_map(error_map, score_mode, score_param)
            scores.extend(batch_scores.detach().cpu().numpy().tolist())
            names.extend(Path(path).name for path in paths)
    return np.asarray(scores, dtype=np.float64), names


def reconstruction_scores_labeled(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_roi: float,
    score_mode: str,
    score_param: float,
    config: TrainConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    names: list[str] = []
    input_mode = "full_image" if config is None else config.input_mode
    tile_size = 64 if config is None else config.tile_size
    tile_stride = 32 if config is None else config.tile_stride
    tile_agg_mode = "max" if config is None else config.tile_agg_mode
    with torch.inference_mode():
        for images, batch_labels, paths in loader:
            images = images.to(device)
            if input_mode == "random_patch":
                batch_scores = reconstruction_tile_scores(
                    model,
                    images,
                    score_mode,
                    score_param,
                    tile_size,
                    tile_stride,
                    tile_agg_mode,
                )
            else:
                reconstructions = model(images)
                roi_images = apply_roi(images, score_roi)
                roi_reconstructions = apply_roi(reconstructions, score_roi)
                error_map = torch.mean((roi_reconstructions - roi_images) ** 2, dim=1)
                batch_scores = aggregate_error_map(error_map, score_mode, score_param)
            scores.extend(batch_scores.detach().cpu().numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())
            names.extend(Path(path).name for path in paths)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64), names


def latent_embeddings(
    model: MobileNetV3Autoencoder,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig | None = None,
) -> np.ndarray:
    model.eval()
    embeddings: list[np.ndarray] = []
    input_mode = "full_image" if config is None else config.input_mode
    train_patch_size = 64 if config is None else config.train_patch_size
    with torch.inference_mode():
        for batch in loader:
            images = batch[0]
            images = images.to(device)
            if input_mode == "random_patch":
                images = center_crop_to_size(images, train_patch_size)
            encoded = model.encode(images)
            pooled = torch.mean(encoded, dim=(2, 3))
            embeddings.append(pooled.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def fit_latent_gaussian(train_embeddings: np.ndarray, covariance_eps: float) -> dict[str, np.ndarray | float]:
    mean = np.mean(train_embeddings, axis=0, dtype=np.float64)
    covariance = np.cov(train_embeddings.astype(np.float64), rowvar=False)
    covariance += np.eye(covariance.shape[0], dtype=np.float64) * covariance_eps
    precision = np.linalg.inv(covariance)
    return {
        "mean": mean,
        "precision": precision,
        "covariance_eps": float(covariance_eps),
    }


def latent_mahalanobis_scores(
    embeddings: np.ndarray,
    latent_stats: dict[str, np.ndarray | float],
) -> np.ndarray:
    mean = np.asarray(latent_stats["mean"], dtype=np.float64)
    precision = np.asarray(latent_stats["precision"], dtype=np.float64)
    centered = embeddings.astype(np.float64) - mean
    return np.einsum("bi,ij,bj->b", centered, precision, centered).astype(np.float64)


def fit_standardizer(reference_scores: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(reference_scores))
    std = float(np.std(reference_scores))
    if std < 1e-12:
        std = 1.0
    return {"mean": mean, "std": std}


def apply_standardizer(scores: np.ndarray, standardizer: dict[str, float]) -> np.ndarray:
    return (scores - standardizer["mean"]) / standardizer["std"]


def build_anomaly_scores(
    config: TrainConfig,
    recon_scores: dict[str, np.ndarray],
    latent_scores: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if config.anomaly_score_mode == "reconstruction":
        return recon_scores, {"mode": config.anomaly_score_mode}

    if config.anomaly_score_mode == "latent_mahalanobis":
        return latent_scores, {"mode": config.anomaly_score_mode}

    if config.anomaly_score_mode != "combined":
        raise ValueError(f"Unsupported anomaly_score_mode: {config.anomaly_score_mode}")

    if not 0.0 <= config.latent_score_weight <= 1.0:
        raise ValueError("latent_score_weight must be in [0, 1].")

    recon_standardizer = fit_standardizer(recon_scores["val"])
    latent_standardizer = fit_standardizer(latent_scores["val"])
    combined_scores = {}
    for split_name in recon_scores:
        recon_standardized = apply_standardizer(recon_scores[split_name], recon_standardizer)
        latent_standardized = apply_standardizer(latent_scores[split_name], latent_standardizer)
        combined_scores[split_name] = (
            config.latent_score_weight * recon_standardized
            + (1.0 - config.latent_score_weight) * latent_standardized
        ).astype(np.float64)

    return combined_scores, {
        "mode": config.anomaly_score_mode,
        "reconstruction_weight": float(config.latent_score_weight),
        "latent_weight": float(1.0 - config.latent_score_weight),
        "reconstruction_standardizer": recon_standardizer,
        "latent_standardizer": latent_standardizer,
    }


def choose_threshold_balanced(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> dict[str, float]:
    candidates = np.unique(np.concatenate([normal_scores, anomaly_scores]))
    best_threshold = float(candidates[0])
    best_accuracy = -1.0
    best_tpr = 0.0
    best_tnr = 0.0

    for threshold in candidates:
        tpr = float(np.mean(anomaly_scores > threshold))
        tnr = float(np.mean(normal_scores <= threshold))
        balanced_accuracy = 0.5 * (tpr + tnr)
        if balanced_accuracy > best_accuracy:
            best_accuracy = balanced_accuracy
            best_threshold = float(threshold)
            best_tpr = tpr
            best_tnr = tnr

    return {
        "threshold": best_threshold,
        "balanced_accuracy": best_accuracy,
        "threshold_tpr": best_tpr,
        "threshold_tnr": best_tnr,
        "mode": "balanced",
    }


def choose_threshold_proliv_quantile(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    quantile: float,
) -> dict[str, float]:
    threshold = float(np.quantile(anomaly_scores, quantile))
    tpr = float(np.mean(anomaly_scores > threshold))
    tnr = float(np.mean(normal_scores <= threshold))
    return {
        "threshold": threshold,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "threshold_tpr": tpr,
        "threshold_tnr": tnr,
        "mode": "proliv_quantile",
        "quantile": float(quantile),
    }


def save_histogram(normal_scores: np.ndarray, anomaly_scores: np.ndarray, threshold: float, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=40, alpha=0.7, label="val_normal", color="#4C72B0")
    plt.hist(anomaly_scores, bins=40, alpha=0.7, label="proliv", color="#DD8452")
    plt.axvline(threshold, color="#000000", linestyle="--", label=f"threshold={threshold:.6f}")
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.title("Anomaly score distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_latent_projection(
    normal_embeddings: np.ndarray,
    anomaly_embeddings: np.ndarray,
    path: Path,
) -> None:
    features = np.concatenate([normal_embeddings, anomaly_embeddings], axis=0)
    labels = np.concatenate(
        [
            np.zeros(len(normal_embeddings), dtype=np.int64),
            np.ones(len(anomaly_embeddings), dtype=np.int64),
        ]
    )
    projection = PCA(n_components=2).fit_transform(features)

    plt.figure(figsize=(9, 7))
    plt.scatter(
        projection[labels == 0, 0],
        projection[labels == 0, 1],
        s=14,
        alpha=0.65,
        label="val_normal",
        color="#4C72B0",
    )
    plt.scatter(
        projection[labels == 1, 0],
        projection[labels == 1, 1],
        s=24,
        alpha=0.75,
        label="proliv",
        color="#DD8452",
    )
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Latent space projection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    config: TrainConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        images, paths = next(iter(loader))
        images = images[:8].to(device)
        targets = prepare_model_inputs(images, config, training=False)
        reconstructions = model(targets).detach().cpu()
        originals = targets.detach().cpu()

    for index, (original, reconstruction) in enumerate(zip(originals, reconstructions, strict=True), start=1):
        original_path = output_dir / f"sample_{index:02d}_original.png"
        reconstruction_path = output_dir / f"sample_{index:02d}_reconstruction.png"
        diff_path = output_dir / f"sample_{index:02d}_diff.png"
        save_tensor_image(original, original_path)
        save_tensor_image(reconstruction, reconstruction_path)
        save_tensor_image(torch.abs(original - reconstruction), diff_path)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    plt.imsave(path, array)


def save_loss_curves(history: dict[str, list[float]], best_epoch: int, path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["train_loss"], label="train_loss", color="#4C72B0", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="val_loss", color="#DD8452", linewidth=2)
    plt.axvline(best_epoch, color="#000000", linestyle="--", label=f"best_epoch={best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_dataloaders(split: DatasetSplit, config: TrainConfig) -> dict[str, DataLoader]:
    dataset_input_mode = "central_roi" if config.input_mode == "central_roi" else "full_image"
    roi_ratio = config.train_roi if dataset_input_mode == "central_roi" else 1.0
    train_ds = ImageDataset(
        split.train_normal,
        image_size=config.image_size,
        augment=config.train_augment,
        input_mode=dataset_input_mode,
        roi_ratio=roi_ratio,
    )
    val_ds = ImageDataset(
        split.val_normal,
        image_size=config.image_size,
        input_mode=dataset_input_mode,
        roi_ratio=roi_ratio,
    )
    proliv_ds = ImageDataset(
        split.proliv,
        image_size=config.image_size,
        input_mode=dataset_input_mode,
        roi_ratio=roi_ratio,
    )
    test_ds = LabeledImageDataset(
        split.test_samples,
        image_size=config.image_size,
        input_mode=dataset_input_mode,
        roi_ratio=roi_ratio,
    )

    return {
        "train": DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers),
        "val": DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers),
        "proliv": DataLoader(proliv_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers),
        "test": DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    *,
    show_progress: bool = True,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    monitored_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}", leave=False, disable=not show_progress)
        for images, _paths in progress:
            images = images.to(device)
            targets = prepare_model_inputs(images, config, training=True)
            inputs = apply_denoising_corruption(targets, config.denoising_noise_std, config.denoising_mask_ratio)
            optimizer.zero_grad(set_to_none=True)
            reconstructions = model(inputs)
            loss = criterion(reconstructions, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            progress.set_postfix(train_loss=f"{np.mean(train_losses):.6f}")

        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for images, _paths in val_loader:
                images = images.to(device)
                targets = prepare_model_inputs(images, config, training=False)
                reconstructions = model(targets)
                val_loss = criterion(reconstructions, targets)
                val_losses.append(float(val_loss.detach().cpu().item()))

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))
        logging.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f",
            epoch,
            history["train_loss"][-1],
            history["val_loss"][-1],
        )

        current_val_loss = history["val_loss"][-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if current_val_loss < monitored_val_loss - config.early_stopping_min_delta:
            monitored_val_loss = current_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                stopped_early = True
                logging.info(
                    "early_stopping epoch=%d best_epoch=%d best_val_loss=%.6f",
                    epoch,
                    best_epoch,
                    best_val_loss,
                )
                break

    model.load_state_dict(best_state_dict)
    return {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "epochs_ran": len(history["train_loss"]),
        "stopped_early": stopped_early,
    }


def evaluate_test(test_scores: np.ndarray, test_labels: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (test_scores > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = float(np.mean(predictions == test_labels))
    return {
        "accuracy": accuracy,
        "tpr": float(tpr),
        "tnr": float(tnr),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def encoder_source_name(encoder_name: str) -> str:
    if encoder_name.startswith("facebook/vit-mae"):
        return "huggingface_transformers"
    return "huggingface_timm"


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)
    config = TrainConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        train_augment=not args.disable_train_augment,
        input_mode=args.input_mode,
        train_roi=args.train_roi,
        train_patch_size=args.train_patch_size,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        tile_agg_mode=args.tile_agg_mode,
        score_roi=args.score_roi,
        score_mode=args.score_mode,
        score_param=args.score_param,
        anomaly_score_mode=args.anomaly_score_mode,
        latent_score_weight=args.latent_score_weight,
        latent_covariance_eps=args.latent_covariance_eps,
        threshold_mode=args.threshold_mode,
        proliv_quantile=args.proliv_quantile,
        encoder_name=args.encoder_name,
        encoder_out_index=args.encoder_out_index,
        bottleneck_channels=args.bottleneck_channels,
        decoder_mode=args.decoder_mode,
        freeze_encoder=not args.train_encoder,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        denoising_noise_std=args.denoising_noise_std,
        denoising_mask_ratio=args.denoising_mask_ratio,
    )

    set_seed(config.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Starting task 2 pipeline")
    logging.info("Dataset root: %s", args.dataset_root)
    logging.info("Output directory: %s", output_dir)

    split = load_task2_splits(args.dataset_root, val_fraction=config.val_fraction, seed=config.seed)
    loaders = make_dataloaders(split, config)
    device = detect_device()
    logging.info("device=%s", device)
    logging.info(
        "train_normal=%d val_normal=%d proliv=%d test=%d",
        len(split.train_normal),
        len(split.val_normal),
        len(split.proliv),
        len(split.test_samples),
    )
    logging.info(
        "encoder=%s decoder=%s image_size=%d batch_size=%d epochs=%d",
        config.encoder_name,
        config.decoder_mode,
        config.image_size,
        config.batch_size,
        config.epochs,
    )

    logging.info("Building model")
    model = MobileNetV3Autoencoder(
        encoder_name=config.encoder_name,
        encoder_out_index=config.encoder_out_index,
        bottleneck_channels=config.bottleneck_channels,
        image_size=config.image_size,
        freeze_encoder=config.freeze_encoder,
        decoder_mode=config.decoder_mode,
    ).to(device)
    logging.info("Training autoencoder")
    training_result = train_model(model, loaders["train"], loaders["val"], config, device, show_progress=not args.quiet)

    logging.info("Scoring validation, proliv and test splits")
    val_recon_scores, val_names = reconstruction_scores(
        model, loaders["val"], device, config.score_roi, config.score_mode, config.score_param, config
    )
    proliv_recon_scores, proliv_names = reconstruction_scores(
        model, loaders["proliv"], device, config.score_roi, config.score_mode, config.score_param, config
    )
    test_recon_scores, test_labels, test_names = reconstruction_scores_labeled(
        model, loaders["test"], device, config.score_roi, config.score_mode, config.score_param, config
    )

    train_embeddings = latent_embeddings(model, loaders["train"], device, config)
    val_embeddings = latent_embeddings(model, loaders["val"], device, config)
    proliv_embeddings = latent_embeddings(model, loaders["proliv"], device, config)
    test_embeddings = latent_embeddings(model, loaders["test"], device, config)
    latent_stats = fit_latent_gaussian(train_embeddings, config.latent_covariance_eps)
    recon_scores = {
        "val": val_recon_scores,
        "proliv": proliv_recon_scores,
        "test": test_recon_scores,
    }
    latent_scores = {
        "val": latent_mahalanobis_scores(val_embeddings, latent_stats),
        "proliv": latent_mahalanobis_scores(proliv_embeddings, latent_stats),
        "test": latent_mahalanobis_scores(test_embeddings, latent_stats),
    }
    anomaly_scores, anomaly_score_info = build_anomaly_scores(config, recon_scores, latent_scores)
    val_scores = anomaly_scores["val"]
    proliv_scores = anomaly_scores["proliv"]
    test_scores = anomaly_scores["test"]

    if config.threshold_mode == "proliv_quantile":
        threshold_info = choose_threshold_proliv_quantile(val_scores, proliv_scores, config.proliv_quantile)
    else:
        threshold_info = choose_threshold_balanced(val_scores, proliv_scores)
    test_metrics = evaluate_test(test_scores, test_labels, threshold_info["threshold"])
    logging.info(
        "threshold=%.8f mode=%s test_accuracy=%.4f test_tpr=%.4f test_tnr=%.4f",
        threshold_info["threshold"],
        threshold_info["mode"],
        test_metrics["accuracy"],
        test_metrics["tpr"],
        test_metrics["tnr"],
    )

    checkpoint_name = config.encoder_name.replace(".", "_").replace("/", "_")
    model_path = output_dir / f"{checkpoint_name}_autoencoder.pt"
    logging.info("Saving model checkpoint to %s", model_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "threshold": threshold_info["threshold"],
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
        },
        model_path,
    )

    save_loss_curves(
        {
            "train_loss": training_result["train_loss"],
            "val_loss": training_result["val_loss"],
        },
        int(training_result["best_epoch"]),
        output_dir / "loss_curves.png",
    )
    save_histogram(val_scores, proliv_scores, threshold_info["threshold"], output_dir / "score_histogram.png")
    save_latent_projection(val_embeddings, proliv_embeddings, output_dir / "latent_space.png")
    save_examples(model, loaders["val"], device, output_dir / "reconstructions", config)
    logging.info("Saved plots and reconstruction examples")

    metrics = {
        "config": asdict(config),
        "device": str(device),
        "encoder": config.encoder_name,
        "encoder_source": encoder_source_name(config.encoder_name),
        "input_mode": config.input_mode,
        "train_roi": config.train_roi,
        "train_patch_size": config.train_patch_size,
        "tile_size": config.tile_size,
        "tile_stride": config.tile_stride,
        "tile_agg_mode": config.tile_agg_mode,
        "score_roi": config.score_roi,
        "score_mode": config.score_mode,
        "score_param": config.score_param,
        "anomaly_score_mode": config.anomaly_score_mode,
        "latent_score_weight": config.latent_score_weight,
        "anomaly_score_info": anomaly_score_info,
        "train_history": {
            "train_loss": training_result["train_loss"],
            "val_loss": training_result["val_loss"],
        },
        "training": {
            "best_epoch": int(training_result["best_epoch"]),
            "best_val_loss": float(training_result["best_val_loss"]),
            "epochs_ran": int(training_result["epochs_ran"]),
            "stopped_early": bool(training_result["stopped_early"]),
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
        },
        "threshold_info": threshold_info,
        "test_metrics": test_metrics,
        "latent_projection": {
            "method": "pca",
            "normal_samples": int(len(val_embeddings)),
            "anomaly_samples": int(len(proliv_embeddings)),
        },
        "latent_stats": {
            "covariance_eps": float(latent_stats["covariance_eps"]),
            "embedding_dim": int(train_embeddings.shape[1]),
            "train_samples": int(train_embeddings.shape[0]),
        },
        "val_score_summary": {
            "mean": float(np.mean(val_scores)),
            "std": float(np.std(val_scores)),
            "min": float(np.min(val_scores)),
            "max": float(np.max(val_scores)),
        },
        "proliv_score_summary": {
            "mean": float(np.mean(proliv_scores)),
            "std": float(np.std(proliv_scores)),
            "min": float(np.min(proliv_scores)),
            "max": float(np.max(proliv_scores)),
        },
        "samples": {
            "val_examples": [{"name": name, "score": float(score)} for name, score in zip(val_names[:20], val_scores[:20], strict=True)],
            "proliv_examples": [
                {"name": name, "score": float(score)}
                for name, score in zip(proliv_names[:20], proliv_scores[:20], strict=True)
            ],
            "test_examples": [
                {"name": name, "score": float(score), "label": int(label)}
                for name, score, label in zip(test_names[:20], test_scores[:20], test_labels[:20], strict=True)
            ],
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_lines = [
        "# Task 2 Results",
        "",
        f"- Device: `{device}`",
        f"- Encoder: `{config.encoder_name}` from `{encoder_source_name(config.encoder_name)}`",
        f"- Decoder mode: `{config.decoder_mode}`",
        f"- Input mode: `{config.input_mode}`",
        f"- Train ROI: `{config.train_roi:.2f}`",
        f"- Train patch size: `{config.train_patch_size}`",
        f"- Tile size: `{config.tile_size}`",
        f"- Tile stride: `{config.tile_stride}`",
        f"- Tile aggregation: `{config.tile_agg_mode}`",
        f"- Score ROI: `{config.score_roi:.2f}`",
        f"- Score mode: `{config.score_mode}`",
        f"- Score param: `{config.score_param:.4f}`",
        f"- Anomaly score mode: `{config.anomaly_score_mode}`",
        f"- Latent score weight: `{config.latent_score_weight:.2f}`",
        f"- Epochs requested: `{config.epochs}`",
        f"- Epochs ran: `{training_result['epochs_ran']}`",
        f"- Best epoch: `{training_result['best_epoch']}`",
        f"- Best val loss: `{training_result['best_val_loss']:.6f}`",
        f"- Early stopping: `{training_result['stopped_early']}`",
        f"- Early stopping patience: `{config.early_stopping_patience}`",
        f"- Denoising noise std: `{config.denoising_noise_std:.4f}`",
        f"- Denoising mask ratio: `{config.denoising_mask_ratio:.4f}`",
        f"- Threshold mode: `{threshold_info['mode']}`",
        f"- Threshold: `{threshold_info['threshold']:.8f}`",
        f"- Validation threshold TNR: `{threshold_info['threshold_tnr']:.4f}`",
        f"- Proliv threshold TPR: `{threshold_info['threshold_tpr']:.4f}`",
        f"- Test accuracy: `{test_metrics['accuracy']:.4f}`",
        f"- Test TPR: `{test_metrics['tpr']:.4f}`",
        f"- Test TNR: `{test_metrics['tnr']:.4f}`",
        "",
        "## Files",
        "",
        "- `metrics.json`",
        "- `loss_curves.png`",
        "- `score_histogram.png`",
        "- `latent_space.png`",
        f"- `{model_path.name}`",
        "- `reconstructions/`",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    logging.info("Saved metrics.json and summary.md")


if __name__ == "__main__":
    main()
