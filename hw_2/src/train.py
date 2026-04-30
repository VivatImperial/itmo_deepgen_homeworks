import logging
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from hw_2.src.utils import save_epoch_metrics_csv, save_json, save_loss_plot, save_run_summary


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def _create_run_logger(output_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"hw2.train.{output_path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_path / "train.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _close_run_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _save_run_state(
    output_path: Path,
    run_config: dict[str, Any],
    history: dict[str, list[float]],
    epoch_metrics: list[dict[str, float | int]],
) -> None:
    metrics = {"config": run_config, "epochs": epoch_metrics}
    save_json(run_config, output_path / "config.json")
    save_json(history, output_path / "losses.json")
    save_json(metrics, output_path / "metrics.json")
    save_epoch_metrics_csv(epoch_metrics, output_path / "metrics.csv")
    save_run_summary(metrics, output_path / "summary.md")
    save_loss_plot(history, output_path / "loss_plot.png")


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    output_dir: str | Path,
    latent_dim: int,
    epochs: int,
    lr: float,
    beta1: float,
    device: torch.device,
    max_batches: int | None = None,
    sample_count: int = 64,
    config: dict[str, Any] | None = None,
) -> dict[str, list[float]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger = _create_run_logger(output_path)
    run_config = config or {}

    try:
        logger.info(
            "starting training | output_dir=%s | epochs=%s | latent_dim=%s | device=%s",
            output_path,
            epochs,
            latent_dim,
            device,
        )
        generator.to(device)
        discriminator.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        fixed_noise = torch.randn(sample_count, latent_dim, 1, 1, device=device)
        history: dict[str, list[float]] = {
            "g_loss": [],
            "d_loss": [],
            "d_x": [],
            "d_g_z_before": [],
            "d_g_z_after": [],
        }
        epoch_metrics: list[dict[str, float | int]] = []

        for epoch in range(1, epochs + 1):
            started_at = perf_counter()
            epoch_g_loss: list[float] = []
            epoch_d_loss: list[float] = []
            epoch_d_x: list[float] = []
            epoch_d_g_z_before: list[float] = []
            epoch_d_g_z_after: list[float] = []
            logger.info("epoch %s/%s started", epoch, epochs)
            progress = tqdm(dataloader, desc=f"epoch {epoch}/{epochs}", leave=False)
            for batch_index, (real_images, _) in enumerate(progress, start=1):
                if max_batches is not None and batch_index > max_batches:
                    break

                real_images = real_images.to(device)
                batch_size = real_images.size(0)
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)

                discriminator.zero_grad(set_to_none=True)
                real_logits = discriminator(real_images)
                d_real_loss = criterion(real_logits, real_labels)
                noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_images = generator(noise)
                fake_logits = discriminator(fake_images.detach())
                d_fake_loss = criterion(fake_logits, fake_labels)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                optimizer_d.step()
                d_x = torch.sigmoid(real_logits).mean().detach()
                d_g_z_before = torch.sigmoid(fake_logits).mean().detach()

                generator.zero_grad(set_to_none=True)
                fake_logits_for_g = discriminator(fake_images)
                g_loss = criterion(fake_logits_for_g, real_labels)
                g_loss.backward()
                optimizer_g.step()
                d_g_z_after = torch.sigmoid(fake_logits_for_g).mean().detach()

                g_loss_value = float(g_loss.detach().cpu())
                d_loss_value = float(d_loss.detach().cpu())
                d_x_value = float(d_x.cpu())
                d_g_z_before_value = float(d_g_z_before.cpu())
                d_g_z_after_value = float(d_g_z_after.cpu())

                history["g_loss"].append(g_loss_value)
                history["d_loss"].append(d_loss_value)
                history["d_x"].append(d_x_value)
                history["d_g_z_before"].append(d_g_z_before_value)
                history["d_g_z_after"].append(d_g_z_after_value)
                epoch_g_loss.append(g_loss_value)
                epoch_d_loss.append(d_loss_value)
                epoch_d_x.append(d_x_value)
                epoch_d_g_z_before.append(d_g_z_before_value)
                epoch_d_g_z_after.append(d_g_z_after_value)
                progress.set_postfix(g_loss=history["g_loss"][-1], d_loss=history["d_loss"][-1])

            if not epoch_g_loss:
                raise RuntimeError(f"epoch {epoch} finished without training iterations")

            with torch.no_grad():
                samples = generator(fixed_noise).detach().cpu()
            save_image(samples, output_path / f"samples_epoch_{epoch:03d}.png", normalize=True, value_range=(-1, 1))
            torch.save(generator.state_dict(), output_path / f"generator_epoch_{epoch:03d}.pt")
            epoch_metric = {
                "epoch": epoch,
                "iterations": len(epoch_g_loss),
                "g_loss_mean": _mean(epoch_g_loss),
                "d_loss_mean": _mean(epoch_d_loss),
                "d_x_mean": _mean(epoch_d_x),
                "d_g_z_before_mean": _mean(epoch_d_g_z_before),
                "d_g_z_after_mean": _mean(epoch_d_g_z_after),
                "duration_sec": perf_counter() - started_at,
            }
            epoch_metrics.append(epoch_metric)
            _save_run_state(output_path, run_config, history, epoch_metrics)
            logger.info(
                (
                    "epoch %s/%s completed | iterations=%s | g_loss=%.6f | d_loss=%.6f | "
                    "D(x)=%.6f | D(G(z)) before=%.6f | D(G(z)) after=%.6f | duration=%.3fs"
                ),
                epoch,
                epochs,
                epoch_metric["iterations"],
                epoch_metric["g_loss_mean"],
                epoch_metric["d_loss_mean"],
                epoch_metric["d_x_mean"],
                epoch_metric["d_g_z_before_mean"],
                epoch_metric["d_g_z_after_mean"],
                epoch_metric["duration_sec"],
            )

        logger.info("training completed | epochs=%s | iterations=%s", epochs, len(history["g_loss"]))
        return history
    finally:
        _close_run_logger(logger)
