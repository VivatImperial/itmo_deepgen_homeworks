import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_epoch_metrics_csv(epoch_metrics: list[dict[str, float | int]], path: str | Path) -> None:
    fieldnames = [
        "epoch",
        "g_loss_mean",
        "d_loss_mean",
        "iterations",
        "d_x_mean",
        "d_g_z_before_mean",
        "d_g_z_after_mean",
        "duration_sec",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_metrics)


def save_run_summary(metrics: dict[str, Any], path: str | Path) -> None:
    config = metrics["config"]
    epochs = metrics["epochs"]
    final = epochs[-1]
    lines = [
        f"# GAN Run Summary: {config.get('norm', 'unknown')}",
        "",
        "## Config",
        "",
        f"- norm: `{config.get('norm')}`",
        f"- device: `{config.get('device')}`",
        f"- epochs: `{config.get('epochs')}`",
        f"- batch size: `{config.get('batch_size')}`",
        f"- lr: `{config.get('lr')}`",
        f"- seed: `{config.get('seed')}`",
        "",
        "## Final Epoch",
        "",
        f"- epoch: `{final['epoch']}`",
        f"- iterations: `{final['iterations']}`",
        f"- G loss mean: `{final['g_loss_mean']:.6f}`",
        f"- D loss mean: `{final['d_loss_mean']:.6f}`",
        f"- D(x): `{final['d_x_mean']:.6f}`",
        f"- D(G(z)) before G update: `{final['d_g_z_before_mean']:.6f}`",
        f"- D(G(z)) after G update: `{final['d_g_z_after_mean']:.6f}`",
        f"- duration sec: `{final['duration_sec']:.3f}`",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_loss_plot(history: dict[str, list[float]], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["g_loss"], label="G loss")
    ax.plot(history["d_loss"], label="D loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
