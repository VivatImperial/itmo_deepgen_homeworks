import argparse
import json
from pathlib import Path

from hw_2.scripts.train_gan import run_training_from_args


def build_experiment_args(
    data_root: Path,
    output_root: Path,
    norms: list[str],
    epochs: int,
    batch_size: int,
    image_size: int,
    latent_dim: int,
    base_channels: int,
    max_items: int | None,
    max_batches: int | None,
    device: str,
) -> list[dict[str, object]]:
    common = {
        "data_root": data_root,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "lr": 0.0002,
        "beta1": 0.5,
        "workers": 2,
        "max_items": max_items,
        "max_batches": max_batches,
        "sample_count": 64,
        "seed": 999,
        "device": device,
    }
    return [
        {
            **common,
            "norm": norm,
            "output_dir": output_root / norm,
        }
        for norm in norms
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HW2 GAN normalization experiments.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("hw_2/artifacts/norm_experiments"))
    parser.add_argument("--norms", nargs="+", choices=["batch", "instance", "layer", "none"], default=["batch", "instance", "none"])
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser


def write_comparison_summary(output_root: Path, norms: list[str]) -> None:
    rows: list[dict[str, object]] = []
    for norm in norms:
        metrics_path = output_root / norm / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics file does not exist: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        final_epoch = metrics["epochs"][-1]
        rows.append(
            {
                "norm": norm,
                "epoch": final_epoch["epoch"],
                "g_loss_mean": final_epoch["g_loss_mean"],
                "d_loss_mean": final_epoch["d_loss_mean"],
                "d_x_mean": final_epoch["d_x_mean"],
                "d_g_z_before_mean": final_epoch["d_g_z_before_mean"],
                "d_g_z_after_mean": final_epoch["d_g_z_after_mean"],
            }
        )

    lines = [
        "# Normalization Experiment Summary",
        "",
        "| Norm | Final epoch | G loss | D loss | D(x) | D(G(z)) before | D(G(z)) after |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {norm} | {epoch} | {g_loss_mean:.6f} | {d_loss_mean:.6f} | "
            "{d_x_mean:.6f} | {d_g_z_before_mean:.6f} | {d_g_z_after_mean:.6f} |".format(**row)
        )
    lines.append("")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    for experiment_args in build_experiment_args(
        data_root=args.data_root,
        output_root=args.output_root,
        norms=args.norms,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        max_items=args.max_items,
        max_batches=args.max_batches,
        device=args.device,
    ):
        run_training_from_args(argparse.Namespace(**experiment_args))
    write_comparison_summary(args.output_root, args.norms)


if __name__ == "__main__":
    main()
