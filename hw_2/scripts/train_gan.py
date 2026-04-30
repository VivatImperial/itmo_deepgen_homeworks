import argparse
from pathlib import Path

from hw_2.src.data import create_image_dataloader
from hw_2.src.models import Discriminator, Generator, weights_init
from hw_2.src.train import train_gan
from hw_2.src.utils import select_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HW2 GAN on a prepared CelebA ImageFolder dataset.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("hw_2/artifacts/batch"))
    parser.add_argument("--norm", choices=["batch", "instance", "layer", "none"], default="batch")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--device", default="auto")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size != 64:
        raise ValueError("image_size must be 64 for the current generator and discriminator")
    if args.base_channels <= 0 or args.base_channels % 2 != 0:
        raise ValueError("base_channels must be positive and even")


def run_training_from_args(args: argparse.Namespace) -> dict[str, list[float]]:
    validate_args(args)
    set_seed(args.seed)
    device = select_device(args.device)
    dataloader = create_image_dataloader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        max_items=args.max_items,
    )
    generator = Generator(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        image_channels=3,
        norm_type=args.norm,
    )
    discriminator = Discriminator(base_channels=args.base_channels, image_channels=3)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    return train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        device=device,
        max_batches=args.max_batches,
        sample_count=args.sample_count,
        config={
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
            "norm": args.norm,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
            "lr": args.lr,
            "beta1": args.beta1,
            "workers": args.workers,
            "max_items": args.max_items,
            "max_batches": args.max_batches,
            "sample_count": args.sample_count,
            "seed": args.seed,
            "device": str(device),
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training_from_args(args)


if __name__ == "__main__":
    main()
