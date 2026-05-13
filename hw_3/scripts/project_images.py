from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm.auto import tqdm

from hw_3.src.config import ensure_parent_dir
from hw_3.src.images import list_images, load_rgb_image
from hw_3.src.latents import save_w_plus
from hw_3.src.logging_utils import configure_logging
from hw_3.src.stylegan import ProjectionConfig, StyleGANGenerator, tensor_to_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project aligned portraits into StyleGAN W+ space.")
    parser.add_argument("--input-dir", required=True, help="Directory with aligned portraits.")
    parser.add_argument("--output-dir", required=True, help="Directory for projected W+ latents.")
    parser.add_argument("--stylegan-checkpoint", required=True, help="Path to StyleGAN generator checkpoint.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--steps", type=int, default=500, help="Projection optimization steps.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Projection learning rate.")
    parser.add_argument("--image-size", type=int, default=1024, help="Projection image size.")
    parser.add_argument("--latent-layers", type=int, help="W+ layer count. Defaults to generator.num_ws.")
    parser.add_argument("--latent-channels", type=int, help="W+ channel count. Defaults to generator w_avg size.")
    parser.add_argument(
        "--projector",
        choices=("official", "mse"),
        default="official",
        help="Projection backend. official uses NVIDIA VGG perceptual projector.",
    )
    parser.add_argument("--w-avg-samples", type=int, default=10000, help="Samples for official W statistics.")
    parser.add_argument("--noise-weight", type=float, default=1e5, help="Official projector noise regularization.")
    parser.add_argument(
        "--include",
        action="append",
        help="Only project files whose stem contains this text. Can be passed multiple times.",
    )
    parser.add_argument("--max-images", type=int, help="Project at most this many images after filtering.")
    parser.add_argument("--save-reconstructions", action="store_true", help="Save reconstruction images.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    logging.info("Loading StyleGAN checkpoint from %s on %s", args.stylegan_checkpoint, args.device)
    generator = StyleGANGenerator.load(args.stylegan_checkpoint, device=args.device)
    output_dir = Path(args.output_dir)
    config = ProjectionConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        latent_layers=args.latent_layers,
        latent_channels=args.latent_channels,
        w_avg_samples=args.w_avg_samples,
        regularize_noise_weight=args.noise_weight,
    )

    image_paths = list_images(args.input_dir)
    if args.include:
        image_paths = [
            path
            for path in image_paths
            if any(fragment in path.stem for fragment in args.include)
        ]
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        image_paths = image_paths[: args.max_images]
    logging.info(
        "Projecting %d aligned images from %s with %s projector",
        len(image_paths),
        args.input_dir,
        args.projector,
    )
    for image_path in tqdm(image_paths, desc="project images", disable=args.quiet):
        logging.info("Projecting %s", image_path.name)
        image = load_rgb_image(image_path)
        if args.projector == "official":
            latent = generator.project_image_official(
                image,
                config=config,
                verbose=args.verbose,
            )
        else:
            latent = generator.project_image(
                image,
                config=config,
                progress=not args.quiet,
                description=f"opt {image_path.stem}",
            )
        latent_path = output_dir / f"{image_path.stem}.pt"
        save_w_plus(latent, latent_path)
        logging.info("Saved latent to %s", latent_path)
        if args.save_reconstructions:
            reconstruction = tensor_to_image(generator.synthesis(latent.to(generator.device)))
            reconstruction_path = ensure_parent_dir(output_dir / f"{image_path.stem}_projection.png")
            reconstruction.save(reconstruction_path)
            logging.info("Saved reconstruction to %s", reconstruction_path)
    logging.info("Projection finished. Outputs are in %s", output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
