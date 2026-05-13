from __future__ import annotations

import argparse
import logging

from hw_3.scripts._common import parse_layer_range
from hw_3.src.config import ensure_parent_dir
from hw_3.src.latents import load_w_plus, save_w_plus
from hw_3.src.logging_utils import configure_logging
from hw_3.src.stylegan import StyleGANGenerator, tensor_to_image
from hw_3.src.transfers import style_transfer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mix W+ layers for StyleGAN style transfer.")
    parser.add_argument("--source", required=True, help="Style source W+ latent.")
    parser.add_argument("--target", required=True, help="Identity target W+ latent.")
    parser.add_argument("--output", required=True, help="Output mixed W+ latent.")
    parser.add_argument("--layers", default="7-17", help="Inclusive layer range, for example 7-17.")
    parser.add_argument("--stylegan-checkpoint", help="Optional StyleGAN checkpoint for image rendering.")
    parser.add_argument("--output-image", help="Optional rendered image path.")
    parser.add_argument("--device", default="cpu", help="Torch device for optional rendering.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    layer_range = parse_layer_range(args.layers)
    logging.info("Running style transfer with W+ layers %s", args.layers)
    source = load_w_plus(args.source)
    target = load_w_plus(args.target)
    mixed = style_transfer(source, target, layer_range)
    save_w_plus(mixed, args.output)
    logging.info("Saved mixed latent to %s", args.output)
    if args.output_image:
        if not args.stylegan_checkpoint:
            raise ValueError("--stylegan-checkpoint is required when --output-image is used")
        logging.info("Rendering style transfer image on %s", args.device)
        generator = StyleGANGenerator.load(args.stylegan_checkpoint, device=args.device)
        image = tensor_to_image(generator.synthesis(mixed.to(generator.device)))
        image.save(ensure_parent_dir(args.output_image))
        logging.info("Saved rendered image to %s", args.output_image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
