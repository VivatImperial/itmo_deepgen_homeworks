from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from hw_4.src.config import load_config
from hw_4.src.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate class images for DreamBooth prior preservation.")
    parser.add_argument("--config", default="hw_4/configs/hw4.json")
    parser.add_argument("--output-dir", default="hw_4/data/class")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_model": config.base_model,
        "class_prompt": config.class_prompt,
        "count": args.count,
        "seed": config.seed,
    }
    (output_dir / "class_images.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if args.dry_run:
        logging.info("Dry run: wrote class image metadata to %s", output_dir)
        return 0

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(config.base_model, torch_dtype=torch.float16).to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(config.seed)
    for index in range(args.count):
        image = pipe(config.class_prompt, generator=generator, num_inference_steps=30).images[0]
        image.save(output_dir / f"class_{index:04d}.jpg", quality=95)
    logging.info("Generated %d class images in %s", args.count, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
