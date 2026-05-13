from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from hw_4.src.config import load_config
from hw_4.src.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed-seed HW4 comparison samples.")
    parser.add_argument("--config", default="hw_4/configs/hw4.json")
    parser.add_argument("--model", required=True, help="Pipeline model path or Hugging Face model id.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lora-weights", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=30)
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
        "model": args.model,
        "lora_weights": args.lora_weights,
        "prompts": config.comparison_prompts,
        "seed": config.seed,
        "steps": args.steps,
    }
    (output_dir / "inference.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if args.dry_run:
        logging.info("Dry run: wrote inference metadata to %s", output_dir)
        return 0

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device)
    if args.lora_weights:
        pipe.load_lora_weights(args.lora_weights)
    generator = torch.Generator(device=args.device).manual_seed(config.seed)
    for index, prompt in enumerate(config.comparison_prompts, start=1):
        image = pipe(prompt, generator=generator, num_inference_steps=args.steps).images[0]
        image.save(output_dir / f"sample_{index:02d}.png")
    logging.info("Saved %d samples to %s", len(config.comparison_prompts), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
