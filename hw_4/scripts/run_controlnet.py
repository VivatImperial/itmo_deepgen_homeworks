from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image, ImageFilter

from hw_4.src.config import load_config
from hw_4.src.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Canny-like ControlNet samples for HW4.")
    parser.add_argument("--config", default="hw_4/configs/hw4.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--control-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lora-weights", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _edge_image(path: str | Path, size: int) -> Image.Image:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return image.filter(ImageFilter.FIND_EDGES).convert("RGB")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    control = _edge_image(args.control_image, config.resolution)
    control.save(output_dir / "control_edges.png")
    metadata = {
        "model": args.model,
        "controlnet_model": config.controlnet_model,
        "lora_weights": args.lora_weights,
        "control_image": args.control_image,
        "prompt": config.comparison_prompts[0],
        "seed": config.seed,
    }
    (output_dir / "controlnet.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if args.dry_run:
        logging.info("Dry run: wrote ControlNet metadata to %s", output_dir)
        return 0

    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    controlnet = ControlNetModel.from_pretrained(config.controlnet_model, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to(args.device)
    if args.lora_weights:
        pipe.load_lora_weights(args.lora_weights)
    generator = torch.Generator(device=args.device).manual_seed(config.seed)
    image = pipe(config.comparison_prompts[0], image=control, generator=generator, num_inference_steps=args.steps).images[0]
    image.save(output_dir / "controlnet_sample.png")
    logging.info("Saved ControlNet sample to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
