from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hw_4.src.commands import DreamBoothRun, build_dreambooth_command, run_command
from hw_4.src.config import load_config
from hw_4.src.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full UNet DreamBooth training through diffusers.")
    parser.add_argument("--config", default="hw_4/configs/hw4.json")
    parser.add_argument("--diffusers-dir", default="hw_4/vendor/diffusers")
    parser.add_argument("--instance-dir", default="hw_4/data/processed/instance")
    parser.add_argument("--class-dir", default="hw_4/data/class")
    parser.add_argument("--output-dir", default="hw_4/artifacts/unet")
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    config = load_config(args.config)
    run = DreamBoothRun(
        diffusers_dir=args.diffusers_dir,
        pretrained_model=config.base_model,
        instance_dir=args.instance_dir,
        class_dir=args.class_dir,
        output_dir=args.output_dir,
        instance_prompt=config.instance_prompt,
        class_prompt=config.class_prompt,
        resolution=config.resolution,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        seed=config.seed,
    )
    command = build_dreambooth_command(run)
    logging.info("DreamBooth UNet command: %s", " ".join(command))
    run_command(command, log_path=Path(args.output_dir) / "run.log", dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
