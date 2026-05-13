from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hw_4.src.dataset import prepare_dataset
from hw_4.src.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare HW4 DreamBooth instance images.")
    parser.add_argument("--manifest", default="hw_4/manifests/subject_images.json")
    parser.add_argument("--output-dir", default="hw_4/data/processed")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    records = prepare_dataset(Path(args.manifest), Path(args.output_dir), image_size=args.image_size)
    logging.info("Prepared %d images in %s", len(records), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
