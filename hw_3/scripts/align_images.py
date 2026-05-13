from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm.auto import tqdm

from hw_3.src.logging_utils import configure_logging
from hw_3.src.images import (
    align_face,
    center_crop_square,
    list_images,
    load_landmarks,
    load_rgb_image,
    resize_image,
    save_image,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align homework portraits for StyleGAN projection.")
    parser.add_argument("--input-dir", required=True, help="Directory with raw portraits.")
    parser.add_argument("--output-dir", required=True, help="Directory for aligned portraits.")
    parser.add_argument(
        "--landmarks-dir",
        help="Directory with five-point JSON landmarks matching each image stem.",
    )
    parser.add_argument(
        "--mode",
        choices=("landmarks", "center-crop"),
        default="landmarks",
        help="Alignment mode. Use center-crop only as a rough fallback.",
    )
    parser.add_argument("--image-size", type=int, default=1024, help="Output square image size.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    output_dir = Path(args.output_dir)
    image_paths = list_images(args.input_dir)
    logging.info("Aligning %d images from %s to %s", len(image_paths), args.input_dir, output_dir)

    for image_path in tqdm(image_paths, desc="align images", disable=args.quiet):
        logging.debug("Aligning %s", image_path.name)
        image = load_rgb_image(image_path)
        if args.mode == "landmarks":
            if not args.landmarks_dir:
                raise ValueError("--landmarks-dir is required when --mode landmarks")
            landmarks = load_landmarks(Path(args.landmarks_dir) / f"{image_path.stem}.json")
            aligned = align_face(image, landmarks, output_size=args.image_size)
        else:
            aligned = resize_image(center_crop_square(image), args.image_size)
        save_image(aligned, output_dir / image_path.name)
    logging.info("Saved aligned images to %s", output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
