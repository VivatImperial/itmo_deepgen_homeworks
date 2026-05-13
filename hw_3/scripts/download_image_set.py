from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hw_3.src.config import ensure_file, ensure_parent_dir
from hw_3.src.logging_utils import configure_logging


@dataclass(frozen=True)
class ImageEntry:
    id: str
    role: str
    pose: str
    url: str
    filename: str
    source: str


@dataclass(frozen=True)
class ImageManifest:
    images: list[ImageEntry]


def load_manifest(path: str | Path) -> ImageManifest:
    manifest_path = ensure_file(path, "image manifest")
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list) or not images:
        raise ValueError("Image manifest must contain at least one image")

    entries: list[ImageEntry] = []
    for item in images:
        if not isinstance(item, dict):
            raise ValueError("Each image manifest entry must be an object")
        entries.append(
            ImageEntry(
                id=str(item["id"]),
                role=str(item["role"]),
                pose=str(item["pose"]),
                url=str(item["url"]),
                filename=str(item["filename"]),
                source=str(item["source"]),
            )
        )
    return ImageManifest(images=entries)


def download_file(
    url: str,
    output_path: str | Path,
    *,
    retries: int = 3,
    delay_seconds: float = 2.0,
) -> Path:
    destination = ensure_parent_dir(output_path)
    request = Request(url, headers={"User-Agent": "itmo-homework/1.0"})
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())
            return destination
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay_seconds * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}") from last_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the public HW3 portrait image set.")
    parser.add_argument("--manifest", default="hw_3/image_manifest.json", help="Path to image manifest.")
    parser.add_argument("--output-dir", default="hw_3/data/raw", help="Directory for downloaded images.")
    parser.add_argument("--role", choices=("all", "base", "style", "expression"), default="all")
    parser.add_argument("--skip-existing", action="store_true", help="Do not redownload existing files.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per image after transient errors.")
    parser.add_argument("--delay-seconds", type=float, default=2.0, help="Base delay between retries.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir)
    selected = [
        image
        for image in manifest.images
        if args.role == "all" or image.role == args.role
    ]
    logging.info("Downloading %d images to %s", len(selected), output_dir)

    for image in selected:
        output_path = output_dir / image.filename
        if args.skip_existing and output_path.exists():
            logging.info("Skipping existing %s", output_path)
            continue
        logging.info("Downloading %s from %s", image.filename, image.url)
        download_file(
            image.url,
            output_path,
            retries=args.retries,
            delay_seconds=args.delay_seconds,
        )
        logging.info("Saved %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
