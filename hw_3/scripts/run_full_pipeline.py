from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from hw_3.src.images import load_rgb_image, save_image
from hw_3.src.logging_utils import configure_logging


BASE_TARGET = "person_01_barack_obama"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full HW3 StyleGAN sampling pipeline.")
    parser.add_argument("--manifest", default="hw_3/image_manifest.json", help="Image manifest path.")
    parser.add_argument("--raw-dir", default="hw_3/data/raw", help="Raw downloaded image directory.")
    parser.add_argument("--landmarks-dir", default="hw_3/data/landmarks", help="Five-point landmarks directory.")
    parser.add_argument("--aligned-dir", default="hw_3/data/aligned", help="Aligned image output directory.")
    parser.add_argument("--artifacts-dir", default="hw_3/artifacts", help="Generated artifacts root.")
    parser.add_argument("--vendor-dir", default="hw_3/vendor/stylegan2-ada-pytorch", help="StyleGAN vendor dir.")
    parser.add_argument("--stylegan-checkpoint", required=True, help="StyleGAN checkpoint path.")
    parser.add_argument("--arcface-checkpoint", required=True, help="ArcFace checkpoint path.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example mps or cuda.")
    parser.add_argument("--projection-steps", type=int, default=1000, help="Projection optimization steps.")
    parser.add_argument("--face-swap-steps", type=int, default=300, help="Face swap optimization steps.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing raw images.")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True, help="Clean generated outputs first.")
    parser.add_argument("--quiet", action="store_true", help="Reduce child-script progress output.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError(f"Manifest must contain an images list: {path}")
    return images


def _by_role(images: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [image for image in images if image.get("role") == role]


def _run_module(module: str, args: list[str], *, env: dict[str, str]) -> None:
    command = [sys.executable, "-m", module, *args]
    logging.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True, env=env)


def _pipeline_env(vendor_dir: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    vendor = str(Path(vendor_dir))
    env["PYTHONPATH"] = vendor + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _clean_outputs(*paths: str | Path) -> None:
    for path in paths:
        resolved = Path(path)
        if resolved.exists():
            shutil.rmtree(resolved)


def _clean_artifacts_root(path: str | Path) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _image_path(directory: str | Path, entry: dict[str, Any]) -> Path:
    return Path(directory) / str(entry["filename"])


def _projection_path(projections_dir: Path, entry_or_id: dict[str, Any] | str) -> Path:
    image_id = entry_or_id if isinstance(entry_or_id, str) else str(entry_or_id["id"])
    return projections_dir / f"{image_id}_projection.png"


def _latent_path(projections_dir: Path, entry_or_id: dict[str, Any] | str) -> Path:
    image_id = entry_or_id if isinstance(entry_or_id, str) else str(entry_or_id["id"])
    return projections_dir / f"{image_id}.pt"


def _display_id(entry: dict[str, Any]) -> str:
    image_id = str(entry["id"])
    prefix, number, *_ = image_id.split("_")
    return f"{prefix[0]}{number}"


def _chunked(items: list[tuple[str, Path | None]], size: int) -> list[list[tuple[str, Path | None]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _labeled_grid(rows: list[list[tuple[str, Path | None]]], output: Path, *, cell: int = 180) -> None:
    label_h = 36
    pad = 8
    columns = max(len(row) for row in rows)
    canvas = Image.new(
        "RGB",
        (columns * cell + (columns - 1) * pad, len(rows) * (cell + label_h + pad)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        y = row_index * (cell + label_h + pad)
        for column_index, (label, path) in enumerate(row):
            x = column_index * (cell + pad)
            draw.text((x + 4, y + 6), label[:32], fill=(0, 0, 0))
            if path is None:
                continue
            image = load_rgb_image(path).resize((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y + label_h))
    save_image(canvas, output)


def _build_final_grids(
    *,
    images: list[dict[str, Any]],
    base_people: list[dict[str, Any]],
    style_images: list[dict[str, Any]],
    expression_images: list[dict[str, Any]],
    aligned_dir: Path,
    projections_dir: Path,
    style_dir: Path,
    expression_dir: Path,
    face_swap_dir: Path,
    final_dir: Path,
) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)

    aligned_cells = [(_display_id(image), _image_path(aligned_dir, image)) for image in images]
    _labeled_grid(
        _chunked(aligned_cells, 4),
        final_dir / "aligned_grid.png",
    )

    projection_rows = [
        [
            (f"aligned {_display_id(image)}", _image_path(aligned_dir, image)),
            (f"projection {_display_id(image)}", _projection_path(projections_dir, image)),
        ]
        for image in images
    ]
    _labeled_grid(projection_rows, final_dir / "projection_comparison.png")

    style_rows = [
        [
            ("target projection", _projection_path(projections_dir, BASE_TARGET)),
            (f"style {_display_id(style)}", _projection_path(projections_dir, style)),
            (
                "mixed result",
                style_dir / f"{BASE_TARGET}_{style['id']}.png",
            ),
        ]
        for style in style_images
    ]
    _labeled_grid(style_rows, final_dir / "style_transfer_grid.png")

    expression_rows = [
        [
            ("target projection", _projection_path(projections_dir, BASE_TARGET)),
            (f"expression {_display_id(expression)}", _projection_path(projections_dir, expression)),
            (
                "mixed result",
                expression_dir / f"{BASE_TARGET}_{expression['id']}.png",
            ),
        ]
        for expression in expression_images
    ]
    _labeled_grid(expression_rows, final_dir / "expression_transfer_grid.png")

    face_rows: list[list[tuple[str, Path | None]]] = []
    for source in base_people:
        row = []
        for target in base_people:
            if source["id"] == target["id"]:
                row.append((f"{_display_id(source)} self", None))
            else:
                row.append(
                    (
                        f"{_display_id(source)} -> {_display_id(target)}",
                        face_swap_dir / f"{source['id']}_to_{target['id']}.png",
                    )
                )
        face_rows.append(row)
    _labeled_grid(face_rows, final_dir / "face_swap_matrix.png", cell=150)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    images = _load_manifest(args.manifest)
    base_people = _by_role(images, "base")
    style_images = _by_role(images, "style")
    expression_images = _by_role(images, "expression")
    artifacts_dir = Path(args.artifacts_dir)
    projections_dir = artifacts_dir / "projections"
    style_dir = artifacts_dir / "style_transfer"
    expression_dir = artifacts_dir / "expression_transfer"
    face_swap_dir = artifacts_dir / "face_swap"
    final_dir = artifacts_dir / "final"
    env = _pipeline_env(args.vendor_dir)
    quiet_flag = ["--quiet"] if args.quiet else []

    if args.clean:
        _clean_outputs(args.aligned_dir)
        _clean_artifacts_root(artifacts_dir)

    if not args.skip_download:
        _run_module(
            "hw_3.scripts.download_image_set",
            [
                "--manifest",
                args.manifest,
                "--output-dir",
                args.raw_dir,
                "--skip-existing",
                "--retries",
                "5",
                "--delay-seconds",
                "5",
            ],
            env=env,
        )

    _run_module(
        "hw_3.scripts.align_images",
        [
            "--input-dir",
            args.raw_dir,
            "--landmarks-dir",
            args.landmarks_dir,
            "--output-dir",
            args.aligned_dir,
            "--mode",
            "landmarks",
            "--image-size",
            "1024",
            *quiet_flag,
        ],
        env=env,
    )

    _run_module(
        "hw_3.scripts.project_images",
        [
            "--input-dir",
            args.aligned_dir,
            "--output-dir",
            str(projections_dir),
            "--stylegan-checkpoint",
            args.stylegan_checkpoint,
            "--device",
            args.device,
            "--steps",
            str(args.projection_steps),
            "--learning-rate",
            "0.1",
            "--image-size",
            "1024",
            "--projector",
            "official",
            "--w-avg-samples",
            "10000",
            "--noise-weight",
            "100000",
            "--save-reconstructions",
            *quiet_flag,
        ],
        env=env,
    )

    for style in style_images:
        _run_module(
            "hw_3.scripts.run_style_transfer",
            [
                "--source",
                str(_latent_path(projections_dir, style)),
                "--target",
                str(_latent_path(projections_dir, BASE_TARGET)),
                "--output",
                str(style_dir / f"{BASE_TARGET}_{style['id']}.pt"),
                "--layers",
                "7-17",
                "--stylegan-checkpoint",
                args.stylegan_checkpoint,
                "--output-image",
                str(style_dir / f"{BASE_TARGET}_{style['id']}.png"),
                "--device",
                args.device,
            ],
            env=env,
        )

    for expression in expression_images:
        _run_module(
            "hw_3.scripts.run_expression_transfer",
            [
                "--source",
                str(_latent_path(projections_dir, expression)),
                "--target",
                str(_latent_path(projections_dir, BASE_TARGET)),
                "--output",
                str(expression_dir / f"{BASE_TARGET}_{expression['id']}.pt"),
                "--layers",
                "4-6",
                "--stylegan-checkpoint",
                args.stylegan_checkpoint,
                "--output-image",
                str(expression_dir / f"{BASE_TARGET}_{expression['id']}.png"),
                "--device",
                args.device,
            ],
            env=env,
        )

    for source in base_people:
        for target in base_people:
            if source["id"] == target["id"]:
                continue
            _run_module(
                "hw_3.scripts.run_face_swap",
                [
                    "--source-image",
                    str(_image_path(args.aligned_dir, source)),
                    "--target-latent",
                    str(_latent_path(projections_dir, target)),
                    "--output-latent",
                    str(face_swap_dir / f"{source['id']}_to_{target['id']}.pt"),
                    "--output-image",
                    str(face_swap_dir / f"{source['id']}_to_{target['id']}.png"),
                    "--stylegan-checkpoint",
                    args.stylegan_checkpoint,
                    "--arcface-checkpoint",
                    args.arcface_checkpoint,
                    "--device",
                    args.device,
                    "--steps",
                    str(args.face_swap_steps),
                    "--learning-rate",
                    "0.005",
                    "--identity-weight",
                    "1.0",
                    "--latent-weight",
                    "0.1",
                    "--layers",
                    "4-10",
                    *quiet_flag,
                ],
                env=env,
            )

    _build_final_grids(
        images=images,
        base_people=base_people,
        style_images=style_images,
        expression_images=expression_images,
        aligned_dir=Path(args.aligned_dir),
        projections_dir=projections_dir,
        style_dir=style_dir,
        expression_dir=expression_dir,
        face_swap_dir=face_swap_dir,
        final_dir=final_dir,
    )
    logging.info("Full pipeline finished. Final PNG grids are in %s", final_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
