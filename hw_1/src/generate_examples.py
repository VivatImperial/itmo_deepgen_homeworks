from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import ImageFilter

from hw_1.src.avatar_generator import AvatarGenerator
from hw_1.src.style_generator import StyleGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "generated",
    )
    return parser


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_markdown(path: Path, styles_payload: list[dict[str, Any]], avatars_payload: list[dict[str, Any]]) -> None:
    lines = [
        "# Generated Examples",
        "",
        "## Styles",
        "",
    ]

    for index, sample in enumerate(styles_payload, start=1):
        lines.append(f"### Style {index}")
        lines.append(f"- Bayesian probability: `{sample['bayesian_probability_scientific']}`")
        lines.append(f"- MLE probability: `{sample['mle_probability_scientific']}`")
        for category, value in sample["sampled_style"].items():
            lines.append(f"- {category}: {value}")
        lines.append("")

    lines.extend(["## Avatars", ""])

    for index, sample in enumerate(avatars_payload, start=1):
        lines.append(f"### Avatar {index}")
        lines.append(f"- File: `avatar_{index}.png`")
        lines.append(f"- Bayesian probability: `{sample['bayesian_probability_scientific']}`")
        lines.append(f"- MLE probability: `{sample['mle_probability_scientific']}`")
        lines.append(f"- Opaque ratio: `{sample['opaque_ratio']:.4f}`")
        lines.append(f"- Average channel support count: `{sample['average_channel_support_count']:.4f}`")
        lines.append(f"- Global opaque prior: `{sample['global_opaque_prior']:.4f}`")
        lines.append(f"- Visualization postprocess: `{sample['visual_postprocess']}`")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    avatar_dir = repo_root / "hw_1" / "avatars"

    style_generator = StyleGenerator(seed=args.seed)
    style_samples = style_generator.sample_many(args.count)

    avatar_generator = AvatarGenerator(seed=args.seed)
    avatar_generator.fit(avatar_dir)
    avatar_samples = []

    for index, (image, metadata) in enumerate(avatar_generator.sample_many(args.count), start=1):
        image_path = output_dir / f"avatar_{index}.png"
        processed_image = image.filter(ImageFilter.MedianFilter(size=5))
        processed_image.save(image_path)
        sample_metadata = {
            "file": image_path.name,
            "visual_postprocess": "median_filter_size_5",
            **metadata,
        }
        avatar_samples.append(sample_metadata)

    save_json(
        output_dir / "styles.json",
        {
            "seed": args.seed,
            "count": args.count,
            "samples": style_samples,
        },
    )
    save_json(
        output_dir / "avatars.json",
        {
            "seed": args.seed,
            "count": args.count,
            "samples": avatar_samples,
        },
    )
    save_markdown(output_dir / "report.md", style_samples, avatar_samples)


if __name__ == "__main__":
    main()
