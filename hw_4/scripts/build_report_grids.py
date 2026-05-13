from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hw_4.src.logging_utils import configure_logging
from hw_4.src.report import build_labeled_grid, write_metrics_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final HW4 report grids from generated samples.")
    parser.add_argument("--samples-root", default="hw_4/artifacts/samples")
    parser.add_argument("--output-dir", default="hw_4/artifacts/final")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _sample(path: Path, name: str) -> Path:
    sample = path / name / "sample_01.png"
    if not sample.exists():
        raise FileNotFoundError(f"Missing sample image: {sample}")
    return sample


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    samples_root = Path(args.samples_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        [
            ("baseline", _sample(samples_root, "baseline")),
            ("unet", _sample(samples_root, "unet")),
            ("lora r4", _sample(samples_root, "lora_rank_4")),
            ("lora r8", _sample(samples_root, "lora_rank_8")),
            ("lora r16", _sample(samples_root, "lora_rank_16")),
        ]
    ]
    build_labeled_grid(rows, output_dir / "comparison_grid.png")
    summary = {"samples_root": str(samples_root), "grid": "comparison_grid.png"}
    write_metrics_summary(output_dir / "metrics_summary.json", summary)
    (output_dir / "report_artifacts.md").write_text(
        "# HW4 Final Artifacts\n\n- `comparison_grid.png`\n- `metrics_summary.json`\n",
        encoding="utf-8",
    )
    logging.info("Saved report grid and summary to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
