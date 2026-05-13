from __future__ import annotations

import argparse

from hw_3.src.config import LayerRange


def parse_layer_range(value: str) -> LayerRange:
    try:
        start_text, end_text = value.split("-", maxsplit=1)
        return LayerRange(int(start_text), int(end_text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Layer range must use START-END format") from exc
