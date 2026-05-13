from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LayerRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")

    def indices(self) -> list[int]:
        return list(range(self.start, self.end + 1))


STYLE_LAYER_RANGE = LayerRange(7, 17)
EXPRESSION_LAYER_RANGE = LayerRange(4, 6)
COARSE_POSE_LAYER_RANGE = LayerRange(0, 3)


def ensure_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing file for {label}: {resolved}")
    return resolved


def ensure_dir(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_dir():
        raise NotADirectoryError(f"Expected directory for {label}: {resolved}")
    return resolved


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
