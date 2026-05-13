from pathlib import Path

import pytest

from hw_3.src.config import LayerRange, ensure_dir, ensure_file, ensure_parent_dir


def test_ensure_file_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing file"):
        ensure_file(tmp_path / "missing.pkl", "StyleGAN checkpoint")


def test_ensure_dir_rejects_file(tmp_path: Path) -> None:
    file_path = tmp_path / "input.jpg"
    file_path.write_text("not a directory")

    with pytest.raises(NotADirectoryError, match="Expected directory"):
        ensure_dir(file_path, "raw image directory")


def test_ensure_parent_dir_creates_parent(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "image.png"

    ensure_parent_dir(output_path)

    assert output_path.parent.is_dir()


def test_layer_range_expands_inclusive_indices() -> None:
    assert LayerRange(4, 7).indices() == [4, 5, 6, 7]


def test_layer_range_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="start"):
        LayerRange(8, 3)
