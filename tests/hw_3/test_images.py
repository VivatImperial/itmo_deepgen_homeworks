from pathlib import Path

import pytest
from PIL import Image

from hw_3.src.images import (
    FaceLandmarks,
    _similarity_transform_matrix,
    align_face,
    center_crop_square,
    list_images,
    load_landmarks,
    make_image_grid,
    resize_image,
    save_grid,
)


def _write_image(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    image = Image.new("RGB", size, color=(120, 20, 40))
    image.save(path)


def test_list_images_returns_supported_files_sorted(tmp_path: Path) -> None:
    _write_image(tmp_path / "b.png")
    _write_image(tmp_path / "a.jpg")
    _write_image(tmp_path / "c.webp")
    (tmp_path / "notes.txt").write_text("ignore me")

    names = [path.name for path in list_images(tmp_path)]

    assert names == ["a.jpg", "b.png", "c.webp"]


def test_center_crop_square_uses_short_side() -> None:
    image = Image.new("RGB", (10, 6), color="white")

    cropped = center_crop_square(image)

    assert cropped.size == (6, 6)


def test_resize_image_returns_requested_size() -> None:
    image = Image.new("RGB", (12, 6), color="white")

    resized = resize_image(image, 16)

    assert resized.size == (16, 16)


def test_make_image_grid_uses_row_and_column_count() -> None:
    images = [Image.new("RGB", (4, 4), color="white") for _ in range(6)]

    grid = make_image_grid(images, columns=3, cell_size=8, padding=2)

    assert grid.size == (28, 18)


def test_save_grid_writes_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "tables" / "grid.png"
    images = [Image.new("RGB", (4, 4), color="white")]

    save_grid(images, output_path, columns=1, cell_size=8)

    assert output_path.is_file()


def test_load_landmarks_reads_five_point_json(tmp_path: Path) -> None:
    path = tmp_path / "face.json"
    path.write_text(
        """
        {
          "left_eye": [10, 20],
          "right_eye": [30, 20],
          "nose": [20, 32],
          "left_mouth": [12, 44],
          "right_mouth": [28, 44]
        }
        """
    )

    landmarks = load_landmarks(path)

    assert landmarks.left_eye == (10.0, 20.0)
    assert landmarks.right_mouth == (28.0, 44.0)


def test_align_face_returns_requested_size() -> None:
    image = Image.new("RGB", (80, 80), color="white")
    landmarks = FaceLandmarks(
        left_eye=(25, 25),
        right_eye=(55, 25),
        nose=(40, 40),
        left_mouth=(30, 55),
        right_mouth=(50, 55),
    )

    aligned = align_face(image, landmarks, output_size=32)

    assert aligned.size == (32, 32)


def test_similarity_transform_does_not_anisotropically_stretch_face() -> None:
    landmarks = FaceLandmarks(
        left_eye=(250, 385),
        right_eye=(434, 386),
        nose=(338, 548),
        left_mouth=(248, 704),
        right_mouth=(421, 700),
    )

    matrix = _similarity_transform_matrix(landmarks, output_size=1024)
    a, b, _, d, e, _ = matrix
    first_axis = (a**2 + d**2) ** 0.5
    second_axis = (b**2 + e**2) ** 0.5

    assert first_axis == pytest.approx(second_axis)
