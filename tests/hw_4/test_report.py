from PIL import Image

from hw_4.src.report import build_labeled_grid


def test_build_labeled_grid_creates_report_image(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (32, 32), "red").save(first)
    Image.new("RGB", (32, 32), "blue").save(second)
    output = tmp_path / "grid.png"

    build_labeled_grid(
        rows=[
            [("baseline", first), ("lora rank 8", second)],
        ],
        output=output,
        cell_size=64,
    )

    assert output.exists()
    assert Image.open(output).size[0] > 128
