import json

from PIL import Image

from hw_4.src.dataset import prepare_dataset


def test_prepare_dataset_crops_resizes_and_writes_captions(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (120, 140), (120, 80, 40)).save(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "subject": "Ada Lovelace",
                "images": [
                    {
                        "id": "ada_01",
                        "url": str(source),
                        "source": "local smoke image",
                        "caption": "a photo of sks person face",
                        "crop_box": [10, 20, 110, 120],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "prepared"

    records = prepare_dataset(manifest, output_dir, image_size=64)

    image_path = output_dir / "instance" / "ada_01.jpg"
    caption_path = output_dir / "instance" / "ada_01.txt"
    metadata_path = output_dir / "metadata.json"
    assert records[0].image_path == image_path
    assert Image.open(image_path).size == (64, 64)
    assert caption_path.read_text(encoding="utf-8") == "a photo of sks person face\n"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["image_count"] == 1
