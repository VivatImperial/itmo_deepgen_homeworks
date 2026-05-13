import json
from pathlib import Path

from hw_3.scripts.download_image_set import load_manifest


def test_load_manifest_rejects_empty_image_set(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"images": []}))

    try:
        load_manifest(path)
    except ValueError as exc:
        assert "at least one image" in str(exc)
    else:
        raise AssertionError("Expected manifest validation to fail")


def test_load_manifest_reads_image_entries(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": "person_01",
                        "role": "base",
                        "pose": "frontal",
                        "url": "https://example.com/person.jpg",
                        "filename": "person_01.jpg",
                        "source": "Example",
                    }
                ]
            }
        )
    )

    manifest = load_manifest(path)

    assert manifest.images[0].id == "person_01"
    assert manifest.images[0].filename == "person_01.jpg"
