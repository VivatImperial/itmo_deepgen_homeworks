import json

from hw_4.src.config import load_config


def test_load_config_parses_training_protocol(tmp_path):
    config_path = tmp_path / "hw4.json"
    config_path.write_text(
        json.dumps(
            {
                "base_model": "runwayml/stable-diffusion-v1-5",
                "controlnet_model": "lllyasviel/sd-controlnet-canny",
                "subject_token": "sks",
                "class_name": "person",
                "instance_prompt": "a photo of sks person face",
                "class_prompt": "a photo of a person face",
                "resolution": 512,
                "seed": 42,
                "lora_ranks": [4, 8, 16],
                "comparison_prompts": [
                    "a photo of sks person face in a city",
                    "a photo of sks person face on a beach",
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.instance_prompt == "a photo of sks person face"
    assert config.class_prompt == "a photo of a person face"
    assert config.lora_ranks == [4, 8, 16]
    assert len(config.comparison_prompts) == 2
