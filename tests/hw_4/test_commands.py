from hw_4.src.commands import (
    DreamBoothRun,
    LoRARun,
    build_dreambooth_command,
    build_lora_command,
)


def test_build_dreambooth_command_uses_diffusers_script_and_prompts():
    run = DreamBoothRun(
        diffusers_dir="vendor/diffusers",
        pretrained_model="runwayml/stable-diffusion-v1-5",
        instance_dir="data/processed/instance",
        class_dir="data/processed/class",
        output_dir="artifacts/unet",
        instance_prompt="a photo of sks person face",
        class_prompt="a photo of a person face",
        resolution=512,
        train_steps=800,
        learning_rate=1e-6,
        seed=42,
    )

    command = build_dreambooth_command(run)

    assert command[:2] == ["accelerate", "launch"]
    assert "examples/dreambooth/train_dreambooth.py" in command[2]
    assert "--instance_prompt=a photo of sks person face" in command
    assert "--with_prior_preservation" in command


def test_build_lora_command_sets_rank_and_output_dir():
    run = LoRARun(
        diffusers_dir="vendor/diffusers",
        pretrained_model="runwayml/stable-diffusion-v1-5",
        instance_dir="data/processed/instance",
        class_dir="data/processed/class",
        output_dir="artifacts/lora_rank_8",
        instance_prompt="a photo of sks person face",
        class_prompt="a photo of a person face",
        resolution=512,
        train_steps=800,
        learning_rate=1e-4,
        seed=42,
        rank=8,
    )

    command = build_lora_command(run)

    assert "examples/dreambooth/train_dreambooth_lora.py" in command[2]
    assert "--rank=8" in command
    assert "--output_dir=artifacts/lora_rank_8" in command
