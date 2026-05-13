from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DreamBoothRun:
    diffusers_dir: str
    pretrained_model: str
    instance_dir: str
    class_dir: str
    output_dir: str
    instance_prompt: str
    class_prompt: str
    resolution: int
    train_steps: int
    learning_rate: float
    seed: int
    mixed_precision: str = "fp16"


@dataclass(frozen=True)
class LoRARun(DreamBoothRun):
    rank: int = 8


def _script_path(diffusers_dir: str, script_name: str) -> str:
    return str(Path(diffusers_dir) / "examples" / "dreambooth" / script_name)


def build_dreambooth_command(run: DreamBoothRun) -> list[str]:
    return [
        "accelerate",
        "launch",
        _script_path(run.diffusers_dir, "train_dreambooth.py"),
        f"--pretrained_model_name_or_path={run.pretrained_model}",
        f"--instance_data_dir={run.instance_dir}",
        f"--class_data_dir={run.class_dir}",
        f"--output_dir={run.output_dir}",
        f"--instance_prompt={run.instance_prompt}",
        f"--class_prompt={run.class_prompt}",
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        f"--resolution={run.resolution}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        "--gradient_checkpointing",
        f"--learning_rate={run.learning_rate}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--max_train_steps={run.train_steps}",
        f"--seed={run.seed}",
        f"--mixed_precision={run.mixed_precision}",
    ]


def build_lora_command(run: LoRARun) -> list[str]:
    return [
        "accelerate",
        "launch",
        _script_path(run.diffusers_dir, "train_dreambooth_lora.py"),
        f"--pretrained_model_name_or_path={run.pretrained_model}",
        f"--instance_data_dir={run.instance_dir}",
        f"--class_data_dir={run.class_dir}",
        f"--output_dir={run.output_dir}",
        f"--instance_prompt={run.instance_prompt}",
        f"--class_prompt={run.class_prompt}",
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        f"--resolution={run.resolution}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        "--gradient_checkpointing",
        f"--learning_rate={run.learning_rate}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--max_train_steps={run.train_steps}",
        f"--seed={run.seed}",
        f"--mixed_precision={run.mixed_precision}",
        f"--rank={run.rank}",
    ]


def run_command(command: list[str], *, log_path: str | Path, dry_run: bool) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("$ " + " ".join(command) + "\n", encoding="utf-8")
    if dry_run:
        return
    with path.open("a", encoding="utf-8") as log_file:
        subprocess.run(command, check=True, stdout=log_file, stderr=subprocess.STDOUT)
