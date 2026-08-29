"""Print staged server commands for Qwen3-VL LoRA-SFT v2."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen_lora_sft_v2.json")
    parser.add_argument(
        "--stage",
        choices=["all", "access", "prepare", "split", "clip-audit", "smoke", "train"],
        default="all",
    )
    return parser.parse_args()


def q(value: object) -> str:
    text = str(value)
    if text.startswith("~/"):
        return "$HOME/" + text[2:]
    return shlex.quote(text)


def prefix(config: dict[str, Any]) -> str:
    return config.get(
        "command_prefix",
        "source ~/tools/miniconda3/etc/profile.d/conda.sh && "
        "conda activate orena-focus && cd ~/workspace/VLM-Competition",
    )


def dataset_args(config: dict[str, Any]) -> str:
    return " ".join(f"--dataset {q(dataset)}" for dataset in config["datasets"])


def print_block(title: str, command: str) -> None:
    print(f"\n## {title}\n")
    print("```bash")
    print(command)
    print("```")


def access_command(config: dict[str, Any]) -> str:
    access_output = config.get(
        "access_output",
        "~/workspace/focus-runs/data-audit/qwen-lora-sft-v2-access-check.json",
    )
    return (
        f"{prefix(config)} && python scripts/check_focus_dataset_access.py "
        f"--root-dir {q(config['root_dir'])} {dataset_args(config)} --track {q(config['track'])} "
        f"--json-output {q(access_output)}"
    )


def prepare_commands(config: dict[str, Any]) -> list[str]:
    commands = []
    for dataset in config["datasets"]:
        commands.append(
            f"{prefix(config)} && python scripts/prepare_focus_data.py "
            f"--root-dir {q(config['root_dir'])} --dataset {q(dataset)} --skip-frames"
        )
    return commands


def split_command(config: dict[str, Any]) -> str:
    return (
        f"{prefix(config)} && python scripts/audit_and_split_segment_train.py "
        f"--root-dir {q(config['root_dir'])} {dataset_args(config)} "
        f"--val-fraction {q(config['val_fraction'])} --seed {q(config['seed'])} "
        f"--output-dir {q(config['data_audit_dir'])}"
    )


def clip_audit_command(config: dict[str, Any]) -> str:
    return (
        f"{prefix(config)} && python scripts/audit_sft_clip_windows.py "
        f"--input-jsonl {q(config['data_audit_dir'] + '/sft_train_overlay.jsonl')} "
        f"--input-jsonl {q(config['data_audit_dir'] + '/sft_val_overlay.jsonl')} "
        f"--output-dir {q(config['clip_audit_dir'])}"
    )


def train_command(config: dict[str, Any], smoke: bool) -> str:
    output_dir = config["smoke_output_dir"] if smoke else config["train_output_dir"]
    max_train = 128 if smoke else config["max_train_samples"]
    max_val = 32 if smoke else config["max_val_samples"]
    epochs = 1 if smoke else config["epochs"]
    nproc = config.get("smoke_nproc_per_node", 1) if smoke else config.get("nproc_per_node", 1)
    training_script = config.get(
        "training_script", "scripts/train_qwen3vl_lora_sft_smoke.py"
    )
    if nproc > 1:
        launcher = f"torchrun --standalone --nproc-per-node {q(nproc)}"
    else:
        launcher = "CUDA_VISIBLE_DEVICES=0 python"
    return (
        f"{prefix(config)} && {launcher} {q(training_script)} "
        f"--model-id {q(config['model_id'])} "
        f"--train-jsonl {q(config['train_jsonl'])} "
        f"--val-jsonl {q(config['val_jsonl'])} "
        f"--output-dir {q(output_dir)} "
        f"--max-train-samples {q(max_train)} --max-val-samples {q(max_val)} "
        f"--epochs {q(epochs)} --learning-rate {q(config['learning_rate'])} "
        f"--lora-r {q(config['lora_r'])} --lora-alpha {q(config['lora_alpha'])} "
        f"--lora-dropout {q(config['lora_dropout'])} "
        f"--video-stride {q(config['video_stride'])} "
        f"--video-min-frames {q(config['video_min_frames'])} "
        f"--video-max-frames {q(config['video_max_frames'])} "
        f"--width {q(config['width'])} --height {q(config['height'])} "
        f"--gradient-accumulation-steps {q(config['gradient_accumulation_steps'])}"
        + (
            f" --attn-implementation {q(config['attn_implementation'])}"
            if "attn_implementation" in config
            else ""
        )
    )


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    if args.stage in {"all", "access"}:
        print_block("1. Check Dataset Access", access_command(config))
    if args.stage in {"all", "prepare"}:
        for dataset, command in zip(config["datasets"], prepare_commands(config)):
            print_block(f"2. Prepare {dataset}", command)
    if args.stage in {"all", "split"}:
        print_block("3. Build Combined SFT Split", split_command(config))
    if args.stage in {"all", "clip-audit"}:
        print_block("4. Audit Clip Windows", clip_audit_command(config))
    if args.stage in {"all", "smoke"}:
        print_block("5. Smoke Train 128 Rows", train_command(config, smoke=True))
    if args.stage in {"all", "train"}:
        print_block("6. Full Train", train_command(config, smoke=False))


if __name__ == "__main__":
    main()
