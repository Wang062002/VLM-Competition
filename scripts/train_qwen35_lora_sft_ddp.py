"""Train Qwen3.5 LoRA-SFT on ORena FOCUS SEGMENT with single-node DDP.

The script is designed for the Cybertron 4 x L20 training job, while still
supporting a one-GPU smoke test. Launch distributed training with ``torchrun``.
Each process owns one GPU and a disjoint shard of the training rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import decord
import numpy as np
import torch
import torch.distributed as dist
from progiter import ProgIter
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from focus import FO_DEFINITIONS_FILE

LOGGER = logging.getLogger("qwen35_lora_sft_ddp")


class InvalidClipError(ValueError):
    """Raised when a QA time window cannot be cut from its source video."""


def normalize_limit(value: int | None) -> int | None:
    return None if value is None or value <= 0 else value


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sample_question(row: dict[str, Any]) -> str:
    for message in row["messages"]:
        if message["role"] != "user":
            continue
        for item in message["content"]:
            if item.get("type") == "text":
                return str(item["text"])
    raise ValueError(f"No user text found for qID={row.get('qID')}")


def sample_answer(row: dict[str, Any]) -> str:
    for message in row["messages"]:
        if message["role"] == "assistant":
            return str(message["content"])
    raise ValueError(f"No assistant answer found for qID={row.get('qID')}")


def clip_bounds(
    row: dict[str, Any], video_path: Path, frames: int, base_fps: float
) -> dict[str, Any]:
    start_frame = round(float(row["start_time"]) * base_fps)
    end_frame = round(float(row["end_time"]) * base_fps)
    return {
        "qID": row.get("qID"),
        "videoID": row.get("videoID"),
        "start_time": row.get("start_time"),
        "end_time": row.get("end_time"),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frames": frames,
        "base_fps": base_fps,
        "video_path": str(video_path),
    }


def validate_clip_rows(
    rows: list[dict[str, Any]], split_name: str, output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    metadata_cache: dict[Path, tuple[int, float]] = {}
    for row in rows:
        video_path = Path(row["overlay_video_path"])
        reason = ""
        frames = 0
        base_fps = 0.0
        if not video_path.exists():
            reason = "missing_video"
        else:
            if video_path not in metadata_cache:
                vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
                base_fps = float(vr.get_avg_fps()) or 25.0
                metadata_cache[video_path] = (len(vr), base_fps)
                del vr
            frames, base_fps = metadata_cache[video_path]
            bounds = clip_bounds(row, video_path, frames, base_fps)
            if frames <= 0:
                reason = "empty_video"
            elif bounds["start_frame"] >= frames:
                reason = "start_beyond_video"
            elif bounds["end_frame"] < 0:
                reason = "end_before_video"
            elif bounds["end_frame"] < bounds["start_frame"]:
                reason = "end_before_start"
        if reason:
            invalid = dict(row)
            invalid.update(clip_bounds(row, video_path, frames, base_fps))
            invalid["invalid_reason"] = reason
            invalid_rows.append(invalid)
        else:
            valid_rows.append(row)
    if invalid_rows:
        invalid_path = output_dir / f"invalid_clips_{split_name}.jsonl"
        write_jsonl(invalid_path, invalid_rows)
        LOGGER.warning(
            "Filtered %s invalid %s clip rows. Details: %s",
            len(invalid_rows),
            split_name,
            invalid_path,
        )
    return valid_rows, invalid_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--video-stride", type=int, default=25)
    parser.add_argument("--video-min-frames", type=int, default=4)
    parser.add_argument("--video-max-frames", type=int, default=64)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default=(
            "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,out_proj,"
            "gate_proj,up_proj,down_proj"
        ),
        help="Comma-separated language-backbone LoRA target suffixes.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-fused-optimizer", action="store_true")
    parser.add_argument("--skip-clip-validation", action="store_true")
    parser.add_argument("--no-save-every-epoch", action="store_true")
    parser.add_argument(
        "--invalid-clip-policy",
        choices=["skip", "error"],
        default="skip",
    )
    return parser.parse_args()


def require_peft():
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise SystemExit("Missing dependency `peft` in the active environment.") from exc
    return LoraConfig, get_peft_model, prepare_model_for_kbit_training


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise SystemExit("Qwen3.5 training requires CUDA.")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
        )
    return rank, local_rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def evenly_cap_indices(indices: list[int], max_frames: int) -> list[int]:
    if max_frames <= 0 or len(indices) <= max_frames:
        return indices
    if max_frames == 1:
        return [indices[len(indices) // 2]]
    last = len(indices) - 1
    positions = [round(i * last / (max_frames - 1)) for i in range(max_frames)]
    return [indices[position] for position in positions]


def make_capped_video(
    row: dict[str, Any],
    stride: int,
    min_frames: int,
    max_frames: int,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, float, int]:
    """Decode a QA window once and return capped RGB frames in memory."""
    video_path = Path(row["overlay_video_path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Overlay video does not exist: {video_path}")

    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    base_fps = float(vr.get_avg_fps()) or 25.0
    start_frame = round(float(row["start_time"]) * base_fps)
    end_frame = min(round(float(row["end_time"]) * base_fps), len(vr) - 1)
    if start_frame >= len(vr):
        raise InvalidClipError(
            f"qID={row.get('qID')} starts beyond video length: "
            f"start_frame={start_frame}, frames={len(vr)}, video={video_path}"
        )
    if end_frame < start_frame:
        raise InvalidClipError(
            f"qID={row.get('qID')} has an empty clip window: "
            f"start_frame={start_frame}, end_frame={end_frame}"
        )

    indices = list(range(start_frame, end_frame + 1, max(stride, 1))) or [start_frame]
    if len(indices) < min_frames:
        if min_frames == 1:
            indices = [start_frame]
        else:
            span = end_frame - start_frame
            indices = [
                start_frame + round(position * span / (min_frames - 1))
                for position in range(min_frames)
            ]
    indices = evenly_cap_indices(indices, max_frames)
    frames = vr.get_batch(indices).asnumpy()
    del vr

    if len(indices) > 1:
        sampled_duration = (indices[-1] - indices[0]) / base_fps
        clip_fps = (len(indices) - 1) / sampled_duration if sampled_duration > 0 else 1.0
    else:
        clip_fps = max(base_fps / max(stride, 1), 1.0)

    resized_frames = np.empty(
        (len(frames), resolution[1], resolution[0], 3),
        dtype=np.uint8,
    )
    for index, frame in enumerate(frames):
        resized_frames[index] = cv2.resize(
            frame,
            resolution,
            interpolation=cv2.INTER_AREA,
        )
    return resized_frames, clip_fps, len(indices)


def build_messages(
    row: dict[str, Any],
    video_frames: np.ndarray,
    system_prompt: str,
    include_answer: bool,
) -> list[dict[str, Any]]:
    video_item: dict[str, Any] = {
        "type": "video",
        "video": video_frames,
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [video_item, {"type": "text", "text": sample_question(row)}],
        },
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": sample_answer(row)})
    return messages


def apply_qwen35_template(
    processor: AutoProcessor,
    messages: list[dict[str, Any]],
    video_frames: np.ndarray,
    add_generation_prompt: bool,
    clip_fps: float,
    sampled_frames: int,
) -> dict[str, torch.Tensor]:
    formatted = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    return processor(
        text=[formatted],
        videos=[video_frames],
        return_tensors="pt",
        text_kwargs={"add_special_tokens": False},
        videos_kwargs={
            "do_sample_frames": False,
            "video_metadata": [
                {
                    "total_num_frames": sampled_frames,
                    "fps": clip_fps,
                    "duration": sampled_frames / clip_fps,
                    "frames_indices": np.arange(sampled_frames),
                }
            ],
        },
    )


def encode_sample(
    processor: AutoProcessor,
    row: dict[str, Any],
    video_frames: np.ndarray,
    clip_fps: float,
    system_prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sampled_frames = len(video_frames)
    full_messages = build_messages(row, video_frames, system_prompt, True)
    prompt_messages = build_messages(row, video_frames, system_prompt, False)
    full_inputs = apply_qwen35_template(
        processor, full_messages, video_frames, False, clip_fps, sampled_frames
    )
    prompt_inputs = apply_qwen35_template(
        processor, prompt_messages, video_frames, True, clip_fps, sampled_frames
    )

    labels = full_inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    del prompt_inputs
    labels[:, : min(prompt_len, labels.shape[1])] = -100
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    if not torch.any(labels != -100):
        raise RuntimeError(f"No assistant target tokens remain for qID={row.get('qID')}")
    full_inputs["labels"] = labels
    return {key: value.to(device, non_blocking=True) for key, value in full_inputs.items()}


def count_target_matches(model: torch.nn.Module, target_modules: list[str]) -> dict[str, int]:
    counts = {target: 0 for target in target_modules}
    for name, _module in model.named_modules():
        for target in target_modules:
            if name.endswith(target):
                counts[target] += 1
    return counts


def freeze_vision_adapters(model: torch.nn.Module) -> int:
    frozen = 0
    for name, parameter in model.named_parameters():
        if "lora_" in name and (name.startswith("visual.") or ".visual." in name):
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    return frozen


def load_model_and_processor(args: argparse.Namespace, device: torch.device, rank: int):
    LoraConfig, get_peft_model, prepare_model_for_kbit_training = require_peft()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": {"": device.index},
            }
        )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    if not args.load_in_4bit:
        model.to(device)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
    if not args.no_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    target_counts = count_target_matches(model, target_modules)
    missing = [name for name, count in target_counts.items() if count == 0]
    if missing:
        raise SystemExit(
            "LoRA target modules not found in this Qwen3.5 checkpoint: " + ", ".join(missing)
        )

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    frozen_vision_parameters = freeze_vision_adapters(model)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if is_main_process(rank):
        LOGGER.info("LoRA target matches: %s", target_counts)
        LOGGER.info("Frozen visual LoRA parameters: %s", frozen_vision_parameters)
        LOGGER.info(
            "Trainable parameters: %s / %s (%.4f%%)",
            trainable,
            total,
            100 * trainable / total,
        )
    model.train()
    return model, processor, target_counts, trainable, total


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def prepare_rows(
    args: argparse.Namespace,
    output_dir: Path,
    rank: int,
    world_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    validated_train = output_dir / "validated_train_rows.jsonl"
    validated_val = output_dir / "validated_val_rows.jsonl"
    stats_path = output_dir / "clip_validation_stats.json"

    if is_main_process(rank):
        train_rows = read_jsonl(
            Path(args.train_jsonl).expanduser().resolve(),
            normalize_limit(args.max_train_samples),
        )
        val_rows = read_jsonl(
            Path(args.val_jsonl).expanduser().resolve(),
            normalize_limit(args.max_val_samples),
        )
        train_before = len(train_rows)
        val_before = len(val_rows)
        invalid_train: list[dict[str, Any]] = []
        invalid_val: list[dict[str, Any]] = []
        if not args.skip_clip_validation:
            train_rows, invalid_train = validate_clip_rows(train_rows, "train", output_dir)
            val_rows, invalid_val = validate_clip_rows(val_rows, "val", output_dir)
        if args.invalid_clip_policy == "error" and (invalid_train or invalid_val):
            raise SystemExit("Invalid clip rows found; inspect the output directory.")
        if len(train_rows) < world_size:
            raise SystemExit(
                f"Need at least {world_size} valid rows for world_size={world_size}; "
                f"found {len(train_rows)}."
            )
        write_jsonl(validated_train, train_rows)
        write_jsonl(validated_val, val_rows)
        write_json(
            stats_path,
            {
                "train_rows_before_clip_filter": train_before,
                "val_rows_before_clip_filter": val_before,
                "train_rows_loaded": len(train_rows),
                "val_rows_loaded": len(val_rows),
                "invalid_train_clip_rows": len(invalid_train),
                "invalid_val_clip_rows": len(invalid_val),
            },
        )
    barrier(world_size)
    train_rows = read_jsonl(validated_train)
    val_rows = read_jsonl(validated_val)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return train_rows, val_rows, stats


def reduce_loss(loss_sum: float, count: int, device: torch.device, world_size: int) -> float:
    totals = torch.tensor([loss_sum, float(count)], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float((totals[0] / totals[1]).item())


def run_eval_loss(
    model: torch.nn.Module,
    processor: AutoProcessor,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    system_prompt: str,
    resolution: tuple[int, int],
    rank: int,
    world_size: int,
    device: torch.device,
) -> float | None:
    if not rows:
        return None
    local_rows = rows[rank::world_size]
    local_loss_sum = 0.0
    local_count = 0
    model.eval()
    iterator = ProgIter(local_rows, desc="Eval loss") if is_main_process(rank) else local_rows
    with torch.no_grad():
        for row in iterator:
            video_frames, clip_fps, _ = make_capped_video(
                row,
                args.video_stride,
                args.video_min_frames,
                args.video_max_frames,
                resolution,
            )
            inputs = encode_sample(
                processor, row, video_frames, clip_fps, system_prompt, device
            )
            local_loss_sum += float(model(**inputs).loss.detach())
            local_count += 1
    model.train()
    return reduce_loss(local_loss_sum, local_count, device, world_size)


def save_adapter(
    model: torch.nn.Module,
    processor: AutoProcessor,
    destination: Path,
    rank: int,
    world_size: int,
) -> None:
    barrier(world_size)
    if is_main_process(rank):
        destination.mkdir(parents=True, exist_ok=True)
        unwrap_model(model).save_pretrained(destination)
        processor.save_pretrained(destination)
        LOGGER.info("Saved adapter: %s", destination)
    barrier(world_size)


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = distributed_context()
    logging.basicConfig(
        level=logging.INFO if is_main_process(rank) else logging.WARNING,
        format=f"%(asctime)s %(levelname)s rank={rank} %(message)s",
    )
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be at least 1.")
    if args.video_min_frames < 1 or args.video_max_frames < args.video_min_frames:
        raise SystemExit(
            "Video frame limits must satisfy 1 <= --video-min-frames "
            "<= --video-max-frames."
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed, rank)

    output_dir = Path(args.output_dir).expanduser().resolve()
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    try:
        train_rows, val_rows, clip_stats = prepare_rows(
            args, output_dir, rank, world_size
        )
        resolution = (args.width, args.height)
        system_prompt = (
            "You are a surgical assistant. You are given endoscopic video from a "
            "minimally invasive procedure. Analyze the footage and answer the surgical "
            "question based on the visual evidence. Be precise and concise.\n\n"
            + FO_DEFINITIONS_FILE.read_text(encoding="utf-8")
        )

        model, processor, target_counts, trainable, total = load_model_and_processor(
            args, device, rank
        )
        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        optimizer_kwargs: dict[str, Any] = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        }
        if not args.no_fused_optimizer:
            optimizer_kwargs["fused"] = True
        try:
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                **optimizer_kwargs,
            )
        except (RuntimeError, TypeError):
            optimizer_kwargs.pop("fused", None)
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                **optimizer_kwargs,
            )

        usable_per_epoch = len(train_rows) - (len(train_rows) % world_size)
        local_rows_per_epoch = usable_per_epoch // world_size
        optimizer_steps_per_epoch = math.ceil(
            local_rows_per_epoch / args.gradient_accumulation_steps
        )
        run_config = vars(args).copy()
        run_config.update(
            {
                **clip_stats,
                "world_size": world_size,
                "global_effective_batch_size": world_size
                * args.gradient_accumulation_steps,
                "usable_train_rows_per_epoch": usable_per_epoch,
                "dropped_rows_per_epoch_for_equal_ddp_shards": len(train_rows)
                - usable_per_epoch,
                "local_rows_per_epoch": local_rows_per_epoch,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "lora_target_matches": target_counts,
                "torch_version": torch.__version__,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device": torch.cuda.get_device_name(local_rank),
            }
        )
        if is_main_process(rank):
            write_json(output_dir / "run_config.json", run_config)
            LOGGER.info(
                "Starting training: world_size=%s rows/epoch=%s local_rows=%s "
                "epochs=%s optimizer_steps/epoch=%s global_batch=%s",
                world_size,
                usable_per_epoch,
                local_rows_per_epoch,
                args.epochs,
                optimizer_steps_per_epoch,
                run_config["global_effective_batch_size"],
            )

        history_path = output_dir / "train_history.jsonl"
        history = history_path.open("w", encoding="utf-8") if is_main_process(rank) else None
        global_micro_step = 0
        optimizer_step = 0
        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        try:
            for epoch in range(args.epochs):
                epoch_rows = train_rows[:]
                random.Random(args.seed + epoch).shuffle(epoch_rows)
                epoch_rows = epoch_rows[:usable_per_epoch]
                rank_rows = epoch_rows[rank::world_size]
                iterator = (
                    ProgIter(rank_rows, desc=f"Train epoch {epoch + 1}/{args.epochs}")
                    if is_main_process(rank)
                    else rank_rows
                )
                accumulated_loss = 0.0
                group_count = 0

                for local_index, row in enumerate(iterator):
                    group_start = (
                        local_index // args.gradient_accumulation_steps
                    ) * args.gradient_accumulation_steps
                    group_size = min(
                        args.gradient_accumulation_steps,
                        len(rank_rows) - group_start,
                    )
                    group_count += 1
                    should_step = group_count == group_size
                    sync_context = (
                        model.no_sync()
                        if isinstance(model, DDP) and not should_step
                        else nullcontext()
                    )
                    video_frames, clip_fps, sampled_frames = make_capped_video(
                        row,
                        args.video_stride,
                        args.video_min_frames,
                        args.video_max_frames,
                        resolution,
                    )
                    inputs = encode_sample(
                        processor,
                        row,
                        video_frames,
                        clip_fps,
                        system_prompt,
                        device,
                    )
                    with sync_context:
                        raw_loss = model(**inputs).loss
                        (raw_loss / group_size).backward()
                    accumulated_loss += float(raw_loss.detach())
                    global_micro_step += 1

                    if should_step:
                        torch.nn.utils.clip_grad_norm_(
                            (
                                parameter
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            ),
                            args.max_grad_norm,
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1
                        mean_loss = reduce_loss(
                            accumulated_loss, group_count, device, world_size
                        )
                        if history is not None:
                            record = {
                                "epoch": epoch + 1,
                                "micro_step_per_rank": global_micro_step,
                                "optimizer_step": optimizer_step,
                                "loss": mean_loss,
                                "qID_rank0": row.get("qID"),
                                "sampled_frames_rank0": sampled_frames,
                                "elapsed_sec": round(time.perf_counter() - start_time, 3),
                            }
                            history.write(json.dumps(record, ensure_ascii=False) + "\n")
                            history.flush()
                            LOGGER.info(
                                "epoch=%s step=%s loss=%.4f sampled_frames=%s qID=%s",
                                epoch + 1,
                                optimizer_step,
                                mean_loss,
                                sampled_frames,
                                row.get("qID"),
                            )
                        accumulated_loss = 0.0
                        group_count = 0

                if not args.no_save_every_epoch:
                    save_adapter(
                        model,
                        processor,
                        output_dir / f"adapter-epoch-{epoch + 1}",
                        rank,
                        world_size,
                    )
        finally:
            if history is not None:
                history.close()

        training_elapsed_sec = time.perf_counter() - start_time
        train_samples_processed = usable_per_epoch * args.epochs
        train_samples_per_sec = train_samples_processed / training_elapsed_sec
        if is_main_process(rank):
            LOGGER.info(
                "Training throughput: %.4f global samples/s (%s samples in %.1f s)",
                train_samples_per_sec,
                train_samples_processed,
                training_elapsed_sec,
            )
        eval_loss = run_eval_loss(
            model,
            processor,
            val_rows,
            args,
            system_prompt,
            resolution,
            rank,
            world_size,
            device,
        )
        final_dir = output_dir / "adapter-final"
        save_adapter(model, processor, final_dir, rank, world_size)
        summary = {
            "status": "completed",
            "epochs": args.epochs,
            "world_size": world_size,
            "train_samples_per_epoch": usable_per_epoch,
            "train_samples_processed": train_samples_processed,
            "training_elapsed_sec": round(training_elapsed_sec, 3),
            "train_samples_per_sec": train_samples_per_sec,
            "val_samples": len(val_rows),
            "optimizer_steps": optimizer_step,
            "eval_loss": eval_loss,
            "total_elapsed_sec": round(time.perf_counter() - start_time, 3),
            "adapter_dir": str(final_dir),
            "history_path": str(history_path),
        }
        if is_main_process(rank):
            write_json(output_dir / "training_summary.json", summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
