"""Smoke-test LoRA SFT for Qwen3-VL on ORena FOCUS SEGMENT.

This script intentionally starts small. Its goal is to verify the training
plumbing before launching a full run:

- read the leakage-safe JSONL manifests produced by
  `audit_and_split_segment_train.py`
- cut the timestamp-overlaid video segment for each QA sample
- format the sample as a Qwen chat-style video QA example
- inject LoRA adapters
- run a few forward/backward steps
- save the adapter checkpoint

Run the first test with a tiny subset, then scale only after this succeeds.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import decord
import torch
from progiter import ProgIter
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from focus import FO_DEFINITIONS_FILE

LOGGER = logging.getLogger("qwen3vl_lora_sft_smoke")


class InvalidClipError(ValueError):
    """Raised when a QA time window cannot be cut from its source video."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--train-jsonl",
        default=(
            "~/workspace/focus-runs/data-audit/segment-trainval-seed20260707/"
            "sft_train_overlay.jsonl"
        ),
    )
    parser.add_argument(
        "--val-jsonl",
        default=(
            "~/workspace/focus-runs/data-audit/segment-trainval-seed20260707/"
            "sft_val_overlay.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="~/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-val-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--video-stride", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--save-every-step", action="store_true")
    parser.add_argument(
        "--invalid-clip-policy",
        choices=["skip", "error"],
        default="skip",
        help="How to handle samples whose time window is outside the source video.",
    )
    return parser.parse_args()


def require_peft():
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `peft`. Install it in the remote environment with:\n"
            "  pip install peft\n"
            "If using --load-in-4bit, also install bitsandbytes."
        ) from exc
    return LoraConfig, get_peft_model, prepare_model_for_kbit_training


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clip_bounds(row: dict[str, Any], video_path: Path, frames: int, base_fps: float) -> dict[str, Any]:
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
    rows: list[dict[str, Any]],
    split_name: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter rows whose QA time window is outside the source overlay video."""
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
                base_fps = float(vr.get_avg_fps())
                if base_fps <= 0:
                    base_fps = 25.0
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


def make_clip(
    row: dict[str, Any],
    stride: int,
    resolution: tuple[int, int],
    tmp_dir: Path | None = None,
) -> tuple[Path, float]:
    """Cut one overlay video segment to a temporary MP4 file."""
    video_path = Path(row["overlay_video_path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Overlay video does not exist: {video_path}")

    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    base_fps = float(vr.get_avg_fps())
    if base_fps <= 0:
        base_fps = 25.0

    start_frame = round(float(row["start_time"]) * base_fps)
    end_frame = min(round(float(row["end_time"]) * base_fps), len(vr) - 1)
    if start_frame >= len(vr):
        raise InvalidClipError(
            f"Sample {row['qID']} starts beyond video length: "
            f"start_frame={start_frame}, frames={len(vr)}, video={video_path}"
        )

    indices = list(range(start_frame, end_frame + 1, max(stride, 1)))
    if not indices:
        indices = [start_frame]

    frames = vr.get_batch(indices).asnumpy()
    del vr

    out_size = resolution
    tmp = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
        dir=str(tmp_dir) if tmp_dir else None,
    )
    tmp_path = Path(tmp.name)
    tmp.close()

    fps = base_fps / max(stride, 1)
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, out_size, interpolation=cv2.INTER_AREA)
        writer.write(frame_bgr)
    writer.release()

    return tmp_path, fps


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


def build_messages(
    row: dict[str, Any],
    clip_path: Path,
    clip_fps: float,
    resolution: tuple[int, int],
    system_prompt: str,
    include_answer: bool,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{clip_path}",
                    "fps": clip_fps,
                    "video_metadata": {
                        "fps": clip_fps,
                        "width": resolution[0],
                        "height": resolution[1],
                    },
                },
                {"type": "text", "text": sample_question(row)},
            ],
        },
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": sample_answer(row)})
    return messages


def encode_sample(
    processor: AutoProcessor,
    row: dict[str, Any],
    clip_path: Path,
    clip_fps: float,
    resolution: tuple[int, int],
    system_prompt: str,
    device: str,
) -> dict[str, torch.Tensor]:
    full_messages = build_messages(
        row, clip_path, clip_fps, resolution, system_prompt, include_answer=True
    )
    prompt_messages = build_messages(
        row, clip_path, clip_fps, resolution, system_prompt, include_answer=False
    )

    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_image_inputs, full_video_inputs = process_vision_info(full_messages)
    full_inputs = processor(
        text=[full_text],
        images=full_image_inputs,
        videos=full_video_inputs,
        padding=True,
        return_tensors="pt",
    )

    prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_messages)
    prompt_inputs = processor(
        text=[prompt_text],
        images=prompt_image_inputs,
        videos=prompt_video_inputs,
        padding=True,
        return_tensors="pt",
    )

    labels = full_inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, : min(prompt_len, labels.shape[1])] = -100
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    full_inputs["labels"] = labels
    return {key: value.to(device) for key, value in full_inputs.items()}


def load_model_and_processor(args: argparse.Namespace):
    LoraConfig, get_peft_model, prepare_model_for_kbit_training = require_peft()

    processor = AutoProcessor.from_pretrained(args.model_id)

    model_kwargs: dict[str, Any] = {}
    if args.load_in_4bit:
        model_kwargs.update(
            {
                "load_in_4bit": True,
                "device_map": {"": args.device},
                "torch_dtype": torch.bfloat16,
            }
        )
    else:
        model_kwargs.update({"torch_dtype": torch.bfloat16})

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    if not args.load_in_4bit:
        model.to(args.device)

    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None

    if not args.no_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    return model, processor


def run_eval_loss(
    model: torch.nn.Module,
    processor: AutoProcessor,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    system_prompt: str,
    resolution: tuple[int, int],
) -> float | None:
    if not rows:
        return None
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for row in ProgIter(rows, desc="Eval loss"):
            clip_path: Path | None = None
            try:
                clip_path, clip_fps = make_clip(row, args.video_stride, resolution)
                inputs = encode_sample(
                    processor, row, clip_path, clip_fps, resolution, system_prompt, args.device
                )
                loss = model(**inputs).loss
                losses.append(float(loss.detach().cpu()))
            finally:
                if clip_path is not None:
                    clip_path.unlink(missing_ok=True)
    model.train()
    return sum(losses) / len(losses) if losses else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = Path(args.train_jsonl).expanduser().resolve()
    val_jsonl = Path(args.val_jsonl).expanduser().resolve()
    resolution = (args.width, args.height)

    train_rows = read_jsonl(train_jsonl, args.max_train_samples)
    val_rows = read_jsonl(val_jsonl, args.max_val_samples)
    train_rows_before_filter = len(train_rows)
    val_rows_before_filter = len(val_rows)
    train_rows, invalid_train_rows = validate_clip_rows(train_rows, "train", output_dir)
    val_rows, invalid_val_rows = validate_clip_rows(val_rows, "val", output_dir)
    if args.invalid_clip_policy == "error" and (invalid_train_rows or invalid_val_rows):
        raise SystemExit(
            "Invalid clip rows found. See invalid_clips_train.jsonl and/or "
            "invalid_clips_val.jsonl in the output directory."
        )
    if not train_rows:
        raise SystemExit("No valid training rows remain after clip validation.")
    random.Random(args.seed).shuffle(train_rows)

    run_config = vars(args).copy()
    run_config.update(
        {
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "output_dir": str(output_dir),
            "train_rows_before_clip_filter": train_rows_before_filter,
            "val_rows_before_clip_filter": val_rows_before_filter,
            "train_rows_loaded": len(train_rows),
            "val_rows_loaded": len(val_rows),
            "invalid_train_clip_rows": len(invalid_train_rows),
            "invalid_val_clip_rows": len(invalid_val_rows),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )
    write_json(output_dir / "run_config.json", run_config)

    system_prompt = (
        "You are a surgical assistant. You are given endoscopic video from a "
        "minimally invasive procedure. Analyze the footage and answer the surgical "
        "question based on the visual evidence. Be precise and concise.\n\n"
        + FO_DEFINITIONS_FILE.read_text(encoding="utf-8")
    )

    model, processor = load_model_and_processor(args)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_micro_steps = len(train_rows) * args.epochs
    optimizer_steps = math.ceil(total_micro_steps / args.gradient_accumulation_steps)
    LOGGER.info(
        "Starting smoke training: rows=%s epochs=%s micro_steps=%s optimizer_steps≈%s",
        len(train_rows),
        args.epochs,
        total_micro_steps,
        optimizer_steps,
    )

    history_path = output_dir / "train_history.jsonl"
    global_micro_step = 0
    optimizer_step = 0
    accumulated_loss = 0.0
    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(args.epochs):
            for row in ProgIter(train_rows, desc=f"Train epoch {epoch + 1}/{args.epochs}"):
                clip_path: Path | None = None
                try:
                    clip_path, clip_fps = make_clip(row, args.video_stride, resolution)
                    inputs = encode_sample(
                        processor,
                        row,
                        clip_path,
                        clip_fps,
                        resolution,
                        system_prompt,
                        args.device,
                    )
                    outputs = model(**inputs)
                    loss = outputs.loss / args.gradient_accumulation_steps
                    loss.backward()
                    accumulated_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps

                    global_micro_step += 1
                    should_step = (
                        global_micro_step % args.gradient_accumulation_steps == 0
                        or global_micro_step == total_micro_steps
                    )
                    if should_step:
                        torch.nn.utils.clip_grad_norm_(
                            (param for param in model.parameters() if param.requires_grad),
                            args.max_grad_norm,
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1

                        mean_loss = accumulated_loss / args.gradient_accumulation_steps
                        record = {
                            "epoch": epoch + 1,
                            "micro_step": global_micro_step,
                            "optimizer_step": optimizer_step,
                            "loss": mean_loss,
                            "qID": row.get("qID"),
                            "elapsed_sec": round(time.perf_counter() - start_time, 3),
                        }
                        history.write(json.dumps(record, ensure_ascii=False) + "\n")
                        history.flush()
                        LOGGER.info(
                            "step=%s micro_step=%s loss=%.4f qID=%s",
                            optimizer_step,
                            global_micro_step,
                            mean_loss,
                            row.get("qID"),
                        )
                        accumulated_loss = 0.0

                        if args.save_every_step:
                            step_dir = output_dir / f"checkpoint-step-{optimizer_step}"
                            model.save_pretrained(step_dir)
                            processor.save_pretrained(step_dir)
                finally:
                    if clip_path is not None:
                        clip_path.unlink(missing_ok=True)

    eval_loss = run_eval_loss(model, processor, val_rows, args, system_prompt, resolution)
    final_dir = output_dir / "adapter-final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    summary = {
        "status": "completed",
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "train_samples_before_clip_filter": train_rows_before_filter,
        "val_samples_before_clip_filter": val_rows_before_filter,
        "invalid_train_clip_rows": len(invalid_train_rows),
        "invalid_val_clip_rows": len(invalid_val_rows),
        "optimizer_steps": optimizer_step,
        "eval_loss": eval_loss,
        "adapter_dir": str(final_dir),
        "history_path": str(history_path),
    }
    write_json(output_dir / "smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
