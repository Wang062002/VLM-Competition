"""Run one real generation through a completed Qwen3.5 LoRA adapter."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from focus import FO_DEFINITIONS_FILE
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

import scripts.train_qwen35_lora_sft_ddp as training
from submission.segment_qwen35_lora.answer_utils import (
    clean_generated_answer,
    generation_eos_token_ids,
)


TIME_ANSWER = re.compile(r"\d{2}:\d{2}:\d{2}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--video-stride", type=int, default=25)
    parser.add_argument("--video-min-frames", type=int, default=4)
    parser.add_argument("--video-max-frames", type=int, default=64)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the final-adapter generation smoke")
    if args.sample_index < 0:
        raise SystemExit("--sample-index must be non-negative")

    rows = training.read_jsonl(Path(args.manifest), limit=args.sample_index + 1)
    if len(rows) <= args.sample_index:
        raise SystemExit(f"Manifest has no row at index {args.sample_index}")
    row = rows[args.sample_index]
    frames, clip_fps, sampled_frames = training.make_capped_video(
        row,
        args.video_stride,
        args.video_min_frames,
        args.video_max_frames,
        (args.width, args.height),
    )

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    processor = AutoProcessor.from_pretrained(
        args.adapter_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        local_files_only=True,
    ).eval()
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None

    system_prompt = (
        "You are a surgical assistant. You are given endoscopic video from a "
        "minimally invasive procedure. Analyze the footage and answer the surgical "
        "question based on the visual evidence. Be precise and concise.\n\n"
        + FO_DEFINITIONS_FILE.read_text(encoding="utf-8")
    )
    messages = training.build_messages(row, frames, system_prompt, False)
    inputs = training.apply_qwen35_template(
        processor,
        messages,
        frames,
        True,
        clip_fps,
        sampled_frames,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    eos_token_ids = generation_eos_token_ids(processor.tokenizer)
    if eos_token_ids is not None:
        generation_kwargs["eos_token_id"] = eos_token_ids
        generation_kwargs["pad_token_id"] = (
            processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else eos_token_ids[0]
            if isinstance(eos_token_ids, list)
            else eos_token_ids
        )

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    answer_tokens = generated[:, inputs["input_ids"].shape[1] :]
    raw_answer = processor.batch_decode(
        answer_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    answer = clean_generated_answer(raw_answer)
    expected = training.sample_answer(row)

    if not answer:
        raise SystemExit("Generated answer is empty")
    if TIME_ANSWER.fullmatch(expected) and not TIME_ANSWER.fullmatch(answer):
        raise SystemExit(f"Expected a hh:mm:ss answer, got {answer!r}")

    result = {
        "status": "ok",
        "qID": row["qID"],
        "question": training.sample_question(row),
        "expected": expected,
        "raw_answer": raw_answer,
        "cleaned_answer": answer,
        "sampled_frames": sampled_frames,
        "eos_token_ids": eos_token_ids,
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("FINAL_ADAPTER_GENERATION_OK")


if __name__ == "__main__":
    main()
