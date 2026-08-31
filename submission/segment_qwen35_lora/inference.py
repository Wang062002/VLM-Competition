"""ORena FOCUS SEGMENT submission inference for Qwen3.5-4B + LoRA.

Expected resource layout inside the official submission template::

    resources/qwen35-4b/
    resources/qwen35-lora/

The official platform mounts one batch at /input and expects
/output/answer.json. Runtime internet access is unavailable, so the base model,
processor, and adapter must all be present under ``resources``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import decord
import numpy as np
import torch
from focus import Request, Response, load_requests, save_items
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("segment_qwen35_lora")

INPUT_PATH = Path(os.environ.get("ORENA_INPUT_PATH", "/input"))
OUTPUT_PATH = Path(os.environ.get("ORENA_OUTPUT_PATH", "/output"))
RESOURCES_PATH = Path(__file__).parent / "resources"

BASE_MODEL_PATH = RESOURCES_PATH / "qwen35-4b"
ADAPTER_PATH = RESOURCES_PATH / "qwen35-lora"
VIDEO_SUBDIR_CANDIDATES = (
    "overlayed",
    "plain",
    "batch-videos/overlayed",
    "batch-videos/plain",
    "batch-videos",
)

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "1.0"))
VIDEO_MIN_FRAMES = int(os.environ.get("VIDEO_MIN_FRAMES", "4"))
VIDEO_MAX_FRAMES = int(os.environ.get("VIDEO_MAX_FRAMES", "64"))
VIDEO_WIDTH = int(os.environ.get("VIDEO_WIDTH", "640"))
VIDEO_HEIGHT = int(os.environ.get("VIDEO_HEIGHT", "360"))
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION", "sdpa")
USE_BFLOAT16 = os.environ.get("USE_BFLOAT16", "1") != "0"


def read_fo_definitions() -> str:
    """Read the official foreign-object definitions as prompt text."""
    value = json.loads((INPUT_PATH / "FO_definitions.json").read_text(encoding="utf-8"))
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def input_tree_preview(limit: int = 80) -> str:
    try:
        entries: list[str] = []
        for path in sorted(INPUT_PATH.rglob("*")):
            if len(entries) >= limit:
                entries.append("...")
                break
            entries.append(str(path.relative_to(INPUT_PATH)))
        return ", ".join(entries) if entries else "<empty>"
    except Exception as exc:
        return f"<could not inspect input tree: {exc}>"


def clip_path_for(req: Request) -> Path:
    """Locate a qID-named SEGMENT clip across known platform layouts."""
    filename = f"{req.qID}.mp4"
    for subdir in VIDEO_SUBDIR_CANDIDATES:
        candidate = INPUT_PATH / subdir / filename
        if candidate.exists():
            return candidate

    direct = INPUT_PATH / filename
    if direct.exists():
        return direct

    matches = sorted(INPUT_PATH.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find video clip {filename!r} under {INPUT_PATH}. "
        f"Input tree preview: {input_tree_preview()}"
    )


def evenly_cap_indices(indices: list[int], max_frames: int) -> list[int]:
    if max_frames <= 0 or len(indices) <= max_frames:
        return indices
    if max_frames == 1:
        return [indices[len(indices) // 2]]
    last = len(indices) - 1
    positions = [round(index * last / (max_frames - 1)) for index in range(max_frames)]
    return [indices[position] for position in positions]


def sample_video(path: Path) -> tuple[np.ndarray, float]:
    """Decode once and reproduce the capped in-memory training representation."""
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    frame_count = len(reader)
    if frame_count < 1:
        raise ValueError(f"Video has no frames: {path}")

    source_fps = float(reader.get_avg_fps()) or 5.0
    stride = max(round(source_fps / max(VIDEO_FPS, 1e-6)), 1)
    indices = list(range(0, frame_count, stride)) or [0]
    if len(indices) < VIDEO_MIN_FRAMES:
        if VIDEO_MIN_FRAMES == 1:
            indices = [0]
        else:
            indices = [
                round(position * (frame_count - 1) / (VIDEO_MIN_FRAMES - 1))
                for position in range(VIDEO_MIN_FRAMES)
            ]
    indices = evenly_cap_indices(indices, VIDEO_MAX_FRAMES)
    frames = reader.get_batch(indices).asnumpy()
    del reader

    resized = np.empty(
        (len(frames), VIDEO_HEIGHT, VIDEO_WIDTH, 3),
        dtype=np.uint8,
    )
    for index, frame in enumerate(frames):
        resized[index] = cv2.resize(
            frame,
            (VIDEO_WIDTH, VIDEO_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    if len(indices) > 1 and indices[-1] > indices[0]:
        sampled_duration = (indices[-1] - indices[0]) / source_fps
        sampled_fps = (len(indices) - 1) / sampled_duration
    else:
        sampled_fps = max(source_fps / stride, 1.0)
    return resized, sampled_fps


def build_messages(
    question: str,
    video_frames: np.ndarray,
    system_prompt: str,
) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_frames},
                {"type": "text", "text": question},
            ],
        },
    ]


class Qwen35LoraEngine:
    def __init__(self, device: torch.device, system_prompt: str) -> None:
        self.device = device
        self.system_prompt = system_prompt
        self.processor = None
        self.model = None

    def load(self) -> None:
        if not BASE_MODEL_PATH.is_dir():
            raise FileNotFoundError(f"Missing base model directory: {BASE_MODEL_PATH}")
        if not ADAPTER_PATH.is_dir():
            raise FileNotFoundError(f"Missing LoRA adapter directory: {ADAPTER_PATH}")

        LOGGER.info("Loading processor from %s", ADAPTER_PATH)
        self.processor = AutoProcessor.from_pretrained(
            str(ADAPTER_PATH),
            local_files_only=True,
            trust_remote_code=True,
        )

        dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and USE_BFLOAT16
            else torch.float32
        )
        LOGGER.info("Loading base model from %s with dtype=%s", BASE_MODEL_PATH, dtype)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(BASE_MODEL_PATH),
            torch_dtype=dtype,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
        self.model.to(self.device)

        LOGGER.info("Loading LoRA adapter from %s", ADAPTER_PATH)
        self.model = PeftModel.from_pretrained(
            self.model,
            str(ADAPTER_PATH),
            local_files_only=True,
        ).eval()
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_length = None
        for config in (
            self.model.config,
            getattr(self.model.config, "text_config", None),
        ):
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = True

    def predict(self, req: Request) -> str:
        assert self.processor is not None
        assert self.model is not None

        video_path = clip_path_for(req)
        video_frames, sampled_fps = sample_video(video_path)
        sampled_frames = len(video_frames)
        LOGGER.info(
            "qID=%s video=%s sampled_frames=%d sampled_fps=%.4f",
            req.qID,
            video_path,
            sampled_frames,
            sampled_fps,
        )
        messages = build_messages(req.question, video_frames, self.system_prompt)
        formatted = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[formatted],
            videos=[video_frames],
            return_tensors="pt",
            text_kwargs={"add_special_tokens": False},
            videos_kwargs={
                "do_sample_frames": False,
                "video_metadata": [
                    {
                        "total_num_frames": sampled_frames,
                        "fps": sampled_fps,
                        "duration": sampled_frames / sampled_fps,
                        "frames_indices": np.arange(sampled_frames),
                    }
                ],
            },
        )
        inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        answer_tokens = generated[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            answer_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def run() -> int:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    LOGGER.info("=== ORena FOCUS SEGMENT Qwen3.5-LoRA inference start ===")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)
    if device.type == "cuda":
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(device))

    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        LOGGER.error("No requests found")
        return 1
    LOGGER.info("Batch size: %d", len(requests))
    LOGGER.info("Input path: %s", INPUT_PATH)
    LOGGER.info("Input tree preview: %s", input_tree_preview())

    system_prompt = (
        "You are a surgical assistant. You are given endoscopic video from a "
        "minimally invasive procedure. Analyze the footage and answer the surgical "
        "question based on the visual evidence. Be precise and concise.\n\n"
        + read_fo_definitions()
    )
    engine = Qwen35LoraEngine(device=device, system_prompt=system_prompt)
    engine.load()
    LOGGER.info("Model setup completed in %.2f s", time.monotonic() - started)

    responses: list[Response] = []
    failures = 0
    for index, req in enumerate(requests, start=1):
        LOGGER.info("[%d/%d] qID=%s question=%r", index, len(requests), req.qID, req.question)
        sample_started = time.monotonic()
        try:
            answer = engine.predict(req)
        except Exception:
            failures += 1
            LOGGER.exception("[%d/%d] qID=%s failed; emitting empty answer", index, len(requests), req.qID)
            answer = ""
        latency = time.monotonic() - sample_started
        LOGGER.info(
            "[%d/%d] qID=%s latency=%.3f answer=%r",
            index,
            len(requests),
            req.qID,
            latency,
            answer,
        )
        responses.append(Response(qID=req.qID, content=answer, latency=latency))

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_items(responses, OUTPUT_PATH / "answer.json")
    LOGGER.info(
        "Wrote %d responses with %d failures in %.2f s",
        len(responses),
        failures,
        time.monotonic() - started,
    )
    # Preserve all successful answers if one hidden sample is malformed. Local
    # validation still treats any nonzero failure count as a release blocker.
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
