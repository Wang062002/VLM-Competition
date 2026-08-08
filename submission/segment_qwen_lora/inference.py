"""ORena FOCUS SEGMENT submission inference for Qwen3-VL + LoRA.

This file is intended to replace:

    orena-focus-submission-template/segment-algorithm/inference.py

Expected resource layout inside the official template:

    resources/qwen3vl-4b/
    resources/qwen3vl-lora/

The official platform mounts one batch at /input and expects /output/answer.json.
Runtime internet is unavailable, so all model files must be local.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from focus import Request, Response, load_requests, save_items
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover - fail fast in container
    raise RuntimeError("Missing dependency: peft") from exc


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("segment_qwen_lora")

INPUT_PATH = Path(os.environ.get("ORENA_INPUT_PATH", "/input"))
OUTPUT_PATH = Path(os.environ.get("ORENA_OUTPUT_PATH", "/output"))
RESOURCES_PATH = Path(__file__).parent / "resources"

BASE_MODEL_PATH = RESOURCES_PATH / "qwen3vl-4b"
ADAPTER_PATH = RESOURCES_PATH / "qwen3vl-lora"
VIDEO_DIR = INPUT_PATH / "overlayed"

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "1.0"))
VIDEO_MIN_FRAMES = int(os.environ.get("VIDEO_MIN_FRAMES", "4"))
VIDEO_MAX_FRAMES = int(os.environ.get("VIDEO_MAX_FRAMES", "64"))
USE_BFLOAT16 = os.environ.get("USE_BFLOAT16", "1") != "0"


def read_fo_definitions() -> str:
    """Read official FO definitions, encoded as a JSON string."""
    path = INPUT_PATH / "FO_definitions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def clip_path_for(req: Request) -> Path:
    return VIDEO_DIR / f"{req.qID}.mp4"


def inspect_video(path: Path) -> dict[str, float | int]:
    """Best-effort metadata for Qwen video processor."""
    try:
        import decord

        vr = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        n_frames = len(vr)
        frame = vr[0].asnumpy()
        height, width = int(frame.shape[0]), int(frame.shape[1])
        del vr
        return {"fps": fps, "n_frames": n_frames, "width": width, "height": height}
    except Exception:
        LOGGER.exception("Video metadata inspection failed for %s", path)
        return {"fps": 5.0, "n_frames": 0, "width": 640, "height": 360}


def normalize_video_kwargs(video_kwargs: dict) -> dict:
    """Normalize qwen-vl-utils kwargs for single-sample processor calls."""
    normalized = dict(video_kwargs)
    if isinstance(normalized.get("fps"), list) and len(normalized["fps"]) == 1:
        normalized["fps"] = float(normalized["fps"][0])
    return normalized


class QwenLoraEngine:
    def __init__(self, device: str, system_prompt: str) -> None:
        self.device = device
        self.system_prompt = system_prompt
        self.processor = None
        self.model = None

    def load(self) -> None:
        if not BASE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing base model directory: {BASE_MODEL_PATH}")
        if not ADAPTER_PATH.exists():
            raise FileNotFoundError(f"Missing LoRA adapter directory: {ADAPTER_PATH}")

        LOGGER.info("Loading processor from %s", ADAPTER_PATH)
        self.processor = AutoProcessor.from_pretrained(
            str(ADAPTER_PATH),
            local_files_only=True,
        )

        dtype = torch.bfloat16 if self.device.startswith("cuda") and USE_BFLOAT16 else torch.float32
        LOGGER.info("Loading base model from %s with dtype=%s", BASE_MODEL_PATH, dtype)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(BASE_MODEL_PATH),
            torch_dtype=dtype,
            local_files_only=True,
        ).eval()
        self.model.to(self.device)

        LOGGER.info("Loading LoRA adapter from %s", ADAPTER_PATH)
        self.model = PeftModel.from_pretrained(
            self.model,
            str(ADAPTER_PATH),
            local_files_only=True,
        ).eval()

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_length = None

    def predict(self, req: Request) -> str:
        assert self.processor is not None
        assert self.model is not None

        video_path = clip_path_for(req)
        metadata = inspect_video(video_path)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "fps": VIDEO_FPS,
                        "min_frames": VIDEO_MIN_FRAMES,
                        "max_frames": VIDEO_MAX_FRAMES,
                        "video_metadata": {
                            "fps": VIDEO_FPS,
                            "width": metadata["width"],
                            "height": metadata["height"],
                        },
                    },
                    {"type": "text", "text": req.question},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except TypeError:
            image_inputs, video_inputs = process_vision_info(messages)
            video_kwargs = {}
            video_metadatas = None
        else:
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs = list(video_inputs)
                video_metadatas = list(video_metadatas)
            else:
                video_metadatas = None
        video_kwargs = normalize_video_kwargs(video_kwargs)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        trimmed = [ids[len(inputs.input_ids[0]) :] for ids in generated]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def run() -> int:
    started = time.monotonic()
    LOGGER.info("=== ORena FOCUS SEGMENT Qwen-LoRA inference start ===")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Device: %s", device)
    if torch.cuda.is_available():
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))

    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        LOGGER.error("No requests found")
        return 1
    LOGGER.info("Batch size: %d", len(requests))

    fo_definitions = read_fo_definitions()
    system_prompt = (
        "You are a surgical assistant. You are given endoscopic video from a "
        "minimally invasive procedure. Analyze the footage and answer the surgical "
        "question based on the visual evidence. Be precise and concise.\n\n"
        + fo_definitions
    )

    engine = QwenLoraEngine(device=device, system_prompt=system_prompt)
    engine.load()
    LOGGER.info("Model setup completed in %.2f s", time.monotonic() - started)

    responses: list[Response] = []
    failures = 0
    for index, req in enumerate(requests, start=1):
        LOGGER.info("[%d/%d] qID=%s question=%r", index, len(requests), req.qID, req.question)
        t0 = time.monotonic()
        try:
            answer = engine.predict(req)
        except Exception:
            failures += 1
            LOGGER.exception("[%d/%d] qID=%s failed; emitting empty answer", index, len(requests), req.qID)
            answer = ""
        latency = time.monotonic() - t0
        LOGGER.info("[%d/%d] qID=%s latency=%.3f answer=%r", index, len(requests), req.qID, latency, answer)
        responses.append(Response(qID=req.qID, content=answer, latency=latency))

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_items(responses, OUTPUT_PATH / "answer.json")
    LOGGER.info(
        "Wrote %d responses with %d failures in %.2f s",
        len(responses),
        failures,
        time.monotonic() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
