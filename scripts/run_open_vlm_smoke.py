"""Smoke-test candidate open VLMs on ORena FOCUS SEGMENT samples.

This script intentionally uses a conservative common input format: each FOCUS
video clip is sampled into a small set of RGB frames. Video-native models can be
improved later with their specialized video APIs, but this first pass is meant
to quickly answer whether each downloaded model can load, consume visual input,
generate non-empty answers, and pass through the official evaluator.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import decord
import torch
from PIL import Image
from progiter import ProgIter

from focus import FO_DEFINITIONS_FILE, FocusConfig, set_config
from focus.data.base_dataset import FocusDataset
from focus.data.data_models import Response
from focus.data.video_dataset import FocusVideoDataset, VideoSample
from focus.enums import DatasetSplit, Track
from focus.evaluation.evaluator import Evaluator


LOGGER = logging.getLogger("run_open_vlm_smoke")
DEFAULT_CONFIG = "configs/vlm_candidate_models.csv"
DEFAULT_MODEL_DIR = "~/workspace/vlm-models"
DEFAULT_OUTPUT_DIR = "~/workspace/focus-runs/open-vlm-smoke"


def parse_num_eval(value: str) -> int | None:
    if value.lower() in {"none", "all", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--num-eval must be positive or 'none'.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model", action="append", default=None, help="Candidate key. Repeatable.")
    parser.add_argument("--root-dir", default="/home/Jiali_Wang/data/focus")
    parser.add_argument("--dataset", default="heico")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--num-eval", type=parse_num_eval, default=3)
    parser.add_argument("--video-stride", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-evaluator", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def git_commit(repo_dir: Path | None = None) -> str | None:
    try:
        command = ["git"]
        if repo_dir is not None:
            command += ["-C", str(repo_dir)]
        command += ["rev-parse", "HEAD"]
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except Exception:
        return None


def safe_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sample_clip_frames(video_path: Path, count: int) -> list[Image.Image]:
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    total = len(vr)
    if total <= 0:
        raise RuntimeError(f"Clip has no frames: {video_path}")
    count = max(1, min(count, total))
    if count == 1:
        indices = [total // 2]
    else:
        indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
    frames = vr.get_batch(indices).asnumpy()
    del vr
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def system_prompt() -> str:
    return (
        "You are a surgical assistant. You are given sampled frames from an "
        "endoscopic video segment. Answer the surgical question based only on "
        "the visual evidence. Be precise and concise.\n\n"
        + FO_DEFINITIONS_FILE.read_text(encoding="utf-8")
    )


def move_inputs_to_device(inputs: Any, device: str, dtype: torch.dtype | None = None) -> Any:
    if hasattr(inputs, "to"):
        if dtype is not None and device.startswith("cuda"):
            return inputs.to(device, dtype=dtype)
        return inputs.to(device)
    return inputs


def first_parameter_device(model: torch.nn.Module, fallback: str) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return fallback


def patch_transformers_tied_weights_compat() -> None:
    """Bridge MiniCPM remote-code models across Transformers API variants."""
    from transformers.modeling_utils import PreTrainedModel

    existing = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if existing is not None and not isinstance(existing, property):
        return
    if isinstance(existing, property) and existing.fset is not None:
        return

    def get_all_tied_weights_keys(self: Any) -> dict[str, str]:
        stored = self.__dict__.get("all_tied_weights_keys")
        if stored is not None:
            if isinstance(stored, dict):
                return stored
            return {str(key): str(key) for key in stored}
        keys = getattr(self, "_tied_weights_keys", None) or []
        if isinstance(keys, dict):
            return keys
        return {str(key): str(key) for key in keys}

    def set_all_tied_weights_keys(self: Any, value: Any) -> None:
        self.__dict__["all_tied_weights_keys"] = value

    PreTrainedModel.all_tied_weights_keys = property(  # type: ignore[attr-defined]
        get_all_tied_weights_keys,
        set_all_tied_weights_keys,
    )


def patch_minicpm_processor_register_compat() -> None:
    """Allow MiniCPM remote-code processor imports on stricter Transformers builds."""
    from transformers import AutoImageProcessor

    original_register = AutoImageProcessor.register
    if getattr(original_register, "_vlm_competition_patched", False):
        return

    def register_compat(config_class: Any, image_processor_class: Any = None, exist_ok: bool = False) -> Any:
        if isinstance(config_class, str):
            LOGGER.debug(
                "Skipping string-based AutoImageProcessor.register(%r) from remote code.",
                config_class,
            )
            return None
        return original_register(config_class, image_processor_class, exist_ok=exist_ok)

    register_compat._vlm_competition_patched = True  # type: ignore[attr-defined]
    AutoImageProcessor.register = register_compat  # type: ignore[method-assign]


class BaseEngine:
    def __init__(self, model_path: Path, device: str, max_new_tokens: int) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt()

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, frames: list[Image.Image], question: str) -> str:
        raise NotImplementedError


class MiniCPMVEngine(BaseEngine):
    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        patch_transformers_tied_weights_compat()
        patch_minicpm_processor_register_compat()
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
        ).eval()
        if self.device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)

    def predict(self, frames: list[Image.Image], question: str) -> str:
        patch_minicpm_processor_register_compat()
        prompt = self.system_prompt + "\n\nQuestion: " + question
        msgs = [{"role": "user", "content": frames + [prompt]}]
        answer = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            use_image_id=False,
            max_slice_nums=1,
            max_new_tokens=self.max_new_tokens,
            enable_thinking=False,
        )
        if not isinstance(answer, str):
            answer = "".join(list(answer))
        return answer.strip()


class LlavaOneVisionEngine(BaseEngine):
    def load(self) -> None:
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            low_cpu_mem_usage=True,
        ).eval()
        if self.device.startswith("cuda"):
            self.model = self.model.to(self.device)

    def predict(self, frames: list[Image.Image], question: str) -> str:
        content = [{"type": "image"} for _ in frames]
        content.append({"type": "text", "text": self.system_prompt + "\n\nQuestion: " + question})
        conversation = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=frames, text=prompt, return_tensors="pt")
        inputs = move_inputs_to_device(inputs, self.device, torch.float16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()


class GemmaImageTextEngine(BaseEngine):
    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            device_map="auto" if self.device.startswith("cuda") else None,
        ).eval()
        if self.device.startswith("cuda") and not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(self.device)
        self.input_device = getattr(self.model, "device", first_parameter_device(self.model, self.device))

    def predict(self, frames: list[Image.Image], question: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "image", "image": frame} for frame in frames]
                + [{"type": "text", "text": question}],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = move_inputs_to_device(inputs, str(self.input_device), torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()


class InternVLEngine(BaseEngine):
    def load(self) -> None:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer

        patch_transformers_tied_weights_compat()
        self._torchvision_transforms = T
        self._interpolation = InterpolationMode
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            use_fast=False,
            fix_mistral_regex=True,
        )
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
        ).eval()
        if self.device.startswith("cuda"):
            self.model = self.model.to(self.device)

    def _transform(self, image: Image.Image, input_size: int = 448) -> torch.Tensor:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = self._torchvision_transforms.Compose(
            [
                self._torchvision_transforms.Lambda(lambda img: img.convert("RGB")),
                self._torchvision_transforms.Resize(
                    (input_size, input_size),
                    interpolation=self._interpolation.BICUBIC,
                ),
                self._torchvision_transforms.ToTensor(),
                self._torchvision_transforms.Normalize(mean=mean, std=std),
            ]
        )
        return transform(image)

    def predict(self, frames: list[Image.Image], question: str) -> str:
        pixel_values = torch.stack([self._transform(frame) for frame in frames])
        pixel_values = pixel_values.to(torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        if self.device.startswith("cuda"):
            pixel_values = pixel_values.to(self.device)
        num_patches_list = [1] * len(frames)
        frame_tokens = "\n".join(f"Frame-{i + 1}: <image>" for i in range(len(frames)))
        prompt = f"{frame_tokens}\n{self.system_prompt}\n\nQuestion: {question}"
        generation_config = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
            num_patches_list=num_patches_list,
        )
        return str(response).strip()


def build_engine(engine: str, model_path: Path, device: str, max_new_tokens: int) -> BaseEngine:
    if engine == "minicpmv":
        return MiniCPMVEngine(model_path, device, max_new_tokens)
    if engine == "llava_onevision":
        return LlavaOneVisionEngine(model_path, device, max_new_tokens)
    if engine == "internvl":
        return InternVLEngine(model_path, device, max_new_tokens)
    if engine in {"gemma3", "medgemma"}:
        return GemmaImageTextEngine(model_path, device, max_new_tokens)
    raise ValueError(f"Unsupported engine: {engine}")


def run_one_model(
    candidate: dict[str, str],
    args: argparse.Namespace,
    video_dataset: FocusVideoDataset,
    n_eval: int,
    output_root: Path,
) -> dict[str, Any]:
    key = candidate["key"]
    repo_id = candidate["repo_id"]
    model_path = Path(args.model_dir).expanduser().resolve() / safe_dir_name(repo_id)
    output_dir = output_root / key
    output_dir.mkdir(parents=True, exist_ok=True)
    response_jsonl = output_dir / "responses.jsonl"
    errors_jsonl = output_dir / "errors.jsonl"

    run_config = {
        "key": key,
        "repo_id": repo_id,
        "engine": candidate["engine"],
        "model_path": str(model_path),
        "num_eval": n_eval,
        "frames_per_clip": args.frames_per_clip,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "use_overlay": not args.no_overlay,
        "helper_repo_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(output_dir / "run_config.json", run_config)

    LOGGER.info("[%s] Loading engine from %s", key, model_path)
    engine = build_engine(candidate["engine"], model_path, args.device, args.max_new_tokens)
    engine.load()

    requests = []
    references = []
    responses = []
    failures: list[dict[str, Any]] = []

    with response_jsonl.open("w", encoding="utf-8") as response_handle, errors_jsonl.open(
        "w", encoding="utf-8"
    ) as error_handle:
        for i, sample in enumerate(ProgIter(video_dataset, total=n_eval, desc=f"{key}")):
            if i >= n_eval:
                break
            start = time.perf_counter()
            try:
                frames = sample_clip_frames(sample.video_path, args.frames_per_clip)
                prediction = engine.predict(frames, sample.request.question)
                latency = time.perf_counter() - start
            except Exception as exc:
                LOGGER.exception("[%s] qID=%s failed", key, sample.request.qID)
                prediction = f"Inference Error: {str(exc)[:300]}"
                latency = time.perf_counter() - start
                failure = {
                    "qID": sample.request.qID,
                    "videoID": sample.request.videoID,
                    "error": repr(exc),
                }
                failures.append(failure)
                error_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                error_handle.flush()
            finally:
                sample.video_path.unlink(missing_ok=True)

            response = Response(qID=sample.request.qID, content=prediction, latency=latency)
            requests.append(sample.request)
            references.append(sample.reference)
            responses.append(response)
            response_handle.write(
                json.dumps(
                    {
                        "qID": sample.request.qID,
                        "videoID": sample.request.videoID,
                        "question": sample.request.question,
                        "prediction": prediction,
                        "latency": latency,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            response_handle.flush()

    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.skip_evaluator:
        summary_df = None
    else:
        results_df, summary_df = Evaluator(judge_kwargs={"device": args.device}).run(
            requests=requests,
            references=references,
            responses=responses,
            output_dir=str(output_dir),
        )
        results_df.to_csv(output_dir / "results.csv", index=False)
        summary_df.to_csv(output_dir / "summary.csv", index=False)

    status = {
        "key": key,
        "repo_id": repo_id,
        "status": "completed",
        "processed": len(responses),
        "failures": len(failures),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "status.json", status)
    if summary_df is not None:
        print(f"\n== {key} summary ==")
        print(summary_df.to_string(index=False))
    else:
        print(f"\n== {key} responses written; evaluator skipped ==")
    return status


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = read_candidates(config_path)
    by_key = {row["key"]: row for row in candidates}
    selected_keys = args.model or list(by_key)
    missing = [key for key in selected_keys if key not in by_key]
    if missing:
        raise SystemExit(f"Unknown candidate key(s): {missing}. Available: {sorted(by_key)}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    set_config(FocusConfig(root_dir=str(root_dir)))
    base_dataset = FocusDataset(
        dataset=args.dataset,
        split=DatasetSplit(args.split),
        track=Track.SEGMENT,
    )
    video_dataset = FocusVideoDataset(
        base_dataset,
        stride=args.video_stride,
        use_overlay=not args.no_overlay,
        resolution=(args.width, args.height),
    )
    n_total = len(video_dataset)
    n_eval = min(args.num_eval, n_total) if args.num_eval else n_total

    statuses: list[dict[str, Any]] = []
    for key in selected_keys:
        try:
            status = run_one_model(by_key[key], args, video_dataset, n_eval, output_root)
            statuses.append(status)
        except Exception as exc:
            LOGGER.exception("[%s] model-level failure", key)
            status = {
                "key": key,
                "status": "failed",
                "error": repr(exc),
                "output_dir": str(output_root / key),
            }
            statuses.append(status)
            write_json(output_root / key / "status.json", status)
            if not args.continue_on_error:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_json(output_root / "batch_status.json", statuses)
    print(json.dumps(statuses, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
