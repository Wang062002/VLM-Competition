"""Run Qwen3-VL inference and official evaluation on ORena FOCUS SEGMENT."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import torch
from progiter import ProgIter
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from focus import FO_DEFINITIONS_FILE, FocusConfig, set_config
from focus.data.base_dataset import FocusDataset
from focus.data.data_models import Response
from focus.data.video_dataset import FocusVideoDataset, VideoSample
from focus.enums import DatasetSplit, Track
from focus.evaluation.evaluator import Evaluator

LOGGER = logging.getLogger("run_segment_baseline")


def parse_num_eval(value: str) -> int | None:
    if value.lower() in {"none", "all", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--num-eval must be positive or 'none'.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="/data/focus", help="FOCUS root data directory.")
    parser.add_argument("--dataset", default="heico")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional LoRA adapter directory produced by PEFT training.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-eval", type=parse_num_eval, default=3)
    parser.add_argument("--video-stride", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--no-overlay", action="store_true", help="Use original videos instead of overlay videos.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--official-repo-dir",
        default="~/workspace/orena-focus",
        help="Official orena-focus checkout used for commit recording.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Result directory. Defaults to ~/workspace/focus-runs/<timestamp>.",
    )
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


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class QwenInferenceEngine:
    def __init__(
        self,
        model_id: str,
        adapter_dir: str | None,
        device: str,
        max_new_tokens: int,
        video_resolution: tuple[int, int],
    ) -> None:
        self.model_id = model_id
        self.adapter_dir = adapter_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.video_resolution = video_resolution
        self.model = None
        self.processor = None
        self.system_prompt = (
            "You are a surgical assistant. You are given endoscopic video from a "
            "minimally invasive procedure. Analyze the footage and answer the surgical "
            "question based on the visual evidence. Be precise and concise.\n\n"
            + FO_DEFINITIONS_FILE.read_text(encoding="utf-8")
        )

    def load(self) -> None:
        LOGGER.info("Loading processor and model: %s", self.model_id)
        processor_source = self.adapter_dir or self.model_id
        self.processor = AutoProcessor.from_pretrained(processor_source)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
        ).eval()
        self.model.to(self.device)
        if self.adapter_dir:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise SystemExit(
                    "Missing dependency `peft`. Install it with: pip install peft"
                ) from exc
            LOGGER.info("Loading LoRA adapter: %s", self.adapter_dir)
            self.model = PeftModel.from_pretrained(self.model, self.adapter_dir).eval()
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_length = None

    def predict(self, sample: VideoSample) -> str:
        assert self.processor is not None
        assert self.model is not None

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{sample.video_path}",
                            "fps": sample.fps,
                            "video_metadata": {
                                "fps": sample.fps,
                                "width": self.video_resolution[0],
                                "height": self.video_resolution[1],
                            },
                        },
                        {"type": "text", "text": sample.request.question},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            trimmed = [ids[len(inputs.input_ids[0]) :] for ids in generated]
            return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        except Exception as exc:
            LOGGER.exception("[%s] Inference failed", sample.request.qID)
            return f"Inference Error: {str(exc)[:200]}"
        finally:
            sample.video_path.unlink(missing_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = Path.home() / "workspace" / "focus-runs"
        output_dir = base / f"qwen3vl-4b-segment-{stamp}"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "root_dir": str(root_dir),
        "dataset": args.dataset,
        "split": args.split,
        "track": "segment",
        "model_id": args.model_id,
        "adapter_dir": str(Path(args.adapter_dir).expanduser().resolve()) if args.adapter_dir else None,
        "device": args.device,
        "num_eval": args.num_eval,
        "video_stride": args.video_stride,
        "resolution": [args.width, args.height],
        "use_overlay": not args.no_overlay,
        "max_new_tokens": args.max_new_tokens,
        "helper_repo_commit": git_commit(),
        "official_repo_dir": str(Path(args.official_repo_dir).expanduser()),
        "official_repo_commit": git_commit(Path(args.official_repo_dir).expanduser()),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(output_dir / "run_config.json", config_payload)
    LOGGER.info("Writing results to %s", output_dir)

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
    LOGGER.info("Evaluating %s/%s samples", n_eval, n_total)

    engine = QwenInferenceEngine(
        args.model_id,
        str(Path(args.adapter_dir).expanduser().resolve()) if args.adapter_dir else None,
        args.device,
        args.max_new_tokens,
        (args.width, args.height),
    )
    engine.load()

    requests = []
    references = []
    responses = []
    response_jsonl = output_dir / "responses.jsonl"

    with response_jsonl.open("w", encoding="utf-8") as handle:
        for i, sample in enumerate(ProgIter(video_dataset, total=n_eval, desc="Inference")):
            if i >= n_eval:
                break
            start = time.perf_counter()
            prediction = engine.predict(sample)
            latency = time.perf_counter() - start
            response = Response(
                qID=sample.request.qID,
                content=prediction,
                latency=latency,
            )
            requests.append(sample.request)
            references.append(sample.reference)
            responses.append(response)
            handle.write(
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
            handle.flush()

    if not responses:
        raise RuntimeError("No samples were processed.")

    results_df, summary_df = Evaluator(judge_kwargs={"device": args.device}).run(
        requests=requests,
        references=references,
        responses=responses,
        output_dir=str(output_dir),
    )
    results_df.to_csv(output_dir / "results.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))
    LOGGER.info("Done. Summary saved to %s", output_dir / "summary.csv")


if __name__ == "__main__":
    main()
