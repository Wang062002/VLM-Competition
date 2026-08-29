"""Read-only preflight check for Qwen3.5 LoRA-SFT training."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpus", type=int, default=1)
    parser.add_argument("--model-path")
    parser.add_argument("--train-jsonl")
    parser.add_argument("--val-jsonl")
    parser.add_argument("--json-output")
    return parser.parse_args()


def module_version(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "version": getattr(module, "__version__", "unknown")}


def path_status(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }


def main() -> None:
    args = parse_args()
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: module_version(name)
            for name in ("torch", "transformers", "peft", "decord", "cv2")
        },
        "optional_acceleration_packages": {
            name: module_version(name)
            for name in ("causal_conv1d", "fla", "flash_attn", "kernels")
        },
        "paths": {
            "model": path_status(args.model_path),
            "train_jsonl": path_status(args.train_jsonl),
            "val_jsonl": path_status(args.val_jsonl),
        },
    }

    errors: list[str] = []
    try:
        import torch

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        result["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": torch.cuda.is_bf16_supported()
            if torch.cuda.is_available()
            else False,
            "gpu_count": gpu_count,
            "gpus": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_gib": round(
                        torch.cuda.get_device_properties(index).total_memory / 2**30,
                        2,
                    ),
                    "compute_capability": ".".join(
                        str(part) for part in torch.cuda.get_device_capability(index)
                    ),
                }
                for index in range(gpu_count)
            ],
        }
        if gpu_count < args.require_gpus:
            errors.append(f"Expected at least {args.require_gpus} GPUs, found {gpu_count}.")
        if gpu_count and not torch.cuda.is_bf16_supported():
            errors.append("Visible CUDA device does not report BF16 support.")
    except Exception as exc:
        errors.append(f"PyTorch CUDA check failed: {type(exc).__name__}: {exc}")

    try:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        result["qwen35_classes"] = {
            "AutoProcessor": AutoProcessor.__name__,
            "Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGeneration.__name__,
        }
    except Exception as exc:
        errors.append(f"Qwen3.5 Transformers class check failed: {type(exc).__name__}: {exc}")

    for package_name, status in result["packages"].items():
        if not status["available"]:
            errors.append(f"Missing or broken package: {package_name}: {status['error']}")
    for path_name, status in result["paths"].items():
        if status is not None and not status["exists"]:
            errors.append(f"Missing {path_name} path: {status['path']}")

    result["status"] = "ok" if not errors else "error"
    result["errors"] = errors
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
