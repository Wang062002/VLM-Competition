"""Validate staged offline resources for the Qwen3.5 SEGMENT container."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm-root",
        required=True,
        help="Path to the official template's segment-algorithm directory.",
    )
    return parser.parse_args()


def load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"Expected a JSON object in {path}")
        return {}
    return value


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    args = parse_args()
    algorithm_root = Path(args.algorithm_root).expanduser().resolve()
    base_dir = algorithm_root / "resources" / "qwen35-4b"
    adapter_dir = algorithm_root / "resources" / "qwen35-lora"
    errors: list[str] = []

    required_files = [
        algorithm_root / "inference.py",
        algorithm_root / "answer_utils.py",
        algorithm_root / "requirements.txt",
        base_dir / "config.json",
        base_dir / "tokenizer_config.json",
        adapter_dir / "adapter_config.json",
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "tokenizer_config.json",
    ]
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty required file: {path}")

    for directory in (base_dir, adapter_dir):
        processor_configs = [
            directory / "processor_config.json",
            directory / "preprocessor_config.json",
        ]
        if not any(path.is_file() and path.stat().st_size > 0 for path in processor_configs):
            errors.append(
                f"Missing processor configuration in {directory}; expected "
                "processor_config.json or preprocessor_config.json"
            )

    base_weights = sorted(base_dir.glob("*.safetensors"))
    if not base_weights:
        errors.append(f"No base-model safetensors found in {base_dir}")

    symlinks = sorted(path for path in algorithm_root.rglob("*") if path.is_symlink())
    if symlinks:
        errors.append(
            "Submission resources still contain symlinks: "
            + ", ".join(str(path) for path in symlinks[:20])
        )

    base_config_path = base_dir / "config.json"
    base_config = load_json(base_config_path, errors) if base_config_path.is_file() else {}
    model_type = str(base_config.get("model_type", ""))
    architectures = [str(item) for item in base_config.get("architectures", [])]
    if "qwen3_5" not in model_type.lower() and not any(
        "qwen3_5" in architecture.lower() for architecture in architectures
    ):
        errors.append(
            "Base config does not identify Qwen3.5: "
            f"model_type={model_type!r}, architectures={architectures!r}"
        )

    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config = (
        load_json(adapter_config_path, errors) if adapter_config_path.is_file() else {}
    )
    if adapter_config and str(adapter_config.get("peft_type", "")).upper() != "LORA":
        errors.append(
            f"Expected a LoRA adapter, got peft_type={adapter_config.get('peft_type')!r}"
        )

    base_bytes = directory_size(base_dir) if base_dir.is_dir() else 0
    adapter_bytes = directory_size(adapter_dir) if adapter_dir.is_dir() else 0
    result = {
        "algorithm_root": str(algorithm_root),
        "base_model": {
            "path": str(base_dir),
            "bytes": base_bytes,
            "gib": round(base_bytes / 1024**3, 3),
            "weight_files": [path.name for path in base_weights],
            "model_type": model_type,
            "architectures": architectures,
        },
        "adapter": {
            "path": str(adapter_dir),
            "bytes": adapter_bytes,
            "mib": round(adapter_bytes / 1024**2, 3),
            "peft_type": adapter_config.get("peft_type"),
            "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        },
        "symlink_count": len(symlinks),
        "status": "ok" if not errors else "error",
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
