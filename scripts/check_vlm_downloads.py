"""Check local Hugging Face snapshots for candidate VLM downloads.

This is a lightweight integrity check before running smoke/batch evaluation.
It does not load model weights, so it is safe to run on the remote server even
when GPUs are busy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "configs/vlm_candidate_models.csv"
DEFAULT_MODEL_DIR = "~/workspace/vlm-models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--json-output", default=None)
    parser.add_argument(
        "--min-size-gb",
        type=float,
        default=1.0,
        help="Minimum directory size to consider a model snapshot non-empty.",
    )
    return parser.parse_args()


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def dir_size(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def count_files(path: Path) -> int:
    return sum(1 for file_path in path.rglob("*") if file_path.is_file())


def find_weight_files(path: Path) -> list[Path]:
    suffixes = {".safetensors", ".bin", ".pt"}
    return [file_path for file_path in path.rglob("*") if file_path.suffix in suffixes]


def read_manifest(model_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = model_dir / "download_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, list):
        return {}
    return {str(item.get("key")): item for item in payload if isinstance(item, dict)}


def check_model(row: dict[str, str], model_dir: Path, manifest: dict[str, dict[str, Any]], min_size: int) -> dict[str, Any]:
    key = row["key"]
    repo_id = row["repo_id"]
    target_dir = model_dir / safe_dir_name(repo_id)
    result: dict[str, Any] = {
        "key": key,
        "repo_id": repo_id,
        "target_dir": str(target_dir),
        "exists": target_dir.exists(),
        "status": "missing",
        "size_gb": 0.0,
        "file_count": 0,
        "has_config": False,
        "weight_file_count": 0,
        "manifest_status": manifest.get(key, {}).get("status"),
    }
    if not target_dir.exists():
        result["problems"] = ["target directory does not exist"]
        return result

    size_bytes = dir_size(target_dir)
    files = count_files(target_dir)
    weights = find_weight_files(target_dir)
    has_config = (target_dir / "config.json").exists()
    problems = []
    if size_bytes < min_size:
        problems.append(f"directory smaller than {min_size / (1024**3):.2f} GB")
    if files == 0:
        problems.append("no files found")
    if not has_config:
        problems.append("config.json not found")
    if not weights:
        problems.append("no weight files found")

    result.update(
        {
            "status": "ok" if not problems else "problem",
            "size_gb": round(size_bytes / (1024**3), 3),
            "file_count": files,
            "has_config": has_config,
            "weight_file_count": len(weights),
            "largest_weight_gb": round(max((p.stat().st_size for p in weights), default=0) / (1024**3), 3),
            "problems": problems,
        }
    )
    return result


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    min_size = int(args.min_size_gb * 1024**3)

    candidates = read_candidates(config_path)
    manifest = read_manifest(model_dir)
    results = [check_model(row, model_dir, manifest, min_size) for row in candidates]

    headers = [
        "key",
        "status",
        "size_gb",
        "file_count",
        "has_config",
        "weight_file_count",
        "manifest_status",
    ]
    print(" ".join(f"{header:>22}" for header in headers))
    for item in results:
        print(
            " ".join(
                f"{str(item.get(header, '')):>22}"
                for header in headers
            )
        )
        if item.get("problems"):
            print("  problems:", "; ".join(item["problems"]))

    if args.json_output:
        out = Path(args.json_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("json_output:", out)

    failed = [item for item in results if item["status"] != "ok"]
    if failed:
        raise SystemExit(f"{len(failed)} model download(s) need attention.")


if __name__ == "__main__":
    main()
