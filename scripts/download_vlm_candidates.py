"""Download candidate open-source VLMs for ORena FOCUS model comparison.

The script reads `configs/vlm_candidate_models.csv` by default and downloads
selected Hugging Face repositories with `huggingface_hub.snapshot_download`.

Run on the remote server inside the `orena-focus` environment after `hf auth
login`. Some models, especially Google Gemma/MedGemma, may require accepting
their Hugging Face license terms in the browser before download succeeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "configs/vlm_candidate_models.csv"
DEFAULT_OUTPUT_DIR = "~/workspace/vlm-models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Candidate key to download. Can be passed multiple times. Defaults to all.",
    )
    parser.add_argument("--revision", default=None, help="Optional HF revision for all downloads.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads only.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a target when its directory already contains files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed downloads and continue with the remaining models.",
    )
    parser.add_argument(
        "--disable-xet",
        action="store_true",
        help="Set HF_HUB_DISABLE_XET=1 before importing huggingface_hub.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest JSON path. Defaults to <output-dir>/download_manifest.json.",
    )
    return parser.parse_args()


def safe_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else output_dir / "download_manifest.json"
    )

    candidates = read_candidates(config_path)
    by_key = {row["key"]: row for row in candidates}
    selected_keys = args.model or list(by_key)
    missing = [key for key in selected_keys if key not in by_key]
    if missing:
        raise SystemExit(f"Unknown candidate key(s): {missing}. Available: {sorted(by_key)}")

    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    records: list[dict[str, Any]] = []
    for key in selected_keys:
        row = by_key[key]
        repo_id = row["repo_id"]
        target_dir = output_dir / safe_dir_name(repo_id)
        record: dict[str, Any] = {
            "key": key,
            "repo_id": repo_id,
            "engine": row.get("engine"),
            "target_dir": str(target_dir),
            "revision": args.revision,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "planned" if args.dry_run else "pending",
        }

        print(f"\n== {key}: {repo_id} ==")
        print(f"target: {target_dir}")
        if args.dry_run:
            records.append(record)
            continue

        if target_dir.exists() and any(target_dir.iterdir()) and args.skip_existing:
            print("skip: target already exists; omit --skip-existing to verify/resume")
            record["status"] = "skipped_existing"
            records.append(record)
            continue

        try:
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(
                repo_id=repo_id,
                revision=args.revision,
                local_dir=str(target_dir),
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
            records.append(record)
            write_json(manifest_path, records)
            print(f"failed: {exc!r}")
            print("manifest updated before exit:", manifest_path)
            if args.continue_on_error:
                continue
            raise

        record["status"] = "downloaded"
        record["local_path"] = str(local_path)
        record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        records.append(record)
        write_json(manifest_path, records)
        print(f"downloaded: {local_path}")

    if not args.dry_run:
        write_json(manifest_path, records)
        print("\nManifest:", manifest_path)
    else:
        print("\nDry run complete. No files were downloaded or written.")
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
