"""Validate and package a completed Qwen3.5 LoRA training run for transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
)
PROCESSOR_CONFIG_CANDIDATES = (
    "processor_config.json",
    "preprocessor_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Completed training run directory")
    parser.add_argument("--output-dir", required=True, help="Transfer bundle directory")
    parser.add_argument(
        "--archive-name",
        default="qwen35-lora-e5-final.tar.gz",
        help="Adapter archive filename",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace files from an earlier packaging attempt",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not read {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing or empty required file: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Output already exists (use --force to replace): {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = run_root / "training_summary.json"
    adapter_dir = run_root / "adapter-final"
    history_path = run_root / "train_history.jsonl"

    summary = load_json(summary_path)
    if summary.get("status") != "completed":
        raise SystemExit(
            f"Training run is not completed: status={summary.get('status')!r}"
        )
    if not adapter_dir.is_dir():
        raise SystemExit(f"Missing final adapter directory: {adapter_dir}")
    for filename in REQUIRED_ADAPTER_FILES:
        require_nonempty(adapter_dir / filename)
    if not any(
        (adapter_dir / filename).is_file()
        and (adapter_dir / filename).stat().st_size > 0
        for filename in PROCESSOR_CONFIG_CANDIDATES
    ):
        raise SystemExit(
            "Missing processor configuration; expected one of: "
            + ", ".join(PROCESSOR_CONFIG_CANDIDATES)
        )
    require_nonempty(history_path)

    archive_path = output_dir / args.archive_name
    checksum_path = output_dir / f"{args.archive_name}.sha256"
    copied_summary = output_dir / summary_path.name
    copied_history = output_dir / history_path.name
    manifest_path = output_dir / "artifact_manifest.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        archive_path,
        checksum_path,
        copied_summary,
        copied_history,
        manifest_path,
    ):
        prepare_output(path, args.force)

    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(adapter_dir.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(adapter_dir))

    shutil.copy2(summary_path, copied_summary)
    shutil.copy2(history_path, copied_history)
    archive_digest = sha256(archive_path)
    checksum_path.write_text(
        f"{archive_digest}  {archive_path.name}\n",
        encoding="ascii",
    )

    files = []
    for path in (archive_path, copied_summary, copied_history):
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "source_run": str(run_root),
        "training": summary,
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = {
        "status": "ok",
        "output_dir": str(output_dir),
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_digest,
        "manifest": str(manifest_path),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
