"""Audit SFT JSONL rows against timestamp-overlay video durations.

This script checks whether each sample's start/end time window can be cut from
its referenced overlay video. It writes clean and invalid JSONL files so later
LoRA-SFT runs can use a reproducible, clip-valid manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import decord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        action="append",
        required=True,
        help="Input SFT JSONL path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="~/workspace/focus-runs/data-audit/clip-window-audit",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def classify_row(
    row: dict[str, Any],
    metadata_cache: dict[Path, tuple[int, float]],
) -> tuple[bool, dict[str, Any]]:
    video_path = Path(row["overlay_video_path"])
    frames = 0
    fps = 0.0
    reason = ""

    if not video_path.exists():
        reason = "missing_video"
    else:
        if video_path not in metadata_cache:
            vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
            fps = float(vr.get_avg_fps())
            if fps <= 0:
                fps = 25.0
            metadata_cache[video_path] = (len(vr), fps)
            del vr
        frames, fps = metadata_cache[video_path]

    start_frame = round(float(row["start_time"]) * fps) if fps else 0
    end_frame = round(float(row["end_time"]) * fps) if fps else 0

    if not reason:
        if frames <= 0:
            reason = "empty_video"
        elif start_frame >= frames:
            reason = "start_beyond_video"
        elif end_frame < 0:
            reason = "end_before_video"
        elif end_frame < start_frame:
            reason = "end_before_start"

    audit = dict(row)
    audit.update(
        {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": frames,
            "base_fps": fps,
            "video_path": str(video_path),
        }
    )
    if reason:
        audit["invalid_reason"] = reason
        return False, audit
    return True, audit


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_cache: dict[Path, tuple[int, float]] = {}
    summary: list[dict[str, Any]] = []

    for input_arg in args.input_jsonl:
        input_path = Path(input_arg).expanduser().resolve()
        rows = read_jsonl(input_path)
        valid_rows: list[dict[str, Any]] = []
        invalid_rows: list[dict[str, Any]] = []

        for row in rows:
            is_valid, audited = classify_row(row, metadata_cache)
            if is_valid:
                valid_rows.append(row)
            else:
                invalid_rows.append(audited)

        stem = input_path.stem
        clean_path = output_dir / f"{stem}.clip_valid.jsonl"
        invalid_path = output_dir / f"{stem}.invalid_clips.jsonl"
        write_jsonl(clean_path, valid_rows)
        write_jsonl(invalid_path, invalid_rows)

        summary.append(
            {
                "input_jsonl": str(input_path),
                "total_rows": len(rows),
                "valid_rows": len(valid_rows),
                "invalid_rows": len(invalid_rows),
                "clean_jsonl": str(clean_path),
                "invalid_jsonl": str(invalid_path),
            }
        )

    write_json(output_dir / "clip_window_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
