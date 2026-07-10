"""Audit ORena FOCUS SEGMENT splits and create a leakage-safe train/val split.

Run this on the remote server inside the `orena-focus` conda environment. The
script uses the official HeiCo SEGMENT TRAIN split for SFT data construction and
keeps the official TEST split as held-out evaluation data.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from focus import FocusConfig, set_config
from focus.data.base_dataset import FocusDataset
from focus.enums import DatasetSplit, Track


@dataclass(frozen=True)
class SampleRecord:
    official_split: str
    qID: str
    videoID: str
    start_time: float
    end_time: float
    procedure_type: str
    question: str
    answer: str
    primary: str
    answer_format: str
    clinical: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="/home/Jiali_Wang/data/focus")
    parser.add_argument("--dataset", default="heico")
    parser.add_argument("--track", default="segment", choices=["segment"])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument(
        "--output-dir",
        default="~/workspace/focus-runs/data-audit/segment-trainval-seed20260707",
    )
    parser.add_argument(
        "--include-test-jsonl",
        action="store_true",
        help="Also write a held-out test manifest JSONL. This is for analysis only, not training.",
    )
    return parser.parse_args()


def norm(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    if hasattr(value, "name"):
        return str(getattr(value, "name"))
    return str(value)


def answer_format(reference: object) -> str:
    fmt = getattr(reference, "format", None)
    if fmt is not None and hasattr(fmt, "type"):
        return str(fmt.type)
    raw = getattr(reference, "_format", None)
    if raw is not None:
        return str(raw)
    return norm(fmt)


def load_records(dataset_name: str, split: DatasetSplit) -> list[SampleRecord]:
    dataset = FocusDataset(dataset_name, split, Track.SEGMENT)
    rows: list[SampleRecord] = []
    for request, reference in dataset:
        rows.append(
            SampleRecord(
                official_split=split.value,
                qID=str(request.qID),
                videoID=str(request.videoID),
                start_time=float(request.start_time),
                end_time=float(request.end_time),
                procedure_type=str(request.procedure_type),
                question=str(request.question),
                answer=str(reference.answer),
                primary=norm(reference.primary),
                answer_format=answer_format(reference),
                clinical=bool(reference.clinical),
            )
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_by(records: list[SampleRecord], *fields: str) -> list[dict[str, object]]:
    counter: Counter[tuple[object, ...]] = Counter()
    for record in records:
        counter[tuple(getattr(record, field) for field in fields)] += 1

    rows: list[dict[str, object]] = []
    for key, count in sorted(counter.items()):
        row = {field: value for field, value in zip(fields, key)}
        row["count"] = count
        rows.append(row)
    return rows


def make_stratified_split(
    train_records: list[SampleRecord],
    val_fraction: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")

    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[SampleRecord]] = defaultdict(list)
    for record in train_records:
        groups[(record.primary, record.answer_format)].append(record)

    internal_train: list[SampleRecord] = []
    internal_val: list[SampleRecord] = []

    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        n_val = int(round(len(group) * val_fraction))
        if len(group) > 1:
            n_val = max(1, n_val)
        n_val = min(n_val, max(len(group) - 1, 0))
        internal_val.extend(group[:n_val])
        internal_train.extend(group[n_val:])

    internal_train.sort(key=lambda row: row.qID)
    internal_val.sort(key=lambda row: row.qID)
    return internal_train, internal_val


def to_manifest_row(record: SampleRecord, internal_split: str | None = None) -> dict[str, object]:
    row = asdict(record)
    if internal_split is not None:
        row["internal_split"] = internal_split
    row["duration"] = round(record.end_time - record.start_time, 6)
    row["raw_video_path"] = f"/home/Jiali_Wang/data/focus/heico/videos/{record.videoID}"
    row["overlay_video_path"] = (
        "/home/Jiali_Wang/data/focus/heico/overlayed/"
        f"{Path(record.videoID).stem}_overlay.mp4"
    )
    return row


def to_sft_row(record: SampleRecord, internal_split: str) -> dict[str, object]:
    return {
        "qID": record.qID,
        "internal_split": internal_split,
        "videoID": record.videoID,
        "start_time": record.start_time,
        "end_time": record.end_time,
        "video_source": "overlay",
        "overlay_video_path": (
            "/home/Jiali_Wang/data/focus/heico/overlayed/"
            f"{Path(record.videoID).stem}_overlay.mp4"
        ),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": record.videoID},
                    {"type": "text", "text": record.question},
                ],
            },
            {"role": "assistant", "content": record.answer},
        ],
        "metadata": {
            "dataset": "heico",
            "track": "segment",
            "official_split": record.official_split,
            "primary": record.primary,
            "answer_format": record.answer_format,
            "clinical": record.clinical,
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_config(FocusConfig(root_dir=str(Path(args.root_dir).expanduser())))

    train_records = load_records(args.dataset, DatasetSplit.TRAIN)
    test_records = load_records(args.dataset, DatasetSplit.TEST)
    all_records = train_records + test_records
    internal_train, internal_val = make_stratified_split(
        train_records,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    split_rows = []
    for split_name, records in [
        ("official_train", train_records),
        ("official_test", test_records),
        ("internal_train", internal_train),
        ("internal_val", internal_val),
    ]:
        split_rows.append(
            {
                "split": split_name,
                "count": len(records),
                "unique_videos": len({record.videoID for record in records}),
            }
        )

    write_csv(output_dir / "split_counts.csv", ["split", "count", "unique_videos"], split_rows)
    write_csv(
        output_dir / "distribution_by_official_split_primary.csv",
        ["official_split", "primary", "count"],
        count_by(all_records, "official_split", "primary"),
    )
    write_csv(
        output_dir / "distribution_by_official_split_answer_format.csv",
        ["official_split", "answer_format", "count"],
        count_by(all_records, "official_split", "answer_format"),
    )
    write_csv(
        output_dir / "distribution_by_official_split_primary_answer_format.csv",
        ["official_split", "primary", "answer_format", "count"],
        count_by(all_records, "official_split", "primary", "answer_format"),
    )

    train_manifest_rows = [
        to_manifest_row(record, "train") for record in internal_train
    ] + [to_manifest_row(record, "val") for record in internal_val]
    write_csv(
        output_dir / "train_internal_split_manifest.csv",
        [
            "official_split",
            "qID",
            "videoID",
            "start_time",
            "end_time",
            "procedure_type",
            "question",
            "answer",
            "primary",
            "answer_format",
            "clinical",
            "internal_split",
            "duration",
            "raw_video_path",
            "overlay_video_path",
        ],
        train_manifest_rows,
    )
    write_jsonl(
        output_dir / "sft_train_overlay.jsonl",
        (to_sft_row(record, "train") for record in internal_train),
    )
    write_jsonl(
        output_dir / "sft_val_overlay.jsonl",
        (to_sft_row(record, "val") for record in internal_val),
    )
    if args.include_test_jsonl:
        write_jsonl(
            output_dir / "heldout_test_manifest.jsonl",
            (to_manifest_row(record) for record in test_records),
        )

    summary = {
        "dataset": args.dataset,
        "track": args.track,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "official_train_samples": len(train_records),
        "official_test_samples": len(test_records),
        "internal_train_samples": len(internal_train),
        "internal_val_samples": len(internal_val),
        "output_dir": str(output_dir),
        "policy": (
            "Only official TRAIN samples are used for LoRA-SFT. Official TEST "
            "samples remain held out for final local evaluation."
        ),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote audit files to: {output_dir}")


if __name__ == "__main__":
    main()
