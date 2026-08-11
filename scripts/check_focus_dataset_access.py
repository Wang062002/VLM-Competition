"""Check which FOCUS datasets/splits are visible to the current environment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from focus import FocusConfig, set_config
from focus.data.base_dataset import FocusDataset
from focus.enums import DatasetSplit, Track


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="/home/Jiali_Wang/data/focus")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset name. Repeat or pass comma-separated values. Defaults to heico,lapchole.",
    )
    parser.add_argument("--track", default="segment", choices=["frame", "segment", "procedure"])
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def parse_datasets(values: list[str] | None) -> list[str]:
    if not values:
        return ["heico", "lapchole"]
    datasets: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item and item not in datasets:
                datasets.append(item)
    return datasets


def enum_track(value: str) -> Track:
    return {
        "frame": Track.FRAME,
        "segment": Track.SEGMENT,
        "procedure": Track.PROCEDURE,
    }[value]


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    set_config(FocusConfig(root_dir=str(root_dir)))

    rows: list[dict[str, Any]] = []
    track = enum_track(args.track)
    for dataset_name in parse_datasets(args.dataset):
        for split in [DatasetSplit.TRAIN, DatasetSplit.TEST]:
            row: dict[str, Any] = {
                "dataset": dataset_name,
                "track": args.track,
                "split": split.value,
                "status": "unknown",
                "samples": None,
                "error_type": "",
                "error": "",
            }
            try:
                dataset = FocusDataset(dataset_name, split, track)
            except Exception as exc:  # noqa: BLE001 - report exact access/loader failure.
                row.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            else:
                row.update({"status": "ok", "samples": len(dataset)})
            rows.append(row)

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["dataset", "track", "split", "status", "samples", "error_type", "error"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if value is None else value for key, value in row.items()})

    if args.json_output:
        output = Path(args.json_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
