"""Print evaluator-style summary tables from a long CSV file.

The expected CSV schema is:

    experiment,level,name,accuracy,ci_low,ci_high,count

By default this prints every experiment in
`results/evaluator_style_full_4000_summaries.csv` using a terminal layout close
to the official ORena FOCUS evaluator output.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT = "results/evaluator_style_full_4000_summaries.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="CSV with columns: experiment, level, name, accuracy, ci_low, ci_high, count.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Experiment name to print. Can be passed multiple times. Defaults to all.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimals for accuracy and confidence intervals.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional text file path. If omitted, print to stdout.",
    )
    parser.add_argument(
        "--no-separators",
        action="store_true",
        help="Do not print separator lines between experiments.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> OrderedDict[str, list[dict[str, str]]]:
    required = {"experiment", "level", "name", "accuracy", "ci_low", "ci_high", "count"}
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required CSV columns: {sorted(missing)}")
        for row in reader:
            experiment = row["experiment"]
            grouped.setdefault(experiment, []).append(row)
    return grouped


def format_float(value: str, precision: int) -> str:
    if value is None or value == "":
        return "NaN"
    return f"{float(value):.{precision}f}"


def format_count(value: str) -> str:
    if value is None or value == "":
        return ""
    return str(int(float(value)))


def table_lines(rows: Iterable[dict[str, str]], precision: int) -> list[str]:
    rendered = []
    for row in rows:
        rendered.append(
            {
                "level": row["level"],
                "name": row["name"],
                "accuracy": format_float(row["accuracy"], precision),
                "ci_low": format_float(row["ci_low"], precision),
                "ci_high": format_float(row["ci_high"], precision),
                "count": format_count(row["count"]),
            }
        )

    columns = ["level", "name", "accuracy", "ci_low", "ci_high", "count"]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rendered))
        for column in columns
    }

    lines = [
        (
            f"{'level':>{widths['level']}} "
            f"{'name':>{widths['name']}} "
            f"{'accuracy':>{widths['accuracy']}} "
            f"{'ci_low':>{widths['ci_low']}} "
            f"{'ci_high':>{widths['ci_high']}} "
            f"{'count':>{widths['count']}}"
        )
    ]
    for row in rendered:
        lines.append(
            (
                f"{row['level']:>{widths['level']}} "
                f"{row['name']:>{widths['name']}} "
                f"{row['accuracy']:>{widths['accuracy']}} "
                f"{row['ci_low']:>{widths['ci_low']}} "
                f"{row['ci_high']:>{widths['ci_high']}} "
                f"{row['count']:>{widths['count']}}"
            )
        )
    return lines


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    grouped = read_rows(input_path)

    selected = args.experiment or list(grouped)
    missing = [name for name in selected if name not in grouped]
    if missing:
        available = ", ".join(grouped)
        raise SystemExit(f"Unknown experiment(s): {missing}. Available: {available}")

    output_lines: list[str] = []
    for index, experiment in enumerate(selected):
        if not args.no_separators:
            if index:
                output_lines.append("")
            output_lines.append("=" * 100)
            output_lines.append(experiment)
            output_lines.append("=" * 100)
        output_lines.extend(table_lines(grouped[experiment], args.precision))

    text = "\n".join(output_lines)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
