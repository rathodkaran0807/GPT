"""Utility to inspect the `BATS_CSCO, 5.csv` dataset.

The script provides:
- Basic metadata (row count, column names, time span)
- Summary statistics for OHLC columns
- Counts of NaN values per column

Usage::

    python analyze_bats_csco.py

The script prints results to stdout. It can optionally save a JSON report using
`--json <path>`.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DATA_FILE = Path("BATS_CSCO, 5.csv")
OHLC_COLUMNS = ("open", "high", "low", "close")


@dataclass
class DatasetSummary:
    rows: int
    columns: int
    start_time: str
    end_time: str
    column_names: List[str]
    nan_counts: Dict[str, int]
    ohlc_summary: Dict[str, Dict[str, float]]


def load_dataset(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return reader.fieldnames or [], rows


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def summarize_dataset(columns: List[str], rows: List[Dict[str, str]]) -> DatasetSummary:
    if not rows:
        raise ValueError("Dataset is empty")

    start_time = parse_time(rows[0]["time"])
    end_time = parse_time(rows[-1]["time"])

    nan_counts = {col: 0 for col in columns}
    ohlc_values = {col: [] for col in OHLC_COLUMNS}

    for row in rows:
        for col in columns:
            if row[col] in {"", "NaN", "nan"}:
                nan_counts[col] += 1

        for col in OHLC_COLUMNS:
            value = row[col]
            if value not in {"", "NaN", "nan"}:
                ohlc_values[col].append(float(value))

    ohlc_summary = {
        col: compute_summary(values)
        for col, values in ohlc_values.items()
    }

    return DatasetSummary(
        rows=len(rows),
        columns=len(columns),
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        column_names=columns,
        nan_counts=nan_counts,
        ohlc_summary=ohlc_summary,
    )


def compute_summary(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {key: float("nan") for key in ("min", "max", "mean", "std")}

    total = sum(values)
    mean = total / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": sqrt(variance),
    }


def format_summary(summary: DatasetSummary) -> str:
    lines = [
        "BATS_CSCO dataset summary",
        "-------------------------",
        f"Rows: {summary.rows}",
        f"Columns: {summary.columns}",
        f"Time span: {summary.start_time} -> {summary.end_time}",
        "",
        "Columns:",
    ]
    lines.extend(f"  - {name}" for name in summary.column_names)
    lines.append("")
    lines.append("NaN counts:")
    lines.extend(f"  - {col}: {count}" for col, count in summary.nan_counts.items())
    lines.append("")
    lines.append("OHLC summary statistics:")
    for col, stats in summary.ohlc_summary.items():
        stats_str = ", ".join(f"{key}={value:.6f}" for key, value in stats.items())
        lines.append(f"  - {col}: {stats_str}")
    return "\n".join(lines)


def write_json(summary: DatasetSummary, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(summary), fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to save the summary as JSON.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_FILE,
        help="Path to the CSV dataset (default: BATS_CSCO, 5.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    columns, rows = load_dataset(args.data)
    summary = summarize_dataset(columns, rows)
    print(format_summary(summary))
    if args.json:
        write_json(summary, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
