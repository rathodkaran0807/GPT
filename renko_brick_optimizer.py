"""Evaluate optimal Renko brick size for 5-minute closing prices.

This module reads intraday OHLC data (specifically the close column) and
constructs Renko charts for a sweep of candidate brick sizes. Each candidate is
scored using a simple trend persistence heuristic that rewards brick sizes which
produce numerous bricks while also minimizing noisy back-and-forth reversals.

Example
-------
$ python renko_brick_optimizer.py --data 'BATS_CSCO, 5.csv'
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class RenkoBrickStats:
    """Summary metrics for a single Renko brick size."""

    brick_size: float
    brick_count: int
    reversal_count: int
    trending_ratio: float
    total_move: float
    net_move: float
    score: float

def build_renko_bricks(prices: Iterable[float], brick_size: float) -> List[float]:
    """Construct Renko bricks from a sequence of prices.

    Parameters
    ----------
    prices:
        Iterable of closing prices.
    brick_size:
        Positive float representing the price move required to form a brick.

    Returns
    -------
    List[float]
        A list containing the signed direction of each Renko brick (+1 or -1).
    """

    if brick_size <= 0:
        raise ValueError("brick_size must be positive")

    iterator = iter(prices)
    try:
        last_close = float(next(iterator))
    except StopIteration:
        return []

    brick_directions: List[float] = []
    last_brick_close = last_close

    for price in iterator:
        price = float(price)
        delta = price - last_brick_close

        while delta >= brick_size:
            last_brick_close += brick_size
            brick_directions.append(1.0)
            delta -= brick_size

        while delta <= -brick_size:
            last_brick_close -= brick_size
            brick_directions.append(-1.0)
            delta += brick_size

    return brick_directions


def score_brick_size(prices: Iterable[float], brick_size: float) -> RenkoBrickStats:
    """Compute summary statistics for a Renko brick size."""

    bricks = build_renko_bricks(prices, brick_size)
    brick_count = len(bricks)

    if brick_count == 0:
        return RenkoBrickStats(
            brick_size=brick_size,
            brick_count=0,
            reversal_count=0,
            trending_ratio=0.0,
            total_move=0.0,
            net_move=0.0,
            score=0.0,
        )

    reversal_count = sum(
        1 for prev, curr in zip(bricks[:-1], bricks[1:]) if curr != prev
    )

    trending_ratio = 1.0
    if brick_count > 1:
        trending_ratio = 1.0 - (reversal_count / (brick_count - 1))

    total_move = brick_count * brick_size
    net_move = sum(bricks) * brick_size
    score = trending_ratio * total_move

    return RenkoBrickStats(
        brick_size=brick_size,
        brick_count=brick_count,
        reversal_count=reversal_count,
        trending_ratio=trending_ratio,
        total_move=total_move,
        net_move=net_move,
        score=score,
    )


def generate_candidate_bricks(
    closes: List[float],
    min_brick: float | None,
    max_brick: float | None,
    steps: int,
) -> List[float]:
    """Create a range of candidate brick sizes."""

    price_range = max(closes) - min(closes)
    if price_range <= 0:
        raise ValueError("Closing prices must span a non-zero range")

    if min_brick is None:
        min_brick = price_range / 200.0

    if max_brick is None:
        max_brick = price_range / 10.0

    if min_brick <= 0 or max_brick <= 0:
        raise ValueError("Brick sizes must be positive")

    if min_brick >= max_brick:
        raise ValueError("min_brick must be smaller than max_brick")

    if steps < 2:
        raise ValueError("steps must be at least 2")

    step = (max_brick - min_brick) / (steps - 1)
    return [min_brick + i * step for i in range(steps)]


def load_closing_prices(path: str) -> List[float]:
    """Load closing prices from the provided CSV file."""

    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        if reader.fieldnames is None:
            raise ValueError("CSV file is missing a header row")

        normalized = {name.strip().lower(): name for name in reader.fieldnames}

        close_col_name = None
        for candidate in ("close", "closing"):
            if candidate in normalized:
                close_col_name = normalized[candidate]
                break

        if close_col_name is None:
            raise KeyError(
                "Unable to find a closing price column. Expected one named 'close' or 'closing'."
            )

        closes: List[float] = []
        for row in reader:
            raw_value = row.get(close_col_name)
            if raw_value is None:
                continue

            try:
                closes.append(float(raw_value))
            except (TypeError, ValueError):
                continue

    if not closes:
        raise ValueError("No valid closing prices found in the dataset")

    return closes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Renko brick sizes for 5-minute closing data",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the CSV file containing 5-minute OHLC data",
    )
    parser.add_argument(
        "--min-brick",
        type=float,
        default=None,
        help="Minimum brick size to evaluate. Defaults to price_range / 200.",
    )
    parser.add_argument(
        "--max-brick",
        type=float,
        default=None,
        help="Maximum brick size to evaluate. Defaults to price_range / 10.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of brick sizes to evaluate between min and max (inclusive).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top-performing brick sizes to display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    closes = load_closing_prices(args.data)
    candidates = generate_candidate_bricks(closes, args.min_brick, args.max_brick, args.steps)

    stats: List[RenkoBrickStats] = [
        score_brick_size(closes, float(brick_size)) for brick_size in candidates
    ]

    stats.sort(key=lambda item: item.score, reverse=True)

    top_n = stats[: max(1, min(args.top, len(stats)))]

    header = (
        f"{'brick_size':>12}  {'bricks':>8}  {'reversals':>10}  "
        f"{'trend_ratio':>12}  {'total_move':>12}  {'net_move':>10}  {'score':>12}"
    )
    print("Top brick sizes by Renko trend score:\n")
    print(header)
    print("-" * len(header))
    for stat in top_n:
        print(
            f"{stat.brick_size:12.6f}  "
            f"{stat.brick_count:8d}  "
            f"{stat.reversal_count:10d}  "
            f"{stat.trending_ratio:12.6f}  "
            f"{stat.total_move:12.6f}  "
            f"{stat.net_move:10.6f}  "
            f"{stat.score:12.6f}"
        )

    best = stats[0]
    print()
    print(
        "Suggested brick size:"
        f" {best.brick_size:.6f} (score={best.score:.6f},"
        f" bricks={best.brick_count},"
        f" reversals={best.reversal_count})"
    )


if __name__ == "__main__":
    main()
