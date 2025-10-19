"""Backtest a Supertrend strategy on intraday OHLC data."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import List, Sequence


def load_ohlc(path: str) -> tuple[List[str], List[float], List[float], List[float], List[float]]:
    """Load timestamped OHLC data from a CSV file."""

    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            raise ValueError("CSV file must include a header row")

        normalized = {name.strip().lower(): name for name in reader.fieldnames}

        required = ["time", "open", "high", "low", "close"]
        missing = [col for col in required if col not in normalized]
        if missing:
            raise KeyError(f"Missing expected columns: {', '.join(missing)}")

        times: List[str] = []
        opens: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []

        for row in reader:
            try:
                times.append(row[normalized["time"]])
                opens.append(float(row[normalized["open"]]))
                highs.append(float(row[normalized["high"]]))
                lows.append(float(row[normalized["low"]]))
                closes.append(float(row[normalized["close"]]))
            except (TypeError, ValueError):
                # Skip rows with malformed numeric data
                continue

    if not times:
        raise ValueError("No valid OHLC rows were parsed from the CSV file")

    return times, opens, highs, lows, closes


def compute_atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int) -> List[float | None]:
    """Calculate the Average True Range (ATR) using Wilder's smoothing."""

    if period <= 0:
        raise ValueError("ATR period must be positive")

    tr_values: List[float] = []
    for idx, (high, low) in enumerate(zip(highs, lows)):
        if idx == 0:
            tr_values.append(high - low)
            continue
        prev_close = closes[idx - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)

    atr: List[float | None] = [None] * len(tr_values)
    if len(tr_values) < period:
        return atr

    initial_atr = sum(tr_values[:period]) / period
    atr[period - 1] = initial_atr

    for idx in range(period, len(tr_values)):
        prev_atr = atr[idx - 1]
        if prev_atr is None:
            raise RuntimeError("Unexpected None ATR value during smoothing")
        atr[idx] = (prev_atr * (period - 1) + tr_values[idx]) / period

    return atr


@dataclass
class SupertrendPoint:
    timestamp: str
    close: float
    supertrend: float
    trend: int  # 1 for bullish, -1 for bearish


def compute_supertrend(
    timestamps: Sequence[str],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int,
    multiplier: float,
) -> List[SupertrendPoint]:
    """Compute the Supertrend indicator."""

    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError("High, low, and close series must be the same length")

    atr = compute_atr(highs, lows, closes, period)
    points: List[SupertrendPoint] = []

    final_upper: List[float | None] = [None] * len(highs)
    final_lower: List[float | None] = [None] * len(highs)
    supertrend: List[float | None] = [None] * len(highs)
    trend_dir: List[int | None] = [None] * len(highs)

    for idx in range(len(highs)):
        atr_value = atr[idx]
        if atr_value is None:
            continue

        hl2 = (highs[idx] + lows[idx]) / 2.0
        basic_upper = hl2 + multiplier * atr_value
        basic_lower = hl2 - multiplier * atr_value

        if idx == 0 or final_upper[idx - 1] is None:
            final_upper[idx] = basic_upper
            final_lower[idx] = basic_lower
        else:
            prev_final_upper = final_upper[idx - 1]
            prev_final_lower = final_lower[idx - 1]
            prev_close = closes[idx - 1]

            assert prev_final_upper is not None and prev_final_lower is not None

            final_upper[idx] = (
                basic_upper
                if basic_upper < prev_final_upper or prev_close > prev_final_upper
                else prev_final_upper
            )
            final_lower[idx] = (
                basic_lower
                if basic_lower > prev_final_lower or prev_close < prev_final_lower
                else prev_final_lower
            )

        if idx == 0 or supertrend[idx - 1] is None:
            if closes[idx] <= final_upper[idx]:
                supertrend[idx] = final_upper[idx]
                trend_dir[idx] = -1
            else:
                supertrend[idx] = final_lower[idx]
                trend_dir[idx] = 1
        else:
            prev_supertrend = supertrend[idx - 1]
            prev_trend = trend_dir[idx - 1]
            assert prev_supertrend is not None and prev_trend is not None

            if prev_trend == -1:
                if closes[idx] <= final_upper[idx]:
                    supertrend[idx] = final_upper[idx]
                    trend_dir[idx] = -1
                else:
                    supertrend[idx] = final_lower[idx]
                    trend_dir[idx] = 1
            else:
                if closes[idx] >= final_lower[idx]:
                    supertrend[idx] = final_lower[idx]
                    trend_dir[idx] = 1
                else:
                    supertrend[idx] = final_upper[idx]
                    trend_dir[idx] = -1

        points.append(
            SupertrendPoint(
                timestamp=timestamps[idx],
                close=closes[idx],
                supertrend=supertrend[idx],
                trend=trend_dir[idx],
            )
        )

    return points


@dataclass
class BacktestResult:
    initial_capital: float
    final_capital: float
    net_profit: float
    net_return_pct: float
    total_trades: int
    buy_and_hold_return_pct: float


def run_backtest(points: Sequence[SupertrendPoint], initial_capital: float) -> BacktestResult:
    """Execute a simple Supertrend crossover backtest."""

    cash = initial_capital
    position = 0.0  # number of shares
    total_trades = 0
    prev_trend = None

    closes = [pt.close for pt in points]
    if not closes:
        raise ValueError("Supertrend series is empty")

    for point in points:
        if point.trend is None:
            continue

        if prev_trend is None:
            prev_trend = point.trend
            continue

        # Trend flip from bearish to bullish -> buy
        if prev_trend == -1 and point.trend == 1 and position == 0.0:
            price = point.close
            if price <= 0:
                prev_trend = point.trend
                continue
            position = cash / price
            cash = 0.0
            total_trades += 1
        # Trend flip from bullish to bearish -> sell
        elif prev_trend == 1 and point.trend == -1 and position > 0.0:
            cash = position * point.close
            position = 0.0
            total_trades += 1

        prev_trend = point.trend

    final_value = cash + position * closes[-1]
    net_profit = final_value - initial_capital
    net_return_pct = (net_profit / initial_capital) * 100.0

    buy_and_hold_return_pct = ((closes[-1] - closes[0]) / closes[0]) * 100.0

    return BacktestResult(
        initial_capital=initial_capital,
        final_capital=final_value,
        net_profit=net_profit,
        net_return_pct=net_return_pct,
        total_trades=total_trades,
        buy_and_hold_return_pct=buy_and_hold_return_pct,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supertrend backtest for OHLC data")
    parser.add_argument("--data", required=True, help="Path to the CSV file with OHLC data")
    parser.add_argument("--period", type=int, default=10, help="Supertrend ATR period (default: 10)")
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.5,
        help="Supertrend ATR multiplier (default: 1.5)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1000.0,
        help="Initial capital for the backtest (default: 1000)",
    )
    return parser.parse_args()


def format_currency(value: float) -> str:
    return f"$ {value:,.2f}"


def main() -> None:
    args = parse_args()
    timestamps, opens, highs, lows, closes = load_ohlc(args.data)
    supertrend_points = compute_supertrend(
        timestamps,
        highs,
        lows,
        closes,
        period=args.period,
        multiplier=args.multiplier,
    )
    result = run_backtest(supertrend_points, args.capital)

    print("Supertrend Backtest Summary\n")
    print(f"Data file: {args.data}")
    print(f"Period: {args.period}")
    print(f"Multiplier: {args.multiplier}")
    print(f"Initial capital: {format_currency(result.initial_capital)}")
    print(f"Final capital:   {format_currency(result.final_capital)}")
    print(f"Net profit:      {format_currency(result.net_profit)}")
    print(f"Return:          {result.net_return_pct:.2f}%")
    print(f"Buy & Hold:      {result.buy_and_hold_return_pct:.2f}%")
    print(f"Total trades:    {result.total_trades}")


if __name__ == "__main__":
    main()
