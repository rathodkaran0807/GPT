"""Discover higher-order directional patterns in the CSCO 5-minute dataset using
an indicator-rich neural network pipeline.

The workflow engineers classic technical indicators (EMA gaps, MACD histogram,
RSI, ATR, and stochastic oscillators) alongside OHLC-derived momentum and
volatility signals. A single-hidden-layer neural network is then trained via
mini-batch gradient descent to predict whether the next bar will close above the
current bar. The script surfaces evaluation metrics, key feature influences, and
scans multi-bar trades driven by bullish signals across configurable stop-loss,
profit-target, and trailing-stop grids to locate the most profitable mix.

Usage (default dataset):

    python machine_learning_patterns.py

Command-line flags allow you to override the dataset path, train/test split,
optimizer hyperparameters, network width, capital, and the stop/target/trailing
grid. Inspect ``python machine_learning_patterns.py --help`` for the full set of
options.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

def safe_value(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


DATA_FILE = Path("BATS_CSCO, 5.csv")


@dataclass
class Dataset:
    features: List[List[float]]
    targets: List[int]
    feature_names: List[str]
    entry_prices: List[float]
    next_prices: List[float]
    row_indices: List[int]
    opens: List[float]
    highs: List[float]
    lows: List[float]
    closes: List[float]


@dataclass
class Standardization:
    means: List[float]
    stds: List[float]


@dataclass
class NeuralNetwork:
    w1: List[List[float]]
    b1: List[float]
    w2: List[float]
    b2: float

    def _forward_layer(self, features: Sequence[float]) -> Tuple[List[float], List[float]]:
        pre_activations: List[float] = []
        activations: List[float] = []
        for h in range(len(self.b1)):
            weighted_sum = sum(features[f] * self.w1[f][h] for f in range(len(features))) + self.b1[h]
            pre_activations.append(weighted_sum)
            activations.append(weighted_sum if weighted_sum > 0 else 0.0)
        return pre_activations, activations

    def predict_proba(self, features: Sequence[float]) -> float:
        _, hidden = self._forward_layer(features)
        logit = sum(hidden[h] * self.w2[h] for h in range(len(self.w2))) + self.b2
        logit = max(min(logit, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-logit))

    def predict_label(self, features: Sequence[float]) -> int:
        return 1 if self.predict_proba(features) >= 0.5 else 0


@dataclass
class Evaluation:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


@dataclass
class TrainingResult:
    model: NeuralNetwork
    standardization: Standardization
    evaluation: Evaluation
    train_count: int
    full_train_count: int
    test_count: int
    test_probabilities: List[float]
    test_predictions: List[int]
    threshold: float
    loss_history: List[float]


@dataclass
class TrainTestSplit:
    train: Dataset
    test: Dataset


@dataclass
class BacktestReport:
    starting_cash: float
    ending_value: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    win_loss_ratio: float | None
    profit: float
    return_pct: float
    average_bars_held: float
    max_bars_held: int
    stop_loss_pct: float | None
    take_profit_pct: float | None
    trailing_stop_pct: float | None


def load_dataset(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def to_float(value: str) -> float:
    if value in {"", "NaN", "nan", "None"}:
        return float("nan")
    return float(value)


def compute_log_return(new: float, old: float) -> float:
    if new <= 0 or old <= 0:
        return 0.0
    return math.log(new / old)


def rolling_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def ema(values: Sequence[float], period: int) -> List[float]:
    if period <= 0:
        raise ValueError("EMA period must be positive")
    ema_values = [float("nan")] * len(values)
    multiplier = 2.0 / (period + 1)
    ema_prev: float | None = None
    for idx, value in enumerate(values):
        if math.isnan(value):
            ema_values[idx] = ema_prev if ema_prev is not None else float("nan")
            continue
        if ema_prev is None:
            ema_prev = value
        else:
            ema_prev = (value - ema_prev) * multiplier + ema_prev
        ema_values[idx] = ema_prev
    return ema_values


def compute_macd(values: Sequence[float]) -> Tuple[List[float], List[float], List[float]]:
    ema_fast = ema(values, 12)
    ema_slow = ema(values, 26)
    macd_line = []
    for fast, slow in zip(ema_fast, ema_slow):
        if math.isnan(fast) or math.isnan(slow):
            macd_line.append(float("nan"))
        else:
            macd_line.append(fast - slow)
    signal_line = ema(macd_line, 9)
    histogram = []
    for macd_value, signal_value in zip(macd_line, signal_line):
        if math.isnan(macd_value) or math.isnan(signal_value):
            histogram.append(float("nan"))
        else:
            histogram.append(macd_value - signal_value)
    return ema_fast, ema_slow, histogram


def compute_rsi(closes: Sequence[float], period: int = 14) -> List[float]:
    rsi_values = [float("nan")] * len(closes)
    avg_gain = avg_loss = 0.0
    gain_count = loss_count = 0
    for idx in range(1, len(closes)):
        change = closes[idx] - closes[idx - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain += gain
        avg_loss += loss
        gain_count += 1
        loss_count += 1
        if idx < period:
            continue
        if idx == period:
            avg_gain /= period
            avg_loss /= period
        else:
            avg_gain = ((period - 1) * avg_gain + gain) / period
            avg_loss = ((period - 1) * avg_loss + loss) / period
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_values[idx] = rsi
    return rsi_values


def compute_atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> List[float]:
    atr_values = [float("nan")] * len(closes)
    prev_close = closes[0] if closes else float("nan")
    tr_values: List[float] = []
    atr_prev: float | None = None
    for idx in range(len(closes)):
        high = highs[idx]
        low = lows[idx]
        close = closes[idx]
        if math.isnan(high) or math.isnan(low) or math.isnan(close):
            tr = 0.0
        elif idx == 0 or math.isnan(prev_close):
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
        if idx < period:
            prev_close = close
            continue
        if idx == period:
            atr_prev = sum(tr_values[1:period + 1]) / period
        elif atr_prev is not None:
            atr_prev = ((period - 1) * atr_prev + tr) / period
        atr = atr_prev if atr_prev is not None else 0.0
        atr_values[idx] = atr
        prev_close = close
    return atr_values


def compute_stochastic(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14, smooth: int = 3) -> Tuple[List[float], List[float]]:
    k_values = [float("nan")] * len(closes)
    d_values = [float("nan")] * len(closes)
    for idx in range(len(closes)):
        if idx < period - 1:
            continue
        window_high = max(highs[idx - period + 1 : idx + 1])
        window_low = min(lows[idx - period + 1 : idx + 1])
        denom = window_high - window_low
        if denom <= 0:
            k = 0.0
        else:
            k = (closes[idx] - window_low) / denom * 100.0
        k_values[idx] = k
        if idx >= period - 1 + smooth - 1:
            recent_k = [value for value in k_values[idx - smooth + 1 : idx + 1] if not math.isnan(value)]
            if recent_k:
                d_values[idx] = sum(recent_k) / len(recent_k)
            else:
                d_values[idx] = 0.0
    return k_values, d_values


def build_features(rows: List[dict]) -> Dataset:
    closes = [to_float(row["close"]) for row in rows]
    opens = [to_float(row["open"]) for row in rows]
    highs = [to_float(row["high"]) for row in rows]
    lows = [to_float(row["low"]) for row in rows]

    ema_fast, ema_slow, macd_hist = compute_macd(closes)
    rsi_values = compute_rsi(closes, 14)
    atr_values = compute_atr(highs, lows, closes, 14)
    stoch_k, stoch_d = compute_stochastic(highs, lows, closes, 14, 3)

    max_lag = 35
    feature_names = [
        "log_return_1",  # immediate momentum
        "log_return_5",  # short-term momentum
        "log_return_10",  # longer swing momentum
        "body_pct",  # candle body as share of open
        "range_pct",  # intrabar volatility
        "volatility_5",  # rolling volatility over 5 bars
        "volatility_14",  # extended volatility regime
        "ema_fast_minus_slow",  # 12-26 EMA spread
        "macd_histogram",  # MACD histogram
        "rsi_14",  # RSI (0-1 scaled)
        "atr_pct",  # ATR as share of price
        "stoch_k",  # stochastic %K (0-1 scaled)
        "stoch_d",  # stochastic %D (0-1 scaled)
        "trend_vs_ema",  # close distance from slow EMA
    ]

    features: List[List[float]] = []
    targets: List[int] = []
    entry_prices: List[float] = []
    next_prices: List[float] = []
    row_indices: List[int] = []

    log_returns: List[float] = [0.0]
    for idx in range(1, len(closes)):
        log_returns.append(compute_log_return(closes[idx], closes[idx - 1]))

    for idx in range(max_lag, len(rows) - 1):
        lr_1 = log_returns[idx]
        lr_5 = compute_log_return(closes[idx], closes[idx - 5]) if idx >= 5 else 0.0
        lr_10 = compute_log_return(closes[idx], closes[idx - 10]) if idx >= 10 else 0.0
        body = closes[idx] - opens[idx]
        body_pct = body / opens[idx] if opens[idx] else 0.0
        range_pct = (highs[idx] - lows[idx]) / closes[idx] if closes[idx] else 0.0
        window_5 = log_returns[idx - 4 : idx + 1]
        window_14 = log_returns[idx - 13 : idx + 1]
        vol_5 = rolling_std(window_5)
        vol_14 = rolling_std(window_14)
        ema_gap = ema_fast[idx] - ema_slow[idx]
        macd_val = macd_hist[idx]
        rsi_val = rsi_values[idx] / 100.0 if not math.isnan(rsi_values[idx]) else 0.0
        atr_pct = (atr_values[idx] / closes[idx]) if closes[idx] and not math.isnan(atr_values[idx]) else 0.0
        stoch_k_val = stoch_k[idx] / 100.0 if not math.isnan(stoch_k[idx]) else 0.0
        stoch_d_val = stoch_d[idx] / 100.0 if not math.isnan(stoch_d[idx]) else 0.0
        ema_slow_val = ema_slow[idx]
        trend_vs_ema = ((closes[idx] - ema_slow_val) / closes[idx]) if closes[idx] and not math.isnan(ema_slow_val) else 0.0

        feature_vector = [
            safe_value(lr_1),
            safe_value(lr_5),
            safe_value(lr_10),
            safe_value(body_pct),
            safe_value(range_pct),
            safe_value(vol_5),
            safe_value(vol_14),
            safe_value(ema_gap),
            safe_value(macd_val),
            safe_value(rsi_val),
            safe_value(atr_pct),
            safe_value(stoch_k_val),
            safe_value(stoch_d_val),
            safe_value(trend_vs_ema),
        ]

        features.append(feature_vector)
        targets.append(1 if closes[idx + 1] > closes[idx] else 0)
        entry_prices.append(closes[idx])
        next_prices.append(closes[idx + 1])
        row_indices.append(idx)

    return Dataset(
        features=features,
        targets=targets,
        feature_names=feature_names,
        entry_prices=entry_prices,
        next_prices=next_prices,
        row_indices=row_indices,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
    )


def split_dataset(dataset: Dataset, train_ratio: float) -> Tuple[Dataset, Dataset]:
    total = len(dataset.features)
    train_count = int(total * train_ratio)
    train = Dataset(
        features=dataset.features[:train_count],
        targets=dataset.targets[:train_count],
        feature_names=dataset.feature_names,
        entry_prices=dataset.entry_prices[:train_count],
        next_prices=dataset.next_prices[:train_count],
        row_indices=dataset.row_indices[:train_count],
        opens=dataset.opens,
        highs=dataset.highs,
        lows=dataset.lows,
        closes=dataset.closes,
    )
    test = Dataset(
        features=dataset.features[train_count:],
        targets=dataset.targets[train_count:],
        feature_names=dataset.feature_names,
        entry_prices=dataset.entry_prices[train_count:],
        next_prices=dataset.next_prices[train_count:],
        row_indices=dataset.row_indices[train_count:],
        opens=dataset.opens,
        highs=dataset.highs,
        lows=dataset.lows,
        closes=dataset.closes,
    )
    if not train.features or not test.features:
        raise ValueError("Training or test split ended up empty; adjust train_ratio")
    return train, test


def compute_standardization(features: Sequence[Sequence[float]]) -> Standardization:
    num_features = len(features[0])
    means: List[float] = []
    stds: List[float] = []
    for idx in range(num_features):
        column = [row[idx] for row in features]
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        std = math.sqrt(variance)
        if std < 1e-12:
            std = 1.0
        means.append(mean)
        stds.append(std)
    return Standardization(means=means, stds=stds)


def apply_standardization(features: Sequence[Sequence[float]], stats: Standardization) -> List[List[float]]:
    scaled: List[List[float]] = []
    for row in features:
        scaled.append([(value - mean) / std for value, mean, std in zip(row, stats.means, stats.stds)])
    return scaled


def train_neural_network(
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    learning_rate: float,
    epochs: int,
    hidden_units: int,
    batch_size: int,
    l2: float = 0.0,
    early_stop_patience: int = 0,
    early_stop_delta: float = 1e-4,
) -> Tuple[NeuralNetwork, List[float]]:
    if not features:
        raise ValueError("No training samples provided")
    n_samples = len(features)
    n_features = len(features[0])
    rng = random.Random(42)

    w1: List[List[float]] = [
        [rng.gauss(0.0, 0.1) for _ in range(hidden_units)] for _ in range(n_features)
    ]
    b1: List[float] = [0.0 for _ in range(hidden_units)]
    w2: List[float] = [rng.gauss(0.0, 0.1) for _ in range(hidden_units)]
    b2 = 0.0

    loss_history: List[float] = []
    indices = list(range(n_samples))

    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            batch_indices = indices[start : start + batch_size]
            if not batch_indices:
                continue

            grad_w1 = [[0.0 for _ in range(hidden_units)] for _ in range(n_features)]
            grad_b1 = [0.0 for _ in range(hidden_units)]
            grad_w2 = [0.0 for _ in range(hidden_units)]
            grad_b2 = 0.0

            for idx in batch_indices:
                x = features[idx]
                y = targets[idx]
                z1 = [
                    sum(x[f] * w1[f][h] for f in range(n_features)) + b1[h]
                    for h in range(hidden_units)
                ]
                hidden = [value if value > 0 else 0.0 for value in z1]
                logit = sum(hidden[h] * w2[h] for h in range(hidden_units)) + b2
                logit = max(min(logit, 60.0), -60.0)
                prob = 1.0 / (1.0 + math.exp(-logit))
                error = prob - y

                for h in range(hidden_units):
                    grad_w2[h] += error * hidden[h]
                grad_b2 += error

                for h in range(hidden_units):
                    grad = error * w2[h]
                    if z1[h] <= 0:
                        grad = 0.0
                    grad_b1[h] += grad
                    for f in range(n_features):
                        grad_w1[f][h] += grad * x[f]

            batch_len = len(batch_indices)
            inv_batch = 1.0 / batch_len

            for h in range(hidden_units):
                grad_w2[h] = grad_w2[h] * inv_batch + (l2 * w2[h] if l2 else 0.0)
            grad_b2 *= inv_batch

            for f in range(n_features):
                for h in range(hidden_units):
                    grad = grad_w1[f][h] * inv_batch + (l2 * w1[f][h] if l2 else 0.0)
                    w1[f][h] -= learning_rate * grad
            for h in range(hidden_units):
                grad = grad_b1[h] * inv_batch
                b1[h] -= learning_rate * grad
                w2[h] -= learning_rate * grad_w2[h]
            b2 -= learning_rate * grad_b2

        total_loss = 0.0
        for idx in range(n_samples):
            x = features[idx]
            y = targets[idx]
            z1 = [
                sum(x[f] * w1[f][h] for f in range(n_features)) + b1[h]
                for h in range(hidden_units)
            ]
            hidden = [value if value > 0 else 0.0 for value in z1]
            logit = sum(hidden[h] * w2[h] for h in range(hidden_units)) + b2
            logit = max(min(logit, 60.0), -60.0)
            prob = 1.0 / (1.0 + math.exp(-logit))
            eps = 1e-9
            total_loss += -(y * math.log(prob + eps) + (1 - y) * math.log(1 - prob + eps))
        total_loss /= n_samples
        if l2:
            total_loss += 0.5 * l2 * (
                sum(weight * weight for row in w1 for weight in row)
                + sum(weight * weight for weight in w2)
            )
        loss_history.append(total_loss)
        if early_stop_patience and len(loss_history) >= early_stop_patience:
            window = loss_history[-early_stop_patience:]
            if max(window) - min(window) < early_stop_delta:
                break

    model = NeuralNetwork(w1=w1, b1=b1, w2=w2, b2=b2)
    return model, loss_history


def evaluate_model(
    model: NeuralNetwork,
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
    threshold: float = 0.5,
) -> Evaluation:
    tp = tn = fp = fn = 0
    for row, target in zip(features, targets):
        prob = model.predict_proba(row)
        pred = 1 if prob >= threshold else 0
        if pred == 1 and target == 1:
            tp += 1
        elif pred == 0 and target == 0:
            tn += 1
        elif pred == 1 and target == 0:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return Evaluation(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )


def simulate_trade_exit(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    entry_index: int,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    trailing_stop_pct: float | None,
) -> Tuple[float, int, int, str]:
    entry_price = closes[entry_index]
    if math.isnan(entry_price) or entry_price <= 0:
        return entry_price if entry_price > 0 else 0.0, entry_index, 0, "invalid"

    stop_price = entry_price * (1 - stop_loss_pct) if stop_loss_pct is not None else None
    target_price = entry_price * (1 + take_profit_pct) if take_profit_pct is not None else None
    peak_price = entry_price
    bars_held = 0

    for idx in range(entry_index + 1, len(closes)):
        high = highs[idx]
        low = lows[idx]
        close = closes[idx]
        if any(math.isnan(value) or value <= 0 for value in (high, low, close)):
            continue

        bars_held += 1
        peak_price = max(peak_price, high)

        stop_candidates: List[float] = []
        if stop_price is not None:
            stop_candidates.append(stop_price)
        if trailing_stop_pct is not None:
            trailing_level = peak_price * (1 - trailing_stop_pct)
            stop_candidates.append(trailing_level)
        effective_stop = max(stop_candidates) if stop_candidates else None

        if effective_stop is not None and low <= effective_stop:
            return effective_stop, idx, bars_held, "stop"
        if target_price is not None and high >= target_price:
            return target_price, idx, bars_held, "target"

    final_close = closes[-1]
    if math.isnan(final_close) or final_close <= 0:
        final_close = entry_price
    if bars_held == 0:
        bars_held = 1
    return final_close, len(closes) - 1, bars_held, "time"


def run_backtest_scenario(
    predictions: Sequence[int],
    dataset: Dataset,
    starting_cash: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    trailing_stop_pct: float | None,
) -> BacktestReport:
    cash = starting_cash
    trades = wins = losses = 0
    total_bars = 0
    max_bars = 0
    idx = 0
    row_indices = dataset.row_indices

    while idx < len(predictions):
        if predictions[idx] != 1:
            idx += 1
            continue

        entry_row = row_indices[idx]
        if entry_row >= len(dataset.closes) - 1:
            break
        entry_price = dataset.entry_prices[idx]
        if entry_price <= 0 or math.isnan(entry_price):
            idx += 1
            continue
        if cash <= 0:
            break

        position_value = cash
        shares = position_value / entry_price
        cash = 0.0
        exit_price, exit_row, bars_held, _ = simulate_trade_exit(
            dataset.closes,
            dataset.highs,
            dataset.lows,
            entry_row,
            stop_loss_pct,
            take_profit_pct,
            trailing_stop_pct,
        )
        proceeds = shares * exit_price
        pnl = proceeds - position_value
        cash = proceeds

        trades += 1
        if pnl > 1e-9:
            wins += 1
        elif pnl < -1e-9:
            losses += 1

        total_bars += bars_held
        max_bars = max(max_bars, bars_held)

        idx += 1
        while idx < len(predictions) and row_indices[idx] <= exit_row:
            idx += 1

    ending_value = cash
    profit = ending_value - starting_cash
    return_pct = (profit / starting_cash) * 100 if starting_cash else 0.0
    win_rate = (wins / trades) * 100 if trades else 0.0
    win_loss_ratio = (wins / losses) if losses else (float("inf") if wins > 0 else None)
    average_bars = (total_bars / trades) if trades else 0.0

    return BacktestReport(
        starting_cash=starting_cash,
        ending_value=ending_value,
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        win_loss_ratio=win_loss_ratio,
        profit=profit,
        return_pct=return_pct,
        average_bars_held=average_bars,
        max_bars_held=max_bars,
        stop_loss_pct=(stop_loss_pct * 100 if stop_loss_pct is not None else None),
        take_profit_pct=(take_profit_pct * 100 if take_profit_pct is not None else None),
        trailing_stop_pct=(trailing_stop_pct * 100 if trailing_stop_pct is not None else None),
    )


def scan_backtest_scenarios(
    predictions: Sequence[int],
    dataset: Dataset,
    starting_cash: float,
    stop_losses: Sequence[float],
    take_profits: Sequence[float],
    trailing_stops: Sequence[float],
    *,
    risk_reward: float | None,
    ratio_tolerance: float,
    min_win_rate: float,
) -> List[BacktestReport]:
    results: List[BacktestReport] = []
    trailing_options: List[float | None] = [None]
    trailing_options.extend(trailing_stops)

    for stop_loss in stop_losses:
        for take_profit in take_profits:
            if (
                risk_reward is not None
                and stop_loss
                and take_profit
                and stop_loss > 0
            ):
                ratio = take_profit / stop_loss
                allowed_deviation = abs(risk_reward) * ratio_tolerance
                if abs(ratio - risk_reward) > allowed_deviation:
                    continue
            for trailing in trailing_options:
                report = run_backtest_scenario(
                    predictions=predictions,
                    dataset=dataset,
                    starting_cash=starting_cash,
                    stop_loss_pct=stop_loss,
                    take_profit_pct=take_profit,
                    trailing_stop_pct=trailing,
                )
                results.append(report)

    if min_win_rate > 0:
        results = [report for report in results if report.win_rate >= min_win_rate]

    results.sort(key=lambda report: report.ending_value, reverse=True)
    return results


def normalize_percent_list(values: Sequence[float]) -> List[float]:
    unique: List[float] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return [max(value, 0.0) / 100.0 for value in unique]


def describe_patterns(feature_names: Sequence[str], model: NeuralNetwork) -> List[str]:
    weights: List[Tuple[str, float]] = []
    for idx, name in enumerate(feature_names):
        contribution = 0.0
        for h in range(len(model.w2)):
            contribution += model.w1[idx][h] * model.w2[h]
        weights.append((name, contribution))
    ranked = sorted(weights, key=lambda item: abs(item[1]), reverse=True)

    descriptions: List[str] = []
    for name, weight in ranked:
        direction_phrase = "bullish continuation" if weight > 0 else "bearish continuation"
        direction_side = "bullish" if weight > 0 else "bearish"
        if name == "log_return_1":
            message = (
                f"Immediate 5-minute momentum (log_return_1) favors {direction_phrase} moves on the next bar."
            )
        elif name == "log_return_5":
            message = (
                f"A positive tilt across the past five bars (log_return_5) aligns with {direction_phrase} behavior."
            )
        elif name == "log_return_10":
            message = (
                f"Broader swing momentum over ten bars (log_return_10) points toward {direction_phrase} outcomes."
            )
        elif name == "body_pct":
            message = (
                f"Larger candle bodies relative to the open (body_pct) are linked with {direction_phrase} follow-through."
            )
        elif name == "range_pct":
            message = (
                f"Wide intrabar ranges (range_pct) correlate with {direction_phrase} on the next close."
            )
        elif name == "volatility_5":
            message = (
                f"Short-term volatility (volatility_5) signals {direction_phrase} pressure when elevated."
            )
        elif name == "volatility_14":
            message = (
                f"Sustained volatility over 14 bars (volatility_14) encourages {direction_phrase} setups."
            )
        elif name == "ema_fast_minus_slow":
            message = (
                f"The 12/26 EMA spread (ema_fast_minus_slow) implies {direction_phrase} momentum bias."
            )
        elif name == "macd_histogram":
            message = (
                f"MACD histogram thrust (macd_histogram) points toward {direction_phrase}."
            )
        elif name == "rsi_14":
            message = (
                f"Relative strength (rsi_14) skew indicates {direction_phrase} pressure after normalization."
            )
        elif name == "atr_pct":
            message = (
                f"True range as a share of price (atr_pct) maps to {direction_phrase} moves in this regime."
            )
        elif name == "stoch_k":
            message = (
                f"Stochastic %%K momentum (stoch_k) biases the model toward {direction_phrase} outcomes."
            )
        elif name == "stoch_d":
            message = (
                f"Stochastic %%D confirmation (stoch_d) leans toward {direction_phrase} resolutions."
            )
        elif name == "trend_vs_ema":
            message = (
                f"Distance from the slow EMA (trend_vs_ema) anchors {direction_phrase} expectations."
            )
        else:
            message = f"Feature {name} weight {weight:+.4f} indicates {direction_side} momentum."
        descriptions.append(f"{message} (weight={weight:+.4f})")
    return descriptions


def run_training(
    *,
    data: Path,
    train_ratio: float,
    learning_rate: float,
    epochs: int,
    hidden_units: int,
    batch_size: int,
    train_window: int,
    l2: float,
    patience: int,
    min_delta: float,
    threshold: float,
) -> Tuple[TrainingResult, TrainTestSplit]:
    rows = load_dataset(data)
    dataset = build_features(rows)
    train, test = split_dataset(dataset, train_ratio)

    standardization = compute_standardization(train.features)
    train_scaled = apply_standardization(train.features, standardization)
    test_scaled = apply_standardization(test.features, standardization)

    if train_window > 0 and len(train_scaled) > train_window:
        train_subset_features = train_scaled[-train_window:]
        train_subset_targets = train.targets[-train_window:]
    else:
        train_subset_features = train_scaled
        train_subset_targets = train.targets

    model, loss_history = train_neural_network(
        train_subset_features,
        train_subset_targets,
        learning_rate=learning_rate,
        epochs=epochs,
        hidden_units=hidden_units,
        batch_size=batch_size,
        l2=l2,
        early_stop_patience=patience,
        early_stop_delta=min_delta,
    )

    evaluation = evaluate_model(model, test_scaled, test.targets, threshold=threshold)
    test_probabilities = [model.predict_proba(row) for row in test_scaled]
    test_predictions = [1 if prob >= threshold else 0 for prob in test_probabilities]
    result = TrainingResult(
        model=model,
        standardization=standardization,
        evaluation=evaluation,
        train_count=len(train_subset_targets),
        full_train_count=len(train.features),
        test_count=len(test.features),
        test_probabilities=test_probabilities,
        test_predictions=test_predictions,
        threshold=threshold,
        loss_history=loss_history,
    )
    split = TrainTestSplit(train=train, test=test)
    return result, split


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_FILE,
        help="Path to the OHLC CSV dataset (default: BATS_CSCO, 5.csv)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of samples used for training (default: 0.7)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for neural network training (default: 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Maximum number of training epochs (default: 35)",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=12,
        help="Width of the neural network hidden layer (default: 12)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size for stochastic gradient descent (default: 512)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs with minimal loss change before early stopping (default: 10; 0 disables)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum loss improvement required across the patience window (default: 1e-4)",
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=2500,
        help="Most recent training samples to fit (default: 2500; 0 uses the entire train split)",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.0005,
        help="L2 regularization strength (default: 0.0005)",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for issuing long signals (default: 0.5)",
    )
    parser.add_argument(
        "--top-signals",
        type=int,
        default=0,
        help="Optionally restrict trades to the top-N probabilities that clear the threshold (default: 0 = all)",
    )
    parser.add_argument(
        "--stop-losses",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
        help="Static stop-loss percentages to evaluate (default: 0.2 0.4 0.6 0.8 1.0 1.5)",
    )
    parser.add_argument(
        "--take-profits",
        type=float,
        nargs="+",
        default=[0.4, 0.8, 1.0, 1.2, 1.6, 2.0, 3.0, 4.0],
        help="Profit targets (percent) to evaluate (default: 0.4 0.8 1.0 1.2 1.6 2.0 3.0 4.0)",
    )
    parser.add_argument(
        "--trailing-stops",
        type=float,
        nargs="*",
        default=[1.0, 2.0, 3.0],
        help="Trailing stop percentages to pair with the static stop (default: 1 2 3; omit values to disable)",
    )
    parser.add_argument(
        "--risk-reward",
        type=float,
        default=None,
        help="Optional minimum take-profit / stop-loss ratio (e.g., 2 for a 1:2 risk/reward)",
    )
    parser.add_argument(
        "--ratio-tolerance",
        type=float,
        default=0.05,
        help="Allowable fractional deviation when enforcing the risk/reward ratio (default: 0.05)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1000.0,
        help="Starting capital for the stop-loss/target scenario backtest (default: 1000)",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.0,
        help="Minimum win rate (percentage) required for a scenario to be reported (default: 0)",
    )
    return parser.parse_args(argv)


def print_report(
    *,
    split: TrainTestSplit,
    result: TrainingResult,
    backtests: Sequence[BacktestReport],
    selected_signals: int,
) -> None:
    train_size = result.train_count
    test_size = result.test_count
    print("Advanced machine learning pattern discovery for CSCO 5-minute data")
    print("=================================================================")
    print(
        f"Samples (train/test): {train_size} / {test_size} "
        f"(from {result.full_train_count} total train bars)"
    )
    hidden_units = len(result.model.w2)
    print(
        f"Architecture: {len(split.train.feature_names)} inputs -> {hidden_units} hidden -> 1 output"
    )
    print(f"Signal probability threshold: {result.threshold:.2f}")
    print(f"Signals passing threshold : {selected_signals}")
    print("Features: " + ", ".join(split.train.feature_names))
    print("")
    print("Test set evaluation:")
    eval_result = result.evaluation
    print(f"  Accuracy : {eval_result.accuracy:.4f}")
    print(f"  Precision: {eval_result.precision:.4f}")
    print(f"  Recall   : {eval_result.recall:.4f}")
    print(f"  F1 score : {eval_result.f1:.4f}")
    print(f"  Confusion matrix (TP, TN, FP, FN): {eval_result.true_positive}, {eval_result.true_negative}, {eval_result.false_positive}, {eval_result.false_negative}")
    if result.loss_history:
        print(f"  Epochs trained : {len(result.loss_history)}")
        print(f"  Final training loss: {result.loss_history[-1]:.4f}")
    print("")
    print("Feature influence (highest magnitude first):")
    descriptions = describe_patterns(split.train.feature_names, result.model)
    for line in descriptions:
        print("  - " + line)
    print("")
    print("Stop-loss / target scan on the test set:")
    if not backtests:
        print("  No qualifying trade scenarios matched the filters.")
        return

    best = backtests[0]

    def format_ratio(value: float | None) -> str:
        if value is None:
            return "n/a"
        if math.isinf(value):
            return "Infinity"
        return f"{value:.2f}"

    def format_percent(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.2f}"

    print(f"  Scenarios evaluated : {len(backtests)}")
    print(
        f"  Best ending equity  : ${best.ending_value:,.2f} "
        f"(profit ${best.profit:,.2f}, {best.return_pct:.2f}%)"
    )
    print(
        "  Parameters          : "
        f"stop {format_percent(best.stop_loss_pct)}% / "
        f"target {format_percent(best.take_profit_pct)}% / "
        f"trailing {format_percent(best.trailing_stop_pct)}%"
    )
    print(
        f"  Trades / Wins / Losses : {best.trades} / {best.wins} / {best.losses}"
    )
    print(f"  Win rate            : {best.win_rate:.2f}%")
    print(f"  Win/loss ratio      : {format_ratio(best.win_loss_ratio)}")
    print(
        f"  Avg hold (bars)     : {best.average_bars_held:.2f} "
        f"(max {best.max_bars_held})"
    )

    print("")
    print("  Top scenarios by ending equity:")
    header = (
        f"{'Stop%':>7} {'Target%':>8} {'Trail%':>7} "
        f"{'End Equity':>14} {'Return%':>9} {'Trades':>8} "
        f"{'Win%':>7} {'AvgBars':>8}"
    )
    print("  " + header)
    for report in backtests[:5]:
        print(
            "  "
            f"{format_percent(report.stop_loss_pct):>7} "
            f"{format_percent(report.take_profit_pct):>8} "
            f"{format_percent(report.trailing_stop_pct):>7} "
            f"{report.ending_value:>14,.2f} "
            f"{report.return_pct:>9.2f} "
            f"{report.trades:>8} "
            f"{report.win_rate:>7.2f} "
            f"{report.average_bars_held:>8.2f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result, split = run_training(
        data=args.data,
        train_ratio=args.train_ratio,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        hidden_units=args.hidden_units,
        batch_size=args.batch_size,
        train_window=args.train_window,
        l2=args.l2,
        patience=args.patience,
        min_delta=args.min_delta,
        threshold=args.signal_threshold,
    )
    stop_losses = normalize_percent_list(args.stop_losses)
    take_profits = normalize_percent_list(args.take_profits)
    trailing_stops = normalize_percent_list(args.trailing_stops)

    if not stop_losses:
        raise ValueError("Provide at least one stop-loss percentage")
    if not take_profits:
        raise ValueError("Provide at least one take-profit percentage")

    eligible = [
        (prob, idx)
        for idx, prob in enumerate(result.test_probabilities)
        if prob >= args.signal_threshold
    ]
    if args.top_signals > 0 and len(eligible) > args.top_signals:
        eligible.sort(key=lambda item: item[0], reverse=True)
        selected_indices = {idx for _, idx in eligible[: args.top_signals]}
    else:
        selected_indices = {idx for _, idx in eligible}
    signal_predictions = [1 if idx in selected_indices else 0 for idx in range(len(result.test_probabilities))]
    backtests = scan_backtest_scenarios(
        predictions=signal_predictions,
        dataset=split.test,
        starting_cash=args.capital,
        stop_losses=stop_losses,
        take_profits=take_profits,
        trailing_stops=trailing_stops,
        risk_reward=args.risk_reward,
        ratio_tolerance=args.ratio_tolerance,
        min_win_rate=args.min_win_rate,
    )
    print_report(
        split=split,
        result=result,
        backtests=backtests,
        selected_signals=len(selected_indices),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
