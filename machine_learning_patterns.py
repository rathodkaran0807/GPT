"""Uncover directional patterns in the `BATS_CSCO, 5.csv` dataset using
a lightweight logistic regression optimizer implemented with only the
Python standard library.

The script engineers a small feature set from 5-minute OHLC candles and trains
a logistic regression model to predict whether the next bar's close will finish
above the current close. Coefficients are reported in descending magnitude to
highlight which patterns the model relied on most.

Usage (default dataset):

    python machine_learning_patterns.py

You can override the dataset path, training split, and the optimizer's training
epochs via CLI flags. See `--help` for details.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import math
import random


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


@dataclass
class Standardization:
    means: List[float]
    stds: List[float]


def standardize_row(row: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> List[float]:
    standardized: List[float] = []
    for value, mean, std in zip(row, means, stds):
        adjusted_std = std if std else 1.0
        standardized.append((value - mean) / adjusted_std)
    return standardized


def compute_standardization(features: Sequence[Sequence[float]]) -> Standardization:
    if not features:
        raise ValueError("Cannot standardize an empty feature matrix")
    feature_count = len(features[0])
    means: List[float] = []
    stds: List[float] = []
    for index in range(feature_count):
        column = [row[index] for row in features]
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std)
    return Standardization(means=means, stds=stds)


@dataclass
class Model:
    estimator: "SimpleLogisticRegression"

    def predict_label(self, features: Sequence[float]) -> int:
        return self.estimator.predict(features)

    def predict_proba(self, features: Sequence[float]) -> float:
        return self.estimator.predict_proba(features)

    @property
    def weights(self) -> List[float]:
        return list(self.estimator.weights)

    @property
    def bias(self) -> float:
        return float(self.estimator.bias)


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
class EvaluationReport:
    label: str
    evaluation: Evaluation
    simulation: TradeSimulation
    sample_count: int


@dataclass
class TrainingResult:
    model: Model
    standardization: Standardization
    evaluation: Evaluation
    train_count: int
    test_count: int
    test_predictions: List[int]
    trade_simulation: TradeSimulation
    external_reports: List[EvaluationReport]


class SimpleLogisticRegression:
    def __init__(self, *, learning_rate: float = 0.05, epochs: int = 500, l2_penalty: float = 0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.weights: List[float] = []
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            exp_val = math.exp(-value)
            return 1.0 / (1.0 + exp_val)
        exp_val = math.exp(value)
        return exp_val / (1.0 + exp_val)

    def fit(self, features: Sequence[Sequence[float]], targets: Sequence[int]) -> None:
        if not features:
            raise ValueError("Cannot train logistic regression with no samples")
        feature_count = len(features[0])
        self.weights = [0.0 for _ in range(feature_count)]
        self.bias = 0.0
        indices = list(range(len(features)))
        rng = random.Random(0)

        for _ in range(self.epochs):
            rng.shuffle(indices)
            grad_w = [0.0 for _ in range(feature_count)]
            grad_b = 0.0
            for idx in indices:
                row = features[idx]
                target = targets[idx]
                score = sum(weight * value for weight, value in zip(self.weights, row)) + self.bias
                prediction = self._sigmoid(score)
                error = prediction - target
                for feature_index in range(feature_count):
                    grad_w[feature_index] += error * row[feature_index]
                grad_b += error

            sample_count = float(len(features))
            for feature_index in range(feature_count):
                penalty = self.l2_penalty * self.weights[feature_index]
                gradient = (grad_w[feature_index] / sample_count) + penalty
                self.weights[feature_index] -= self.learning_rate * gradient
            self.bias -= self.learning_rate * (grad_b / sample_count)

    def predict_proba(self, features: Sequence[float]) -> float:
        score = sum(weight * value for weight, value in zip(self.weights, features)) + self.bias
        return self._sigmoid(score)

    def predict(self, features: Sequence[float]) -> int:
        return 1 if self.predict_proba(features) >= 0.5 else 0


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
    return float(math.log(new / old))


def rolling_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))


def build_features(rows: List[dict]) -> Dataset:
    closes = [to_float(row["close"]) for row in rows]
    opens = [to_float(row["open"]) for row in rows]
    highs = [to_float(row["high"]) for row in rows]
    lows = [to_float(row["low"]) for row in rows]

    max_lag = 10
    feature_names = [
        "log_return_1",  # immediate momentum
        "log_return_5",  # weekly-ish momentum
        "log_return_10",  # longer swing momentum
        "body_pct",  # candle body as share of open
        "range_pct",  # intrabar volatility
        "volatility_5",  # rolling volatility over 5 bars
    ]

    features: List[List[float]] = []
    targets: List[int] = []
    entry_prices: List[float] = []
    next_prices: List[float] = []

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
        window = log_returns[idx - 4 : idx + 1]
        vol = rolling_std(window)

        feature_vector = [
            safe_value(lr_1),
            safe_value(lr_5),
            safe_value(lr_10),
            safe_value(body_pct),
            safe_value(range_pct),
            safe_value(vol),
        ]

        features.append(feature_vector)
        targets.append(1 if closes[idx + 1] > closes[idx] else 0)
        entry_prices.append(closes[idx])
        next_prices.append(closes[idx + 1])

    return Dataset(
        features=features,
        targets=targets,
        feature_names=feature_names,
        entry_prices=entry_prices,
        next_prices=next_prices,
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
    )
    test = Dataset(
        features=dataset.features[train_count:],
        targets=dataset.targets[train_count:],
        feature_names=dataset.feature_names,
        entry_prices=dataset.entry_prices[train_count:],
        next_prices=dataset.next_prices[train_count:],
    )
    if not train.features or not test.features:
        raise ValueError("Training or test split ended up empty; adjust train_ratio")
    return train, test


def evaluate_model(
    estimator: SimpleLogisticRegression,
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
) -> Tuple[Evaluation, List[int]]:
    predictions = [estimator.predict(row) for row in features]
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    total = len(targets)
    accuracy = correct / total if total else 0.0

    tp = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 1)
    tn = sum(1 for pred, target in zip(predictions, targets) if pred == 0 and target == 0)
    fp = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 0)
    fn = sum(1 for pred, target in zip(predictions, targets) if pred == 0 and target == 1)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    evaluation = Evaluation(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_positive=int(tp),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
    )
    return evaluation, predictions


@dataclass
class TradeSimulation:
    initial_capital: float
    final_capital: float
    total_return: float
    trades: int


def simulate_trades(
    *,
    predictions: Sequence[int],
    entry_prices: Sequence[float],
    next_prices: Sequence[float],
    initial_capital: float,
) -> TradeSimulation:
    if len(predictions) != len(entry_prices) or len(predictions) != len(next_prices):
        raise ValueError("Predictions and price series must align")

    capital = initial_capital
    for prediction, entry_price, exit_price in zip(predictions, entry_prices, next_prices):
        if entry_price <= 0 or exit_price <= 0:
            continue
        pct_change = (exit_price - entry_price) / entry_price
        if prediction == 1:
            capital *= 1.0 + pct_change
        else:
            capital *= 1.0 - pct_change

    total_return = (capital - initial_capital) / initial_capital if initial_capital else 0.0
    return TradeSimulation(
        initial_capital=float(initial_capital),
        final_capital=float(capital),
        total_return=float(total_return),
        trades=len(predictions),
    )


def describe_patterns(feature_names: Sequence[str], weights: Sequence[float]) -> List[str]:
    ranked = sorted(zip(feature_names, weights), key=lambda item: abs(item[1]), reverse=True)
    descriptions: List[str] = []
    for name, weight in ranked:
        direction = "bullish continuation" if weight > 0 else "bearish continuation"
        if name == "log_return_1":
            message = (
                "Recent 5-minute momentum (log_return_1) carries a %s bias, meaning strong "
                "moves tend to continue into the next bar." % direction
            )
        elif name == "log_return_5":
            message = (
                "A positive 5-bar momentum tilt (log_return_5) supports %s setups." % direction
            )
        elif name == "log_return_10":
            message = (
                "The broader 10-bar trend (log_return_10) aligns with %s behavior." % direction
            )
        elif name == "body_pct":
            message = (
                "Larger candle bodies relative to the open (body_pct) encourage %s follow-through." % direction
            )
        elif name == "range_pct":
            message = (
                "Wide intrabar ranges (range_pct) correspond to %s resolution on the next close." % direction
            )
        elif name == "volatility_5":
            message = (
                "Elevated short-term volatility (volatility_5) is linked with %s outcomes." % direction
            )
        else:
            message = f"Feature {name} weight {weight:+.4f} indicates {direction}."
        descriptions.append(f"{message} (weight={weight:+.4f})")
    return descriptions


def run_training(
    *,
    data: Path,
    train_ratio: float,
    max_iter: int,
    initial_capital: float,
    test_data: Sequence[Path],
) -> Tuple[TrainingResult, Dataset]:
    rows = load_dataset(data)
    dataset = build_features(rows)
    train, test = split_dataset(dataset, train_ratio)

    standardization = compute_standardization(train.features)
    train_scaled = [
        standardize_row(row, standardization.means, standardization.stds)
        for row in train.features
    ]
    test_scaled = [
        standardize_row(row, standardization.means, standardization.stds)
        for row in test.features
    ]

    estimator = SimpleLogisticRegression(epochs=max_iter)
    estimator.fit(train_scaled, train.targets)

    evaluation, predictions = evaluate_model(estimator, test_scaled, test.targets)
    simulation = simulate_trades(
        predictions=predictions,
        entry_prices=test.entry_prices,
        next_prices=test.next_prices,
        initial_capital=initial_capital,
    )
    external_reports: List[EvaluationReport] = []
    if test_data:
        for dataset_path in test_data:
            alternate_rows = load_dataset(dataset_path)
            alternate_dataset = build_features(alternate_rows)
            alternate_scaled = [
                standardize_row(row, standardization.means, standardization.stds)
                for row in alternate_dataset.features
            ]
            alt_evaluation, alt_predictions = evaluate_model(
                estimator, alternate_scaled, alternate_dataset.targets
            )
            alt_simulation = simulate_trades(
                predictions=alt_predictions,
                entry_prices=alternate_dataset.entry_prices,
                next_prices=alternate_dataset.next_prices,
                initial_capital=initial_capital,
            )
            external_reports.append(
                EvaluationReport(
                    label=str(dataset_path),
                    evaluation=alt_evaluation,
                    simulation=alt_simulation,
                    sample_count=len(alternate_dataset.features),
                )
            )

    result = TrainingResult(
        model=Model(estimator=estimator),
        standardization=standardization,
        evaluation=evaluation,
        train_count=len(train.features),
        test_count=len(test.features),
        test_predictions=predictions,
        trade_simulation=simulation,
        external_reports=external_reports,
    )
    return result, dataset


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
        "--max-iter",
        type=int,
        default=500,
        help="Training epochs for the logistic regression optimizer (default: 500)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1000.0,
        help="Starting capital for the trading simulation (default: 1000.0)",
    )
    parser.add_argument(
        "--test-data",
        action="append",
        type=Path,
        default=None,
        help=(
            "Optional dataset path evaluated with the trained model (no additional training). "
            "Repeat the flag to assess multiple files."
        ),
    )
    return parser.parse_args(argv)


def print_report(
    *,
    dataset: Dataset,
    result: TrainingResult,
) -> None:
    train_size = result.train_count
    test_size = result.test_count
    print("Machine learning pattern discovery")
    print("=================================")
    print(f"Samples (train/test): {train_size} / {test_size}")
    print("Features: " + ", ".join(dataset.feature_names))
    print("")
    print("Test set evaluation:")
    eval_result = result.evaluation
    print(f"  Accuracy : {eval_result.accuracy:.4f}")
    print(f"  Precision: {eval_result.precision:.4f}")
    print(f"  Recall   : {eval_result.recall:.4f}")
    print(f"  F1 score : {eval_result.f1:.4f}")
    print(f"  Confusion matrix (TP, TN, FP, FN): {eval_result.true_positive}, {eval_result.true_negative}, {eval_result.false_positive}, {eval_result.false_negative}")
    print("")
    simulation = result.trade_simulation
    print(
        "Trading simulation (initial capital ${:,.2f}):".format(
            simulation.initial_capital
        )
    )
    print(f"  Trades executed : {simulation.trades}")
    print(f"  Final capital   : ${simulation.final_capital:,.2f}")
    print(f"  Net profit      : ${simulation.final_capital - simulation.initial_capital:,.2f} ({simulation.total_return * 100:.2f}% return)")
    print("")
    print("Feature influence (highest magnitude first):")
    descriptions = describe_patterns(dataset.feature_names, result.model.weights)
    for line in descriptions:
        print("  - " + line)
    if result.external_reports:
        print("")
        for report in result.external_reports:
            print(f"Out-of-sample evaluation ({report.label}):")
            print(f"  Samples       : {report.sample_count}")
            print(f"  Accuracy      : {report.evaluation.accuracy:.4f}")
            print(f"  Precision     : {report.evaluation.precision:.4f}")
            print(f"  Recall        : {report.evaluation.recall:.4f}")
            print(f"  F1 score      : {report.evaluation.f1:.4f}")
            print(
                "  Confusion matrix (TP, TN, FP, FN): "
                f"{report.evaluation.true_positive}, {report.evaluation.true_negative}, "
                f"{report.evaluation.false_positive}, {report.evaluation.false_negative}"
            )
            final_return_pct = report.simulation.total_return * 100
            print(
                "  Trading simulation final capital: "
                f"${report.simulation.final_capital:,.2f} "
                f"({report.simulation.trades} trades, {final_return_pct:.2f}% return)"
            )
            print("")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result, dataset = run_training(
        data=args.data,
        train_ratio=args.train_ratio,
        max_iter=args.max_iter,
        initial_capital=args.initial_capital,
        test_data=args.test_data or [],
    )
    print_report(dataset=dataset, result=result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
