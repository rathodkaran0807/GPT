"""Train a simple logistic regression to uncover directional patterns in the
`BATS_CSCO, 5.csv` dataset.

The script engineers a small feature set from 5-minute OHLC candles and learns a
classifier that predicts whether the next bar's close will finish above the
current close. Coefficients are reported in descending magnitude to highlight
which patterns the model relied on most.

Usage (default dataset):

    python machine_learning_patterns.py

You can override the dataset path, training split, number of optimization
iterations, and learning rate via CLI flags. See `--help` for details.
"""

from __future__ import annotations

import argparse
import csv
import math
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


@dataclass
class Standardization:
    means: List[float]
    stds: List[float]


@dataclass
class Model:
    weights: List[float]
    bias: float

    def decision_function(self, features: Sequence[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, features)) + self.bias

    def predict_proba(self, features: Sequence[float]) -> float:
        z = self.decision_function(features)
        return sigmoid(z)

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
    model: Model
    standardization: Standardization
    evaluation: Evaluation
    train_count: int
    test_count: int


def sigmoid(z: float) -> float:
    if z >= 0:
        exp_neg = math.exp(-z)
        return 1 / (1 + exp_neg)
    exp_pos = math.exp(z)
    return exp_pos / (1 + exp_pos)


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

    return Dataset(features=features, targets=targets, feature_names=feature_names)


def split_dataset(dataset: Dataset, train_ratio: float) -> Tuple[Dataset, Dataset]:
    total = len(dataset.features)
    train_count = int(total * train_ratio)
    train = Dataset(
        features=dataset.features[:train_count],
        targets=dataset.targets[:train_count],
        feature_names=dataset.feature_names,
    )
    test = Dataset(
        features=dataset.features[train_count:],
        targets=dataset.targets[train_count:],
        feature_names=dataset.feature_names,
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


def train_logistic_regression(
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    learning_rate: float,
    iterations: int,
) -> Model:
    num_features = len(features[0])
    weights = [0.0 for _ in range(num_features)]
    bias = 0.0
    n = len(features)

    for epoch in range(iterations):
        grad_w = [0.0 for _ in range(num_features)]
        grad_b = 0.0
        for row, target in zip(features, targets):
            z = sum(w * x for w, x in zip(weights, row)) + bias
            pred = sigmoid(z)
            error = pred - target
            for idx in range(num_features):
                grad_w[idx] += error * row[idx]
            grad_b += error

        step = learning_rate / n
        for idx in range(num_features):
            weights[idx] -= step * grad_w[idx]
        bias -= step * grad_b

    return Model(weights=weights, bias=bias)


def evaluate_model(model: Model, features: Sequence[Sequence[float]], targets: Sequence[int]) -> Evaluation:
    tp = tn = fp = fn = 0
    for row, target in zip(features, targets):
        pred = model.predict_label(row)
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
    learning_rate: float,
    iterations: int,
) -> Tuple[TrainingResult, Dataset]:
    rows = load_dataset(data)
    dataset = build_features(rows)
    train, test = split_dataset(dataset, train_ratio)

    standardization = compute_standardization(train.features)
    train_scaled = apply_standardization(train.features, standardization)
    test_scaled = apply_standardization(test.features, standardization)

    model = train_logistic_regression(
        train_scaled,
        train.targets,
        learning_rate=learning_rate,
        iterations=iterations,
    )

    evaluation = evaluate_model(model, test_scaled, test.targets)
    result = TrainingResult(
        model=model,
        standardization=standardization,
        evaluation=evaluation,
        train_count=len(train.features),
        test_count=len(test.features),
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
        "--learning-rate",
        type=float,
        default=0.08,
        help="Learning rate for gradient descent (default: 0.08)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of gradient descent iterations (default: 200)",
    )
    return parser.parse_args(argv)


def print_report(
    *,
    dataset: Dataset,
    result: TrainingResult,
) -> None:
    train_size = result.train_count
    test_size = result.test_count
    print("Machine learning pattern discovery for CSCO 5-minute data")
    print("=========================================================")
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
    print("Feature influence (highest magnitude first):")
    descriptions = describe_patterns(dataset.feature_names, result.model.weights)
    for line in descriptions:
        print("  - " + line)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result, dataset = run_training(
        data=args.data,
        train_ratio=args.train_ratio,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
    )
    print_report(dataset=dataset, result=result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
