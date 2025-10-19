# GPT

## BATS_CSCO Dataset Inspection

Run the helper script to inspect the dataset and produce summary statistics:

```bash
python analyze_bats_csco.py
```

To export the summary as JSON, provide the `--json` flag:

```bash
python analyze_bats_csco.py --json report.json
```

## Renko Brick Size Optimizer

Evaluate Renko brick sizes for the 5-minute closing dataset:

```bash
python renko_brick_optimizer.py --data 'BATS_CSCO, 5.csv'
```

Customize the search range by supplying `--min-brick`, `--max-brick`, and the
number of `--steps` to sample within that range. Display additional candidate
results with the `--top` flag.

## Supertrend Backtest

Run a Supertrend crossover backtest (default parameters: period 10, multiplier
1.5, starting capital $1,000):

```bash
python supertrend_backtest.py --data 'BATS_CSCO, 5.csv'
```

Override the Supertrend configuration or capital by providing the `--period`,
`--multiplier`, or `--capital` flags.

## Machine Learning Pattern Discovery

Train a lightweight logistic regression model to uncover directional patterns
from engineered OHLC features:

```bash
python machine_learning_patterns.py --data 'BATS_CSCO, 5.csv'
```

Adjust the training split, learning rate, or number of optimization iterations
with `--train-ratio`, `--learning-rate`, and `--iterations` respectively.
