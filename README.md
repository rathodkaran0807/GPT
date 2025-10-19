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

Mine directional edges with an indicator-rich neural network that learns from
OHLC momentum, MACD, RSI, ATR, stochastic oscillators, and volatility regimes:

```bash
python machine_learning_patterns.py --data 'BATS_CSCO, 5.csv'
```

Tune the workflow with `--train-ratio`, `--learning-rate`, `--epochs`,
`--hidden-units`, `--batch-size`, `--patience`, `--min-delta`, `--train-window`,
or `--l2` to adjust the optimization schedule, sample window, early-stopping
tolerance, and regularization. Control signal selectivity via
`--signal-threshold` (probability required before opening a trade) and
`--top-signals` to cap execution to the highest-confidence setups.

The report now scans multi-bar trades launched on bullish signals across a
stop-loss / take-profit / trailing-stop grid, surfacing the richest scenario on
the held-out set. Adjust the evaluation grid with `--stop-losses`,
`--take-profits`, `--trailing-stops`, and the bankroll with `--capital`.
Enforce a minimum win rate using `--min-win-rate`, and require a specific
risk/reward profile (for example, `--risk-reward 2 --ratio-tolerance 0.05` for
1:2 trades) to filter the presented scenarios.
