# Freight Rate Prediction

Predicts the posted rate for a freight load from its lane, equipment, distance, weight and date.

The final model is LightGBM trained on log rate per mile with an L1 loss. It weights recent loads more heavily (90-day decay) and averages three seeds. `quote_signal` and `market_index` are left out on purpose, because both made forward-in-time backtests worse. The full reasoning is in `reports/Freight_Rate_Report.docx`.

## Results

These are forward-in-time backtests with the same two-month horizon as the real task. MAPE is measured on normal loads, excluding the roughly 1.3% of rates that look like entry errors.

| Model | Train Jan-Jun, test Jul-Aug | Train Jan-Aug, test Sep-Oct |
| --- | --- | --- |
| distance x quote_signal | 15.71% | 9.80% |
| lane median rate per mile | 3.83% | 3.44% |
| LightGBM + market_index + quote_signal | 3.90% | 3.12% |
| LightGBM + market_index | 2.28% | 4.85% |
| LightGBM, no market signals | 2.50% | 2.18% |
| **Final: no market signals, recency-weighted** | **2.47%** | **1.90%** |

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
source .venv/bin/activate     # macOS / Linux
python -m pip install -r requirements.txt
```

## Run

Run everything from the repo root:

```bash
# 1. train on all of train_test.csv, write validation_predictions.csv and fill data/december_chart_inputs.csv
python src/train_predict.py

# 2. validate both files and draw scorer_results/candidate_december.png
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

# 3. optional: reproduce the model comparison table (reports/backtest_results.csv)
python src/backtest.py

# 4. optional: rebuild the report from the backtest table and the December chart
python src/make_report.py
```

Training takes about a minute on a laptop CPU.

## Layout

```
data/                     input files as provided; december_chart_inputs.csv holds the filled predictions
src/features.py           cleaning (negative/missing weights, missing market_index) and features
src/train_predict.py      final model and both prediction files
src/backtest.py           time-based model comparison
src/make_report.py        builds the DOCX report
score.py                  Spotter's scorer, unchanged
validation_predictions.csv
reports/                  report, backtest table and charts
scorer_results/           candidate_december.png
```
