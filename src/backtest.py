"""Reproduces the model comparison in the report.

Two forward-in-time backtests that mimic the real task (predict ~2 months past the training data):
  window A: train Jan-Jun, test Jul-Aug
  window B: train Jan-Aug, test Sep-Oct
Error is reported on all test loads and on "clean" test loads (excluding the ~1.4% of rates that are
0.2-0.4x or 2.5-4.5x their lane norm, which no model can predict).

Run from the repo root:  python src/backtest.py
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import add_features, as_categories, category_levels, clean, load
from train_predict import FEATURES, PARAMS, RECENCY_DAYS, drop_rate_errors

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"A: Jul-Aug": pd.Timestamp("2025-07-01"), "B: Sep-Oct": pd.Timestamp("2025-09-01")}
ROUNDS = 1500


def split(raw, cutoff):
    tr = raw[raw.date < cutoff]
    te = raw[(raw.date >= cutoff) & (raw.date < cutoff + pd.Timedelta(days=61))]
    tr, wmed, _ = clean(tr)
    te, _, _ = clean(te, wmed)
    tr, te = add_features(tr), add_features(te)
    levels = category_levels(tr, te)
    return as_categories(tr, levels), as_categories(te, levels)


def score(y, p, mask):
    ape = np.abs(p - y) / y
    return {"MAPE all %": ape.mean() * 100, "MAE all $": np.abs(p - y).mean(),
            "MAPE clean %": ape[mask].mean() * 100, "median APE clean %": np.median(ape[mask]) * 100}


def fit_predict(tr, te, feats, weight=None):
    target = np.log(tr.posted_rate / tr.distance)
    model = lgb.train({**PARAMS, "seed": 7}, lgb.Dataset(tr[feats], target, weight=weight), num_boost_round=ROUNDS)
    return te.distance.values * np.exp(model.predict(te[feats]))


def main():
    raw = load(ROOT / "data" / "train_test.csv")
    rows = []
    for name, cutoff in WINDOWS.items():
        tr_all, te = split(raw, cutoff)
        tr = drop_rate_errors(tr_all)
        lane_norm = (tr.posted_rate / tr.distance).groupby([tr.lane, tr.equipment], observed=True).median()
        expected = np.array([lane_norm.get(k, np.nan) for k in zip(te.lane, te.equipment)])
        y = te.posted_rate.values
        ratio = (y / te.distance.values) / expected
        clean_mask = (ratio > 0.6) & (ratio < 1.7)
        age = (tr.date.max() - tr.date).dt.days.to_numpy()

        candidates = {
            "distance x quote_signal": te.distance.values * te.quote_signal.values,
            "lane median rate/mile x distance": te.distance.values * np.where(np.isnan(expected), np.nanmedian(expected), expected),
            "LightGBM + market_index + quote_signal": fit_predict(tr, te, FEATURES + ["market_index", "market_missing", "quote_signal"]),
            "LightGBM + market_index": fit_predict(tr, te, FEATURES + ["market_index", "market_missing"]),
            "LightGBM, no market signals": fit_predict(tr, te, FEATURES),
            "LightGBM, no market signals, recency-weighted (final)": fit_predict(tr, te, FEATURES, np.exp(-age / RECENCY_DAYS)),
        }
        for model_name, p in candidates.items():
            rows.append({"window": name, "model": model_name, **score(y, p, clean_mask)})
        print(f"{name}: train {len(tr):,} loads, test {len(te):,} loads ({(~clean_mask).sum()} rate errors in test)")

    table = pd.DataFrame(rows).round(2)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "backtest_results.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
