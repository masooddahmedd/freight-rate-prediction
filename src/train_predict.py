"""Train the final model on all of train_test (Jan-Oct 2025) and write both prediction files.

Model: LightGBM on log(rate per mile), L1 loss, features = lane, equipment, distance, weight, geography and
day-of-week/month. quote_signal and market_index are deliberately left out (see backtest.py). Known rate
errors are removed from training only, and recent loads are weighted more (90-day decay).

Run from the repo root:  python src/train_predict.py
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import add_features, as_categories, category_levels, clean, load

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT

FEATURES = ["distance", "log_distance", "weight", "equipment", "pickup", "delivery", "lane", "pickup_lat",
            "pickup_lon", "delivery_lat", "delivery_lon", "bearing", "dow", "dom", "weight_missing",
            "weight_was_negative"]
PARAMS = dict(objective="l1", learning_rate=0.03, num_leaves=63, min_child_samples=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, cat_smooth=20, verbose=-1)
ROUNDS = 1500
SEEDS = (7, 11, 23)
RECENCY_DAYS = 90


def drop_rate_errors(df, lo=0.6, hi=1.7):
    rpm = df["posted_rate"] / df["distance"]
    norm = rpm.groupby([df["lane"], df["equipment"]], observed=True).transform("median")
    return df[(rpm / norm).between(lo, hi)]


def city_coordinates(*frames):
    """December inputs carry city names only; coordinates come from the cities seen in the other files."""
    rows = []
    for f in frames:
        rows.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        rows.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(rows).groupby("city")[["lat", "lon"]].median()


def main():
    train_raw = load(DATA / "train_test.csv")
    valid_raw = load(DATA / "validation.csv")
    dec_raw = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])

    coords = city_coordinates(train_raw, valid_raw)
    dec = dec_raw.copy()
    for side in ("pickup", "delivery"):
        dec[f"{side}_lat"] = dec[side].map(coords["lat"])
        dec[f"{side}_lon"] = dec[side].map(coords["lon"])

    train, weight_medians, _ = clean(train_raw)
    valid, _, _ = clean(valid_raw, weight_medians)
    dec_clean, _, _ = clean(dec.drop(columns=["predicted_rate"]), weight_medians)

    train, valid, dec_clean = add_features(train), add_features(valid), add_features(dec_clean)
    levels = category_levels(train, valid, dec_clean)
    train, valid, dec_clean = (as_categories(f, levels) for f in (train, valid, dec_clean))

    train = drop_rate_errors(train)
    age_days = (train["date"].max() - train["date"]).dt.days.to_numpy()
    weights = np.exp(-age_days / RECENCY_DAYS)
    target = np.log(train["posted_rate"] / train["distance"])

    valid_log_rpm = np.zeros(len(valid))
    dec_log_rpm = np.zeros(len(dec_clean))
    for seed in SEEDS:
        model = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(train[FEATURES], target, weight=weights),
                          num_boost_round=ROUNDS)
        valid_log_rpm += model.predict(valid[FEATURES]) / len(SEEDS)
        dec_log_rpm += model.predict(dec_clean[FEATURES]) / len(SEEDS)

    valid_pred = pd.DataFrame({
        "load_id": valid["load_id"],
        "predicted_rate": (valid["distance"] * np.exp(valid_log_rpm)).round(2),
    })
    template = pd.read_csv(DATA / "validation_predictions_template.csv")[["load_id"]]
    valid_pred = template.merge(valid_pred, on="load_id", how="left")
    valid_pred.to_csv(OUT / "validation_predictions.csv", index=False)

    dec_out = dec_raw.copy()
    dec_out["predicted_rate"] = (dec_out["distance"] * np.exp(dec_log_rpm)).round(2)
    dec_out["date"] = dec_out["date"].dt.strftime("%Y-%m-%d")
    dec_out.to_csv(DATA / "december_chart_inputs.csv", index=False)

    print(f"trained on {len(train):,} loads ({len(train_raw) - len(train):,} rate errors excluded)")
    print(f"validation predictions: {len(valid_pred):,} rows, median ${valid_pred.predicted_rate.median():,.0f}")
    print(f"December Lexington > Fort Wayne: ${dec_out.predicted_rate.min():,.2f} - ${dec_out.predicted_rate.max():,.2f}")


if __name__ == "__main__":
    main()
