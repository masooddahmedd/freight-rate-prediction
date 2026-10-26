import numpy as np
import pandas as pd

CATEGORICAL = ["pickup", "delivery", "lane", "equipment"]


def load(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def clean(df, weight_medians=None, daily_market=None):
    """Fix the obvious data-entry problems. Medians are learned on training data and reused at predict time."""
    df = df.copy()

    # Negative weights are sign flips: the magnitudes sit in the normal 15k-47k lb range.
    df["weight_was_negative"] = (df["weight"] < 0).astype(int)
    df["weight"] = df["weight"].abs()

    if weight_medians is None:
        weight_medians = df.groupby("equipment")["weight"].median().to_dict()
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["weight"] = df["weight"].fillna(df["equipment"].map(weight_medians))

    # market_index is a market-wide reading with small per-load noise, so a gap is best filled with that day's median.
    if "market_index" in df.columns:
        if daily_market is None:
            daily_market = df.groupby("date")["market_index"].median()
        df["market_missing"] = df["market_index"].isna().astype(int)
        filled = df["date"].map(daily_market)
        df["market_index"] = df["market_index"].fillna(filled)
        df["market_index"] = df["market_index"].fillna(df["market_index"].median())

    return df, weight_medians, daily_market


def add_features(df, origin=pd.Timestamp("2025-01-01")):
    df = df.copy()
    df["lane"] = df["pickup"] + " > " + df["delivery"]
    df["log_distance"] = np.log(df["distance"])
    df["days"] = (df["date"] - origin).dt.days
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    # Rough direction of travel: headhaul vs backhaul markets price differently.
    df["d_lat"] = df["delivery_lat"] - df["pickup_lat"]
    df["d_lon"] = df["delivery_lon"] - df["pickup_lon"]
    df["bearing"] = np.degrees(np.arctan2(df["d_lat"], df["d_lon"]))

    if "quote_signal" in df.columns:
        df["base_rate"] = df["distance"] * df["quote_signal"]
    return df


def as_categories(df, categories):
    df = df.copy()
    for col in CATEGORICAL:
        df[col] = pd.Categorical(df[col], categories=categories[col])
    return df


def category_levels(*frames):
    levels = {}
    for col in CATEGORICAL:
        values = pd.concat([f[col] for f in frames if col in f]).dropna().unique()
        levels[col] = sorted(values)
    return levels
