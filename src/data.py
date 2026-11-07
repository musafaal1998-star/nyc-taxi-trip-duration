"""Data loading and cleaning for the NYC Taxi Trip Duration competition."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"
SUB_DIR = ROOT / "submissions"

TARGET = "trip_duration"
RANDOM_STATE = 42

# Rough bounding box of New York City (incl. the airports JFK / LGA / EWR).
NYC_LON = (-74.30, -73.70)
NYC_LAT = (40.50, 41.00)

# Training-only cleaning thresholds (see notebook 02 for the reasoning).
MAX_DURATION = 6 * 3600     # trips longer than 6 h are meters left running
MIN_PASSENGERS, MAX_PASSENGERS = 1, 6


def load_data(data_dir: Path = DATA_DIR):
    """Read train and test with parsed datetimes."""
    train = pd.read_csv(data_dir / "train.csv",
                        parse_dates=["pickup_datetime", "dropoff_datetime"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["pickup_datetime"])
    return train, test


def inside_nyc(df: pd.DataFrame) -> pd.Series:
    """True when both pickup and dropoff are inside the NYC bounding box."""
    ok = pd.Series(True, index=df.index)
    for p in ("pickup", "dropoff"):
        ok &= df[f"{p}_longitude"].between(*NYC_LON)
        ok &= df[f"{p}_latitude"].between(*NYC_LAT)
    return ok


def anomaly_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count every type of anomaly we look for. Works on train (with target) and test."""
    rows = {
        "pickup or dropoff outside NYC box": ~inside_nyc(df),
        "passenger_count == 0": df["passenger_count"] == 0,
        "passenger_count > 6": df["passenger_count"] > 6,
        "pickup == dropoff coordinates": (df["pickup_longitude"] == df["dropoff_longitude"])
        & (df["pickup_latitude"] == df["dropoff_latitude"]),
    }
    if TARGET in df:
        rows[f"{TARGET} > 6 h"] = df[TARGET] > MAX_DURATION
        rows[f"{TARGET} < 60 s"] = df[TARGET] < 60
    out = pd.DataFrame({"count": {k: int(v.sum()) for k, v in rows.items()}})
    out["share_%"] = (100 * out["count"] / len(df)).round(3)
    return out


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Drop clear recording errors from the TRAINING part only.

    The test set cannot be cleaned (every row needs a prediction), so the
    validation part is always left untouched to mirror the test set.
    """
    keep = (
        (df[TARGET] <= MAX_DURATION)
        & df["passenger_count"].between(MIN_PASSENGERS, MAX_PASSENGERS)
        & inside_nyc(df)
    )
    return df.loc[keep].copy()


def rmsle(y_true, y_pred) -> float:
    """Competition metric: Root Mean Squared Logarithmic Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def rmse(a, b) -> float:
    """RMSE. On log1p-transformed values it equals RMSLE on the original scale."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
