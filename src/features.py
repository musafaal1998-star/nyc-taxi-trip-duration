"""Feature engineering as sklearn transformers, so everything that is *learned*
(PCA rotation, coordinate clusters) is fitted on the training fold only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from .data import NYC_LAT, NYC_LON, RANDOM_STATE

EARTH_R_KM = 6371.0
AIRPORTS = {"jfk": (-73.7781, 40.6413), "lga": (-73.8740, 40.7769), "ewr": (-74.1745, 40.6895)}

RAW_FEATURES = [
    "vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "store_and_fwd_flag",
]


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def manhattan_km(lon1, lat1, lon2, lat2):
    """Grid ("taxicab") distance: north-south leg + east-west leg."""
    return haversine_km(lon1, lat1, lon1, lat2) + haversine_km(lon1, lat2, lon2, lat2)


def bearing_deg(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x))


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns with binary categoricals encoded as 0/1 (no learning involved)."""
    out = df[RAW_FEATURES].copy()
    out["vendor_id"] = (out["vendor_id"] == 2).astype(np.int8)
    out["store_and_fwd_flag"] = (out["store_and_fwd_flag"] == "Y").astype(np.int8)
    return out


def datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["pickup_datetime"]
    out = pd.DataFrame(index=df.index)
    out["hour"] = t.dt.hour
    out["minute_of_day"] = t.dt.hour * 60 + t.dt.minute
    out["weekday"] = t.dt.weekday
    out["week_hour"] = out["weekday"] * 24 + out["hour"]
    out["month"] = t.dt.month
    out["day_of_year"] = t.dt.dayofyear
    out["is_weekend"] = (out["weekday"] >= 5).astype(np.int8)
    return out


def distance_features(df: pd.DataFrame) -> pd.DataFrame:
    args = (df["pickup_longitude"], df["pickup_latitude"], df["dropoff_longitude"], df["dropoff_latitude"])
    out = pd.DataFrame(index=df.index)
    out["dist_haversine"] = haversine_km(*args)
    out["dist_manhattan"] = manhattan_km(*args)
    out["log_dist_haversine"] = np.log1p(out["dist_haversine"])
    out["bearing"] = bearing_deg(*args)
    out["center_longitude"] = (df["pickup_longitude"] + df["dropoff_longitude"]) / 2
    out["center_latitude"] = (df["pickup_latitude"] + df["dropoff_latitude"]) / 2
    for name, (lon, lat) in AIRPORTS.items():
        near_p = haversine_km(df["pickup_longitude"], df["pickup_latitude"], lon, lat) < 2
        near_d = haversine_km(df["dropoff_longitude"], df["dropoff_latitude"], lon, lat) < 2
        out[f"airport_{name}"] = (near_p | near_d).astype(np.int8)
    inside = (
        df["pickup_longitude"].between(*NYC_LON) & df["pickup_latitude"].between(*NYC_LAT)
        & df["dropoff_longitude"].between(*NYC_LON) & df["dropoff_latitude"].between(*NYC_LAT)
    )
    out["flag_outside_nyc"] = (~inside).astype(np.int8)
    return out


class TaxiFeatures(BaseEstimator, TransformerMixin):
    """Builds the model matrix from the raw dataframe.

    groups: which feature groups to add on top of the raw columns.
        'datetime', 'distance', 'pca' (rotated coordinates), 'clusters' (KMeans zones)
    """

    ALL_GROUPS = ("datetime", "distance", "pca", "clusters")

    def __init__(self, groups=ALL_GROUPS, n_clusters=50):
        self.groups = groups
        self.n_clusters = n_clusters

    def _coords(self, df):
        """All pickup and dropoff points stacked, only points inside NYC."""
        p = df[["pickup_latitude", "pickup_longitude"]].to_numpy()
        d = df[["dropoff_latitude", "dropoff_longitude"]].to_numpy()
        pts = np.vstack([p, d])
        ok = (pts[:, 0] > NYC_LAT[0]) & (pts[:, 0] < NYC_LAT[1]) & (pts[:, 1] > NYC_LON[0]) & (pts[:, 1] < NYC_LON[1])
        return pts[ok]

    def fit(self, X, y=None):
        pts = self._coords(X)
        rng = np.random.RandomState(RANDOM_STATE)
        sample = pts[rng.choice(len(pts), size=min(500_000, len(pts)), replace=False)]
        if "pca" in self.groups:
            # PCA on lat/lon rotates the map so that Manhattan's street grid aligns with the axes.
            self.pca_ = PCA(n_components=2).fit(sample)
        if "clusters" in self.groups:
            self.kmeans_ = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=10_000,
                                           n_init=3, random_state=RANDOM_STATE).fit(sample)
        return self

    def transform(self, X):
        parts = [basic_features(X)]
        if "datetime" in self.groups:
            parts.append(datetime_features(X))
        if "distance" in self.groups:
            parts.append(distance_features(X))
        p = X[["pickup_latitude", "pickup_longitude"]].to_numpy()
        d = X[["dropoff_latitude", "dropoff_longitude"]].to_numpy()
        if "pca" in self.groups:
            pp, dp = self.pca_.transform(p), self.pca_.transform(d)
            parts.append(pd.DataFrame({
                "pickup_pca0": pp[:, 0], "pickup_pca1": pp[:, 1],
                "dropoff_pca0": dp[:, 0], "dropoff_pca1": dp[:, 1],
                "dist_pca_manhattan": np.abs(dp[:, 0] - pp[:, 0]) + np.abs(dp[:, 1] - pp[:, 1]),
            }, index=X.index))
        if "clusters" in self.groups:
            parts.append(pd.DataFrame({
                "pickup_cluster": self.kmeans_.predict(p),
                "dropoff_cluster": self.kmeans_.predict(d),
            }, index=X.index))
        return pd.concat(parts, axis=1)


# Columns treated as categorical by the linear model (one-hot encoded there).
CATEGORICAL_FOR_LINEAR = ["hour", "weekday", "month", "passenger_count", "pickup_cluster", "dropoff_cluster"]
