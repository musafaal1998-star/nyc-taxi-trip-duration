# %% [markdown]
# # 01 — Data description and exploratory data analysis
# **Competition:** [New York City Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) (Kaggle, 2017)
#
# **Task.** Predict how long a taxi trip in New York City will take (`trip_duration`, in seconds),
# knowing only what is available when the trip starts: pickup time, pickup and dropoff coordinates,
# number of passengers, vendor.
#
# **Metric: RMSLE** (Root Mean Squared Logarithmic Error)
#
# $$\varepsilon = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\big(\log(p_i+1) - \log(a_i+1)\big)^2}$$
#
# In plain words: errors are measured on a log scale, so the metric rewards getting the **ratio** right,
# not the absolute number of seconds. Predicting 10 min for a 5 min trip costs as much as predicting
# 2 h for a 1 h trip. Long trips therefore do not dominate the score, and under-prediction is penalised
# slightly more than over-prediction of the same size.
# Practical consequence: we train every model on `log1p(trip_duration)` with ordinary RMSE —
# RMSE on the log scale **is** RMSLE on the original scale.

# %%
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import FIG_DIR, TARGET, anomaly_report, load_data
from src.features import datetime_features, distance_features

FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100
pd.set_option("display.width", 140)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 1. Data loading

# %%
train, test = load_data()
print(f"train: {train.shape[0]:,} rows x {train.shape[1]} columns")
print(f"test:  {test.shape[0]:,} rows x {test.shape[1]} columns")
print("columns only in train:", sorted(set(train.columns) - set(test.columns)))
train.head()

# %% [markdown]
# ## 2. Description of the features

# %%
meaning = {
    "id": "unique trip identifier",
    "vendor_id": "taxi company that recorded the trip (1 or 2)",
    "pickup_datetime": "date and time the meter was started",
    "dropoff_datetime": "date and time the meter was stopped (train only — it reveals the target, so it is NOT used as a feature)",
    "passenger_count": "number of passengers, entered by the driver",
    "pickup_longitude": "pickup longitude",
    "pickup_latitude": "pickup latitude",
    "dropoff_longitude": "dropoff longitude",
    "dropoff_latitude": "dropoff latitude",
    "store_and_fwd_flag": "Y = trip record held in the vehicle memory and sent later (no server connection)",
    "trip_duration": "TARGET: trip duration in seconds (train only)",
}
kind = {
    "id": "identifier", "vendor_id": "categorical (binary)", "pickup_datetime": "datetime",
    "dropoff_datetime": "datetime", "passenger_count": "discrete numeric / ordinal",
    "store_and_fwd_flag": "categorical (binary)", "trip_duration": "numeric (target)",
}
desc = pd.DataFrame({
    "dtype": train.dtypes.astype(str),
    "feature type": [kind.get(c, "numeric (continuous)") for c in train.columns],
    "n_unique": train.nunique(),
    "missing": train.isna().sum(),
    "meaning": [meaning[c] for c in train.columns],
})
desc

# %%
train.describe().T.round(3)

# %%
print("train period:", train.pickup_datetime.min(), "->", train.pickup_datetime.max())
print("test period: ", test.pickup_datetime.min(), "->", test.pickup_datetime.max())

# %% [markdown]
# Train and test cover the **same six months (Jan–Jun 2016)**: Kaggle split the trips randomly,
# not by time. A random hold-out split is therefore a faithful imitation of the test set (used in notebook 02).

# %% [markdown]
# ## 3. Target distribution

# %%
y = train[TARGET]
print(y.describe().round(1))
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(y.clip(upper=y.quantile(0.99)), bins=100, color="#4C72B0")
axes[0].set(title="trip_duration (clipped at 99th percentile)", xlabel="duration, seconds", ylabel="number of trips")
axes[1].hist(np.log1p(y), bins=100, color="#DD8452")
axes[1].set(title="log1p(trip_duration)", xlabel="log(1 + seconds)", ylabel="number of trips")
save(fig, "01_target_distribution")
plt.show()

# %% [markdown]
# **Findings.**
# * The raw target is extremely right-skewed: median ≈ 11 min, mean ≈ 16 min, but the maximum is ~41 days.
# * After `log1p` the distribution is close to normal (peak around e^6.5 ≈ 11 min).
#   This confirms training on the log scale, which is exactly what RMSLE measures.
# * There is a small bump of very short trips (left tail) and a cluster of ~86 400 s (24 h) trips —
#   meters that were never stopped. These are examined in section 6.

# %% [markdown]
# ## 4. Distributions of the main features

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["vendor_id", "passenger_count", "store_and_fwd_flag"]):
    vc = train[col].value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color="#4C72B0")
    ax.set(title=col, xlabel=col, ylabel="number of trips")
    ax.set_yscale("log")
save(fig, "01_categorical_features")
plt.show()
for col in ["vendor_id", "passenger_count", "store_and_fwd_flag"]:
    print(col, (train[col].value_counts(normalize=True).sort_index() * 100).round(2).to_dict())

# %% [markdown]
# **Findings** (y axis is logarithmic):
# * `vendor_id`: two vendors, fairly balanced (vendor 2 ≈ 53%).
# * `passenger_count`: 71% of trips have one passenger; 0 and 7–9 passengers are rare and
#   most likely data-entry errors (60 trips with 0 passengers).
# * `store_and_fwd_flag`: very imbalanced — only 0.55% "Y".

# %%
dt = datetime_features(train)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, labels in zip(axes, ["hour", "weekday", "month"],
                           [None, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]]):
    vc = dt[col].value_counts().sort_index()
    ax.bar(labels if labels else vc.index.astype(str), vc.values, color="#55A868")
    ax.set(title=f"pickups by {col}", xlabel=col, ylabel="number of trips")
axes[0].set_xticks(range(0, 24, 2))
save(fig, "01_datetime_features")
plt.show()

# %% [markdown]
# **Findings.** Demand is lowest around 5 am and peaks at 6–7 pm; Friday and Saturday are the busiest days,
# Monday the quietest; months are roughly equal (January lowest, March highest). The daily series below shows
# a sharp drop on 23–24 January 2016 (the blizzard), and train and test follow the same pattern day by day.

# %%
fig, ax = plt.subplots(figsize=(13, 3.5))
daily = train.set_index("pickup_datetime").resample("D").size()
daily_test = test.set_index("pickup_datetime").resample("D").size()
ax.plot(daily.index, daily.values, label="train")
ax.plot(daily_test.index, daily_test.values, label="test")
ax.set(title="Trips per day", xlabel="date", ylabel="number of trips")
ax.legend()
save(fig, "01_trips_per_day")
plt.show()

# %%
dist = distance_features(train)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(dist["dist_haversine"].clip(upper=30), bins=100, color="#8172B2")
axes[0].set(title="straight-line (haversine) distance, clipped at 30 km", xlabel="km", ylabel="number of trips")
sample = train.sample(200_000, random_state=0)
axes[1].scatter(sample.pickup_longitude, sample.pickup_latitude, s=0.2, alpha=0.3, color="#C44E52")
axes[1].set(xlim=(-74.05, -73.75), ylim=(40.60, 40.90), title="pickup locations (200k sample)",
            xlabel="longitude", ylabel="latitude")
save(fig, "01_distance_and_map")
plt.show()
print(dist["dist_haversine"].describe().round(2))

# %% [markdown]
# **Findings.** The coordinates draw a map of New York: most pickups are in Manhattan, with two dense spots
# far to the east — LaGuardia (≈ -73.87, 40.77) and JFK (≈ -73.78, 40.64) airports. The median trip is
# ~2 km in a straight line; airport trips create a long right tail.
# The raw coordinates alone do not give the model the distance — it has to be engineered (notebook 03).

# %% [markdown]
# ## 5. Relations between the main features and the target

# %%
train["log_duration"] = np.log1p(train[TARGET])
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
med_h = train.groupby(dt["hour"])[TARGET].median() / 60
axes[0].plot(med_h.index, med_h.values, marker="o")
axes[0].set(title="median trip duration by pickup hour", xlabel="hour of day", ylabel="minutes")
pivot = train.groupby([dt["weekday"], dt["hour"]])[TARGET].median().unstack() / 60
sns.heatmap(pivot, ax=axes[1], cmap="viridis", cbar_kws={"label": "median minutes"})
axes[1].set(title="median duration: weekday x hour", xlabel="hour of day", ylabel="weekday (0 = Monday)")
save(fig, "01_duration_by_time")
plt.show()

# %% [markdown]
# **Findings.** Time of day matters a lot: the median trip takes ~8 min at 6 am and ~12 min at 2–3 pm
# (traffic); on Tue–Thu late mornings it reaches ~13 min. On weekends the morning is the fastest time
# (~8 min at 8 am on Sunday), while trips after midnight are longer than on weekdays (nightlife). Hour × weekday is a strong interaction — tree models will capture it.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
s = train.sample(100_000, random_state=1)
s_dist = distance_features(s)["dist_haversine"]
axes[0].hexbin(np.log1p(s_dist), s["log_duration"], gridsize=70, bins="log", cmap="magma_r")
axes[0].set(title="log duration vs log distance (100k sample)",
            xlabel="log(1 + haversine km)", ylabel="log(1 + seconds)")
print("Spearman corr(distance, duration):", round(dist["dist_haversine"].corr(train[TARGET], method="spearman"), 3))

grp = train.groupby("passenger_count")["log_duration"].agg(["mean", "count"])
axes[1].bar(grp.index.astype(str), np.expm1(grp["mean"]) / 60, color="#4C72B0")
axes[1].set(title="geometric-mean duration by passenger_count", xlabel="passenger_count", ylabel="minutes")
save(fig, "01_duration_vs_distance_passengers")
plt.show()

# %%
pd.DataFrame({
    "vendor: geo-mean minutes": np.expm1(train.groupby("vendor_id")["log_duration"].mean()) / 60,
}).round(2), pd.DataFrame({
    "store_and_fwd: geo-mean minutes": np.expm1(train.groupby("store_and_fwd_flag")["log_duration"].mean()) / 60,
}).round(2)

# %% [markdown]
# **Findings.**
# * **Distance is by far the strongest predictor** (Spearman ρ ≈ 0.79): on the log–log scale the
#   relation is almost linear. Points at high distance with tiny duration are errors.
# * `passenger_count` between 1 and 6 has only a weak effect (10.5–11.5 min). Trips with 0 or 7 passengers
#   last under a minute on average — another sign that these records are errors or cancelled rides.
# * Vendors differ little in typical duration (10.5 vs 10.9 min), but vendor 2 records 2 053 of the 2 061
#   trips longer than 6 h (see below). `store_and_fwd = Y` trips are longer (12.6 vs 10.7 min).

# %% [markdown]
# ## 6. Missing values and anomalies

# %%
print("missing values in train:", int(train.isna().sum().sum()), "| in test:", int(test.isna().sum().sum()))
print("dropoff - pickup == trip_duration for every row:",
      bool(((train.dropoff_datetime - train.pickup_datetime).dt.total_seconds() == train[TARGET]).all()))
pd.concat({"train": anomaly_report(train), "test": anomaly_report(test)}, axis=1)

# %%
long = train[train[TARGET] > 6 * 3600]
print("trips > 6 h by vendor:", long.vendor_id.value_counts().to_dict())
print("median distance of trips > 6 h:", round(distance_features(long)["dist_haversine"].median(), 2), "km")
speed = dist["dist_haversine"] / (train[TARGET] / 3600)
print("trips with straight-line speed > 100 km/h:", int((speed > 100).sum()))

# %% [markdown]
# **Findings / data problems.**
# * **No missing values** in either file; `dropoff_datetime` is perfectly consistent with the target
#   (and must be dropped because it is a leak — it does not exist in test).
# * **Extreme durations:** 2 061 trips > 6 h (mostly ≈ 24 h, 2 053 of them vendor 2, median distance only 2.4 km) —
#   meters that were not switched off; a handful last for weeks (max ≈ 41 days).
# * **Very short trips:** 8 595 trips < 60 s; 5 897 trips have identical pickup and dropoff coordinates
#   (cancelled rides or GPS failures). Both kinds also exist in test (0.4% zero-distance trips).
# * **Impossible speeds:** 175 trips imply > 100 km/h in a straight line.
# * **Passenger count** 0 (60 trips) or 7–9 (5 trips).
# * **Coordinates** outside NYC (~1 000 trips, some in the ocean or other states).
#
# How they are handled (drop / keep / flag) and why is decided and tested in notebook 02.

# %% [markdown]
# ## 7. Summary of the EDA
# 1. The target is log-normal-like → model `log1p(duration)`; this also optimises RMSLE directly.
# 2. Trip distance is the dominant signal but is not a raw feature → engineer distances.
# 3. Time of day × day of week has a strong, non-linear effect (traffic) → datetime features + tree models.
# 4. Location matters beyond distance (airports, Manhattan grid) → coordinate clusters / rotated coordinates.
# 5. A small number of clearly wrong records (≈ 0.2%) exists in train; the test set has the same kinds of errors,
#    so the validation part must keep them to estimate the leaderboard score honestly.
