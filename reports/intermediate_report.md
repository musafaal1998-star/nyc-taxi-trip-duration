# Intermediate status report — NYC Taxi Trip Duration
DAM2026 · checkpoint 13 November · Musa Faal, Mariama Secka, Yamai Senghore

**Competition:** https://www.kaggle.com/competitions/nyc-taxi-trip-duration — predict taxi trip duration (seconds) in
New York City from pickup time, coordinates, passenger count and vendor. **Metric:** RMSLE (error on a log scale, so
relative errors count equally for short and long trips). Models are trained on `log1p(duration)` with RMSE, which
is the same as optimising RMSLE.

## What has been done
**Data.** 1 458 644 training and 625 134 test trips, Jan–Jun 2016, 11 columns (no missing values). Train and test
cover the same period (random split), so a random 80/20 hold-out is used for validation. `dropoff_datetime` is
dropped because it reveals the target.

**Key EDA findings**
- The target is heavily right-skewed (median 11 min, max 41 days); `log1p` makes it close to normal.
- Distance is the strongest driver (Spearman ρ = 0.79 with duration), but it is not a raw feature.
- Strong time pattern: median trip ≈ 8 min at 6 am vs ≈ 12–13 min on weekday late mornings and afternoons;
  weekends behave differently.
- Data problems: 2 061 trips > 6 h (meter left on, nearly all vendor 2), 8 595 trips < 1 min, ~1 000 trips with
  coordinates outside NYC, 65 trips with 0 or > 6 passengers.

**Anomalies.** These rows are dropped from the training part only. The validation and test sets keep them, since
every test row needs a prediction. Cleaning has no measurable effect on the score (±0.0005). The anomalous 0.2% of
validation trips cause 23.6% of the squared error, which sets a floor on the achievable score.

**Preprocessing and features.** Binary columns are encoded as 0/1. For the linear model, hour, weekday, month,
passengers and zones are one-hot encoded and the numeric columns are standardised. Engineered features include
datetime features, haversine and Manhattan distance, bearing, airport flags, PCA-rotated coordinates and KMeans
zones. Everything is fitted inside sklearn pipelines on the training part only.

**Current scores (validation RMSLE)**

| model | RMSLE |
|---|---|
| constant (geometric mean) — trivial baseline | 0.797 |
| rule: mean duration by distance bin | 0.514 |
| Ridge regression | 0.466 |
| Random forest (300k sample) | 0.404 |
| LightGBM, untuned | 0.385 |
| LightGBM, tuned (random search + early stopping) | 0.373 |

Feature effect (LightGBM): raw features 0.460 → + datetime 0.417 → + distances 0.386 → + PCA and zones 0.385.

**Kaggle (late submissions):**

| submission | validation | public LB | private LB |
|---|---|---|---|
| LightGBM, untuned | 0.3846 | 0.3901 | 0.3876 |
| LightGBM, tuned | 0.3729 | 0.3771 | **0.3751** |

The leaderboard scores are 0.002–0.005 worse than validation, and tuning improves both by the same amount (−0.012),
so the validation scheme ranks models correctly.

## Plan until the defense (4 & 11 December)
- Add an external data source: OSRM fastest-route distance and time per trip. This is a public Kaggle dataset
  made for this competition and the best-known improvement here.
- Add aggregate features computed on the training fold: average speed per zone pair and hour, and traffic
  counts per hour.
- Try XGBoost or CatBoost and a simple blend of the best models; tune it the same way.
- Compare validation and leaderboard scores for every submission and explain any gap.
- Prepare the slides (15–20 min).

**Roles so far:** Musa Faal: data description, EDA and anomalies (notebook 01). Mariama Secka: preprocessing,
validation scheme and baselines (notebook 02). Yamai Senghore: feature engineering, models and tuning (notebook 03).

| who | task until the defense |
|---|---|
| Musa Faal | OSRM route features (external data) and their EDA; slides on problem, data and EDA |
| Mariama Secka | aggregate speed / traffic features; validation vs leaderboard analysis; slides on validation and baselines |
| Yamai Senghore | XGBoost / CatBoost, blending and tuning, final submissions; slides on models and results |
