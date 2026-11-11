# NYC Taxi Trip Duration — DAM2026 exam project

**Competition:** [New York City Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) (Kaggle, 2017; late submissions are accepted)

## Task
Predict the duration of a taxi trip in New York City (`trip_duration`, seconds) from what is known when the
trip starts: pickup date and time, pickup and dropoff coordinates, number of passengers, vendor and the
`store_and_fwd_flag`. Train: 1 458 644 trips, test: 625 134 trips, both from January–June 2016.

**Target:** `trip_duration` in seconds. Models are trained on `log1p(trip_duration)`.

**Metric:** Root Mean Squared Logarithmic Error

$$\text{RMSLE} = \sqrt{\tfrac{1}{n}\textstyle\sum_i \big(\log(p_i+1) - \log(a_i+1)\big)^2}$$

The metric compares predictions and true values on a log scale, so it rewards getting the *ratio* right:
a 5-minute error on a 5-minute trip is as bad as a 1-hour error on a 1-hour trip. Training with RMSE on
`log1p(y)` optimises RMSLE directly.

## Team
| member | role |
|---|---|
| Musa Faal | data loading and description, exploratory data analysis, missing values and anomalies (`src/data.py`, notebook 01) |
| Mariama Secka | preprocessing, validation scheme, baselines, first submission (`src/models.py`, notebook 02) |
| Yamai Senghore | feature engineering, model comparison, hyper-parameter tuning, final submission (`src/features.py`, notebook 03) |

## Results (20% hold-out validation, RMSLE — lower is better)
| model | validation RMSLE | note |
|---|---|---|
| constant (geometric mean) | 0.7966 | trivial baseline |
| rule: mean by distance bin | 0.5138 | 20 quantile bins of haversine distance |
| Ridge (tuned alpha) | 0.4661 | alpha=0.1, full training part |
| Random forest (300k sample) | 0.4040 | 100 trees, leaf>=10, 138s |
| LightGBM untuned (300k sample) | 0.3929 | 400 trees, 14s |
| LightGBM untuned (full training part) | 0.3846 | 400 trees |
| LightGBM tuned (full training part) | 0.3729 | lr 0.03, 3064 trees, 352s |

Kaggle (late submissions, 7 November 2026):

| submission | validation RMSLE | public LB | private LB |
|---|---|---|---|
| `02_lightgbm_untuned.csv` | 0.3846 | 0.39008 | 0.38762 |
| `03_lightgbm_tuned.csv` | 0.3729 | 0.37713 | **0.37512** |

## Repository structure
```
src/
  data.py        loading, anomaly report, training-set cleaning, metric
  features.py    feature engineering (sklearn transformer: datetime, distances, PCA rotation, KMeans zones)
  models.py      model pipelines: Ridge, Random forest, LightGBM
notebooks/
  01_data_and_eda.ipynb                       data description, EDA, anomalies
  02_preprocessing_validation_baseline.ipynb  anomaly handling, validation scheme, preprocessing, baselines, 1st submission
  03_features_models_tuning.ipynb             feature ablation, 3 model families, tuning, comparison, final submission
  *.py                                        the same notebooks as plain scripts (jupytext), easy to diff
reports/
  figures/       all plots used in the report / slides
  *.csv          result tables written by the notebooks
data/            put train.csv and test.csv here (not committed)
submissions/     generated submission files (not committed)
```

## How to run
```bash
pip install -r requirements.txt
# download train.csv and test.csv from the competition page into data/
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_and_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing_validation_baseline.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_features_models_tuning.ipynb
```
Run time on a 2-core laptop: about 15 s, 4 min and 40 min (most of it is the random search) respectively.

## Validation scheme
Random 80/20 hold-out (`random_state=42`) the test set is a random sample of the same period, so a random
split imitates it. The split is made first; cleaning, encoders, scalers, PCA and KMeans are fitted on the
training part only (inside sklearn pipelines). The validation part is never cleaned, because the test set
contains the same bad records. Hyper-parameters are tuned with 3-fold CV inside the training part.

## References
- Data: NYC Taxi and Limousine Commission, via the Kaggle competition.
- Ideas commonly discussed in the competition's public kernels (haversine/Manhattan distance, PCA rotation of
  coordinates, coordinate clustering) were re-implemented from scratch in `src/features.py`.
