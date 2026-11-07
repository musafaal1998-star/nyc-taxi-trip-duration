"""Model pipelines. Every learned step (features, encoders, scalers, model) is inside one
sklearn Pipeline, so fitting on the training fold can never leak validation information."""
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import RANDOM_STATE
from .features import CATEGORICAL_FOR_LINEAR, TaxiFeatures

LGB_QUICK = dict(n_estimators=400, learning_rate=0.1, num_leaves=127, min_child_samples=50,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                 random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


def linear_pipeline(alpha=1.0, groups=TaxiFeatures.ALL_GROUPS):
    """Ridge regression: one-hot for categorical-like columns, standard scaling for the rest."""
    cats = lambda df: [c for c in df.columns if c in CATEGORICAL_FOR_LINEAR]
    nums = lambda df: [c for c in df.columns if c not in CATEGORICAL_FOR_LINEAR]
    prep = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), cats),
        ("scale", StandardScaler(), nums),
    ])
    return Pipeline([("features", TaxiFeatures(groups=groups)), ("prep", prep), ("model", Ridge(alpha=alpha))])


def lgb_pipeline(params=None, groups=TaxiFeatures.ALL_GROUPS):
    """LightGBM: trees need neither scaling nor one-hot encoding."""
    return Pipeline([("features", TaxiFeatures(groups=groups)),
                     ("model", lgb.LGBMRegressor(**(params or LGB_QUICK)))])


def rf_pipeline(groups=TaxiFeatures.ALL_GROUPS, **rf_params):
    params = dict(n_estimators=100, min_samples_leaf=10, max_features=0.5,
                  n_jobs=-1, random_state=RANDOM_STATE)
    params.update(rf_params)
    return Pipeline([("features", TaxiFeatures(groups=groups)), ("model", RandomForestRegressor(**params))])
