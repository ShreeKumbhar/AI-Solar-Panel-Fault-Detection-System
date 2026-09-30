from __future__ import annotations

import math

from sklearn.ensemble import RandomForestRegressor

MODEL_VERSION = "phase3-baseline-v1"


def create_model(model_name: str, random_state: int = 42):
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",
        )
    raise ValueError(f"Unknown power model: {model_name}")


def model_hyperparameters(model) -> dict:
    parameters = {}
    for key, value in model.get_params().items():
        if isinstance(value, float) and not math.isfinite(value):
            parameters[key] = None
        elif isinstance(value, (str, int, float, bool, type(None))):
            parameters[key] = value
    return parameters
