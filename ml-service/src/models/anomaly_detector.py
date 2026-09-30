from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve

ANOMALY_FEATURES = [
    "residual_w",
    "absolute_error_w",
    "deviation_percent",
    "rolling_mean_residual",
    "rolling_std_residual",
    "rolling_mean_absolute_error",
    "rolling_mean_deviation",
    "previous_residual",
    "previous_deviation",
    "deviation_trend",
    "irradiance",
    "pv_temperature",
]
MODEL_VERSION = "phase4-isolation-forest-v1"
DEFAULT_ROLLING_WINDOW = 10
NORMAL_LABEL = 0


def calculate_power_deviation(actual_power: pd.Series, expected_power: pd.Series) -> pd.DataFrame:
    actual = pd.to_numeric(actual_power, errors="coerce").astype(float)
    expected = pd.to_numeric(expected_power, errors="coerce").astype(float)
    residual = actual - expected
    valid_expected = expected > 1e-9
    relative = pd.Series(np.where(valid_expected, residual / expected, 0.0), index=actual.index)
    result = pd.DataFrame(index=actual.index)
    result["residual_w"] = residual
    result["absolute_error_w"] = residual.abs()
    result["relative_deviation"] = relative
    result["deviation_percent"] = relative * 100.0
    return result


def build_anomaly_features(frame: pd.DataFrame, rolling_window: int = DEFAULT_ROLLING_WINDOW) -> pd.DataFrame:
    required = ["actual_power_w", "expected_power_w", "irradiance", "pv_temperature"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Anomaly feature input is missing columns: {missing}")
    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1.")

    result = frame.copy()
    if "sample_index" in result.columns and not result["sample_index"].is_monotonic_increasing:
        raise ValueError("Historical anomaly features require non-decreasing sample_index order.")
    deviation = calculate_power_deviation(result["actual_power_w"], result["expected_power_w"])
    for column in deviation.columns:
        result[column] = deviation[column]

    residual = result["residual_w"]
    absolute_error = result["absolute_error_w"]
    deviation_percent = result["deviation_percent"]
    result["rolling_mean_residual"] = residual.rolling(rolling_window, min_periods=1).mean()
    result["rolling_std_residual"] = residual.rolling(rolling_window, min_periods=2).std().fillna(0.0)
    result["rolling_mean_absolute_error"] = absolute_error.rolling(rolling_window, min_periods=1).mean()
    result["rolling_mean_deviation"] = deviation_percent.rolling(rolling_window, min_periods=1).mean()
    result["previous_residual"] = residual.shift(1).fillna(0.0)
    result["previous_deviation"] = deviation_percent.shift(1).fillna(0.0)
    result["deviation_trend"] = result["rolling_mean_deviation"].diff().fillna(0.0)

    numeric = result[ANOMALY_FEATURES].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Anomaly features contain NaN or infinite values.")
    return result


def generate_anomaly_frame(
    frame: pd.DataFrame,
    expected_power_model,
    power_features: list[str],
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.DataFrame:
    required = ["sample_index", "actual_power_w", *power_features, "irradiance", "pv_temperature"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Anomaly input is missing required columns: {missing}")
    if not frame["sample_index"].is_monotonic_increasing:
        raise ValueError("Anomaly input must remain in chronological sample_index order.")
    result = frame.copy()
    result["expected_power_w"] = expected_power_model.predict(result[power_features])
    result = build_anomaly_features(result, rolling_window=rolling_window)
    if "fault_label" in result.columns:
        result["is_anomaly_true"] = (result["fault_label"] != NORMAL_LABEL).astype(int)
    return result


def _validate_feature_matrix(features: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    names = list(feature_names)
    missing = [name for name in names if name not in features.columns]
    if missing:
        raise ValueError(f"Anomaly model input is missing features: {missing}")
    matrix = features[names].apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Anomaly model input contains NaN or infinite values.")
    return matrix


def calibrate_threshold(scores, y_true) -> float:
    anomaly_scores = np.asarray(scores, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    if anomaly_scores.shape != labels.shape:
        raise ValueError("scores and y_true must have the same shape.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y_true must contain only binary anomaly labels.")
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float(np.quantile(anomaly_scores, 0.95))
    precision, recall, thresholds = precision_recall_curve(labels, anomaly_scores)
    if len(thresholds) == 0:
        return float(np.median(anomaly_scores))
    f1 = (2.0 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(f1 == np.max(f1))
    best_index = int(best[np.argmax(precision[:-1][best])])
    return float(thresholds[best_index])


class IsolationForestAnomalyDetector:
    def __init__(
        self,
        feature_names: list[str] | None = None,
        n_estimators: int = 200,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.feature_names = list(feature_names or ANOMALY_FEATURES)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.threshold: float | None = None
        self.raw_score_min: float | None = None
        self.raw_score_max: float | None = None
        self.model: IsolationForest | None = None

    def fit(self, features: pd.DataFrame) -> "IsolationForestAnomalyDetector":
        matrix = _validate_feature_matrix(features, self.feature_names)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination="auto",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(matrix)
        raw = self.model.score_samples(matrix)
        anomaly_raw = -raw
        self.raw_score_min = float(anomaly_raw.min())
        self.raw_score_max = float(anomaly_raw.max())
        return self

    def score_samples(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Anomaly detector has not been fitted.")
        matrix = _validate_feature_matrix(features, self.feature_names)
        return self.model.score_samples(matrix)

    def calculate_anomaly_score(self, features: pd.DataFrame) -> np.ndarray:
        raw_anomaly = -self.score_samples(features)
        if self.raw_score_min is None or self.raw_score_max is None:
            raise RuntimeError("Anomaly score normalization is unavailable before fitting.")
        span = max(self.raw_score_max - self.raw_score_min, 1e-12)
        return np.clip((raw_anomaly - self.raw_score_min) / span, 0.0, 1.0)

    def predict_anomaly(self, features: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        selected_threshold = self.threshold if threshold is None else threshold
        if selected_threshold is None:
            raise ValueError("Anomaly threshold has not been calibrated.")
        return self.calculate_anomaly_score(features) >= selected_threshold

    def calibrate(self, features: pd.DataFrame, y_true) -> float:
        threshold = calibrate_threshold(self.calculate_anomaly_score(features), y_true)
        self.threshold = threshold
        return threshold

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestAnomalyDetector":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Anomaly detector artifact not found: {source}")
        return joblib.load(source)
