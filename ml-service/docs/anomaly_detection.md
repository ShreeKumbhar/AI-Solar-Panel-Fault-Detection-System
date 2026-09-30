# Phase 4: anomaly detection

Phase 4 compares actual PV power with the Phase 3 expected healthy-power prediction and detects abnormal behavior. It stops at the binary `NORMAL`/`ANOMALY` decision; fault-type classification is Phase 5.

## Data contract

The Phase 2 partitions are used directly because Phase 3 `power_model/*.csv` files contain healthy rows only. The existing `sample_index` is the chronological ordering field. No timestamp was present in the public MAT files, so no timestamp is invented.

Phase 3 inference uses `vdc1`, `vdc2`, `idc1`, `idc2`, `irradiance`, and `pv_temperature`. Phase 4 then calculates residuals and deviation percentages. Fault labels are excluded from every detector feature and used only for validation threshold calibration and evaluation.

## Historical features

The detector uses residual and context behavior: residual, absolute error, deviation percentage, causal rolling residual/error/deviation statistics, previous residual/deviation, short-term deviation trend, irradiance, and PV temperature. The default rolling window is 10 records. Rolling windows are trailing, never centered, and use only current/past values.

## Detector and calibration

The primary detector is `IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)`. Its raw `score_samples` output is retained; because higher raw values indicate more normal observations, the application anomaly score is the min-max normalized negative raw score, where higher means more anomalous. It is not a probability.

A residual baseline uses absolute deviation percentage. Both thresholds are calibrated on validation labels by maximizing F1, using recall as the tie-breaker. The test partition is untouched until final evaluation. The selected detector is recorded in `data/reports/anomaly_detection_comparison.json`.

## Commands

```bash
source .venv/bin/activate
python scripts/train_anomaly_detector.py
python scripts/evaluate_anomaly_detector.py
pytest -q
```

## Limitations

The public dataset represents one two-string PV plant and not 500 independently monitored physical panels. The future 500-panel farm is a virtual topology. Isolation Forest scores are not probabilities. Dataset imbalance, low irradiance, measurement noise, transients, and model prediction error can affect alerts. Fault labels are used only for calibration/evaluation and no fault classifier is implemented in this phase.
