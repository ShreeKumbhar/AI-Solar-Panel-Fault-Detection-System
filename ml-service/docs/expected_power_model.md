# Phase 3: expected power prediction

Phase 3 trains a regression model for expected healthy PV power. It does not perform anomaly detection or fault classification.

## Target and healthy data

The target is `actual_power_w`, calculated in Phase 1 as the sum of the two measured string powers. The normal operating label observed in the dataset is `fault_label == 0`; only those rows are retained for the dedicated power-model train, validation, and test files. Filtering is applied independently to the existing chronological Phase 2 partitions, so sample order is preserved and future rows are not moved into training.

## Leakage prevention

The model uses only the deterministic feature order:

- `vdc1`, `vdc2`, `idc1`, `idc2`
- `irradiance`, `pv_temperature`

The following are excluded because they are the target, direct components or duplicates of the target, target-derived ratios, labels, identifiers, or conservative exclusions of engineered ratios: `power_string_1_w`, `power_string_2_w`, `actual_power_w`, `power_total_w`, `power_ratio`, `irradiance_power_ratio`, `fault_label`, `sample_index`, `voltage_ratio`, `current_ratio`, `voltage_balance`, `current_balance`, and `temperature_irradiance_ratio`.

Raw voltage/current channels remain because they are independent measured columns; no precomputed power column is supplied to the model.

## Models and selection

The baseline models are `RandomForestRegressor` and `XGBRegressor`. They use fixed, reproducible baseline configurations rather than a hyperparameter search. The model with the lowest validation RMSE is selected, with validation MAE as the tie-breaker. The test set is not used for selection. After selection, the winner is retrained on train plus validation healthy rows, then evaluated once on the untouched healthy test partition.

## Evaluation

The evaluation script writes MAE, RMSE, R2, residual statistics, largest errors, and low/medium/high irradiance-group metrics. It also generates actual-versus-expected, scatter, residual-distribution, residual-versus-irradiance, and feature-importance plots. Relative error is left undefined when actual power is effectively zero.

## Commands

```bash
source .venv/bin/activate
python scripts/train_power_model.py
python scripts/evaluate_power_model.py
pytest -q
```

Training writes the selected model to `artifacts/expected_power_model.joblib`, along with model metadata and the deterministic feature schema. Reports are written under `data/reports/`.

## Limitations

The results describe this public dataset and its observed normal-label definition. They do not establish universal solar-farm performance, deployment readiness, or fault-detection capability. Fault detection and virtual-farm behavior belong to later phases. Feature importance is an interpretability aid, not evidence of causal relationships.
