# Data preprocessing and feature engineering

This phase converts the verified PV dataset into a chronologically ordered, model-ready table without training any final ML model.

## Inputs

- `data/processed/clean_pv_data.csv`

## Outputs

- `data/processed/ml_dataset.csv`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/feature_schema.json`
- `data/processed/preprocessing_report.json`

## Feature approach

The pipeline keeps the original measured signals and adds engineered, domain-informed features such as:

- `power_total_w`
- `power_ratio`
- `voltage_ratio`
- `current_ratio`
- `temperature_irradiance_ratio`
- `irradiance_power_ratio`
- `voltage_balance`
- `current_balance`

These features preserve the original solar panel measurements while exposing the relationships that tend to be informative for fault detection.

## Split strategy

The dataset is partitioned in natural sample order without random shuffling to respect the time-like progression of the collected measurements. The default split is 70/15/15 for train, validation, and test respectively.

## Execution

```bash
source .venv/bin/activate
python scripts/prepare_data.py
```
