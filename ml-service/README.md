# AI-Based Intelligent Solar Panel Fault and Anomaly Detection System

## Project objective

This Phase 1 project focuses on dataset acquisition, inspection, validation, and exploratory data analysis for a real photovoltaic fault dataset. No machine-learning model is trained in this phase.

The public dataset provides real PV measurements and fault labels. The later 500-panel solar farm will be a virtual software topology built from the empirical dataset and controlled data augmentation.

## Dataset source

Repository: https://github.com/clayton-h-costa/pv_fault_dataset

Expected raw files:

- ml-service/data/raw/pv_fault_dataset/dataset_elec.mat
- ml-service/data/raw/pv_fault_dataset/dataset_amb.mat

## How to download the dataset

From the project root:

```bash
cd ml-service
rm -rf data/raw/pv_fault_dataset
git clone https://github.com/clayton-h-costa/pv_fault_dataset.git data/raw/pv_fault_dataset
```

## Python setup

This project targets Python 3.11 for compatibility with the scientific stack.

```bash
cd ml-service
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run dataset inspection

```bash
python scripts/inspect_dataset.py
```

## Open the notebook

```bash
jupyter notebook
```

Then open `notebooks/01_dataset_exploration.ipynb`.

## Generate processed dataset

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
from src.data.cleaner import load_verified_dataframe, save_clean_dataset

base = Path('data/raw/pv_fault_dataset')
df = load_verified_dataframe(base / 'dataset_elec.mat', base / 'dataset_amb.mat')
save_clean_dataset(df, Path('data/processed/clean_pv_data.csv'))
print(df.head())
PY
```

## Run tests

```bash
pytest
```

## Train and evaluate the expected-power model

```bash
source .venv/bin/activate
python scripts/train_power_model.py
python scripts/evaluate_power_model.py
pytest -q
```

Phase 3 trains and evaluates the healthy expected-power regression model only. See [docs/expected_power_model.md](docs/expected_power_model.md) for target selection, leakage prevention, model selection, and limitations.

## Train and evaluate anomaly detection

```bash
source .venv/bin/activate
python scripts/train_anomaly_detector.py
python scripts/evaluate_anomaly_detector.py
pytest -q
```

Phase 4 uses Phase 3 expected power, chronological residual features, and a validation-calibrated Isolation Forest. See [docs/anomaly_detection.md](docs/anomaly_detection.md) for the feature list, threshold method, evaluation rules, and limitations.

## Train and evaluate fault classification

```bash
source .venv/bin/activate
python scripts/train_fault_classifier.py
python scripts/evaluate_fault_classifier.py
pytest -q
```

Phase 5 predicts the verified five-class fault target using the Phase 3/4 inference context. See [docs/fault_classification.md](docs/fault_classification.md) for feature selection, validation model selection, uncertainty handling, and the chronological training limitation.

## Generate the virtual 500-panel farm

```bash
source .venv/bin/activate
python scripts/generate_virtual_farm.py
pytest -q
```

Phase 6 creates a deterministic 20x25 virtual topology, real-data-backed telemetry, and controlled fault events. It does not retrain Phases 3-5 or add application infrastructure. See [docs/virtual_farm.md](docs/virtual_farm.md) for the simulation assumptions and limitations.

## Run the FastAPI ML inference service

```bash
source .venv/bin/activate
python scripts/run_api.py
```

The service exposes `/api/v1/health`, `/api/v1/model`, `/api/v1/predict`, `/api/v1/predict/batch`, and Swagger documentation at `/docs`. It loads the frozen Phase 3-5 artifacts once at startup and does not retrain models. See [docs/fastapi_service.md](docs/fastapi_service.md).

## Output files

- `data/raw/pv_fault_dataset/dataset_elec.mat`
- `data/raw/pv_fault_dataset/dataset_amb.mat`
- `data/processed/clean_pv_data.csv`
- `data/reports/dataset_quality_report.json`
- `data/reports/dataset_report.json`
- `data/reports/fault_distribution.png`
- `data/reports/irradiance_distribution.png`
- `data/reports/temperature_distribution.png`
- `data/reports/power_distribution.png`
- `data/reports/power_vs_irradiance.png`
- `data/reports/voltage_by_fault.png`
- `data/reports/current_by_fault.png`
- `data/reports/power_by_fault.png`
- `data/reports/power_over_time.png`
- `data/reports/correlation_matrix.png`
- `data/reports/power_model_comparison.json`
- `data/reports/power_model_evaluation.json`
- `artifacts/expected_power_model.joblib`
- `artifacts/expected_power_metadata.json`
- `artifacts/expected_power_feature_schema.json`
- `artifacts/anomaly_detector.joblib`
- `artifacts/anomaly_detector_metadata.json`
- `data/reports/anomaly_detection_comparison.json`
- `data/reports/anomaly_detection_evaluation.json`
- `artifacts/fault_classifier.joblib`
- `artifacts/fault_classifier_metadata.json`
- `artifacts/fault_classifier_feature_schema.json`
- `data/reports/fault_classification_comparison.json`
- `data/reports/fault_classification_evaluation.json`
- `data/processed/virtual_panels.csv`
- `data/processed/virtual_telemetry.csv`
- `data/processed/virtual_fault_events.csv`
- `data/reports/virtual_farm_report.json`
- `docs/fastapi_service.md`

## Known limitations

- The raw dataset is intentionally kept untouched.
- This phase does not train or tune any ML model.
- The dataset contains real photovoltaic measurements and fault labels, but no timestamps were found in the MAT files; sample indices are used where time-based plotting is necessary.
- The project is intentionally limited to Phase 1 data acquisition, inspection, validation, and EDA.

## Recommended commit sequence

1. `chore: initialize ML project structure`
2. `feat: add MATLAB dataset loader`
3. `feat: add dataset validation`
4. `feat: add PV power calculation`
5. `feat: add dataset EDA`
6. `feat: save processed dataset and quality report`

## Project structure

```text
ml-service/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── pv_fault_dataset/
│   │       ├── dataset_elec.mat
│   │       └── dataset_amb.mat
│   ├── processed/
│   └── reports/
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── loader.py
│       ├── validator.py
│       └── cleaner.py
├── scripts/
│   └── inspect_dataset.py
├── tests/
│   └── test_dataset.py
└── .venv/
```
