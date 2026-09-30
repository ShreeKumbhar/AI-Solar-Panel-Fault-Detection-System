from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.engineering import engineer_features
from src.features.feature_config import FEATURE_COLUMNS, FEATURE_SCHEMA, ID_COLUMN, TARGET_COLUMN
from src.preprocessing.split import chronological_split


def build_feature_schema() -> dict:
    return {**FEATURE_SCHEMA}


def _save_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def prepare_ml_dataset(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, Path | dict]:
    """Create a chronological ML-ready dataset and save the related train/validation/test partitions."""
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Clean dataset not found: {source}")

    output_root = Path(output_dir) if output_dir is not None else source.parent
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is required in the input dataset.")

    engineered = engineer_features(df)
    feature_columns = [column for column in FEATURE_COLUMNS if column in engineered.columns]
    ml_dataset = engineered[[ID_COLUMN, *feature_columns, TARGET_COLUMN]].copy()

    train_df, validation_df, test_df = chronological_split(
        ml_dataset,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    ml_dataset_path = _save_csv(ml_dataset, output_root / "ml_dataset.csv")
    train_path = _save_csv(train_df, output_root / "train.csv")
    validation_path = _save_csv(validation_df, output_root / "validation.csv")
    test_path = _save_csv(test_df, output_root / "test.csv")

    feature_schema = build_feature_schema()
    feature_schema["feature_columns"] = feature_columns

    feature_schema_path = output_root / "feature_schema.json"
    feature_schema_path.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

    label_summary = {
        "overall": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "train": train_df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "validation": validation_df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "test": test_df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
    }
    report = {
        "rows": len(df),
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "split_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "partition_sizes": {
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        },
        "label_distribution": label_summary,
    }
    report_path = output_root / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "ml_dataset": ml_dataset_path,
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
        "feature_schema": feature_schema_path,
        "report": report_path,
    }


def run_preprocessing_pipeline(
    input_path: str | Path = "data/processed/clean_pv_data.csv",
    output_dir: str | Path | None = None,
) -> dict[str, Path | dict]:
    project_root = Path(__file__).resolve().parents[2]
    source_path = project_root / input_path if not Path(input_path).is_absolute() else Path(input_path)
    output_root = project_root / output_dir if output_dir is not None and not Path(output_dir).is_absolute() else Path(output_dir) if output_dir is not None else source_path.parent
    return prepare_ml_dataset(source_path, output_dir=output_root)
