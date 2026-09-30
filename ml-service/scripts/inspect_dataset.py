from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_mat_file


def _safe_shape(value):
    try:
        return np.asarray(value).shape
    except Exception:
        return "unknown"


def _safe_dtype(value):
    try:
        return np.asarray(value).dtype
    except Exception:
        return type(value)


def _safe_min_max(value):
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return None, None
        if np.issubdtype(arr.dtype, np.number):
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None, None
            return float(arr.min()), float(arr.max())
    except Exception:
        pass
    return None, None


def inspect_mat_file(file_path: str | Path):
    path = Path(file_path)
    print(f"DATASET: {path.name}")
    print(f"File path: {path}")

    data = load_mat_file(path)
    keys = [k for k in data.keys() if not k.startswith("__")]
    print(f"Available MAT keys: {keys}")

    for key in keys:
        value = data[key]
        shape = _safe_shape(value)
        dtype = _safe_dtype(value)
        minimum, maximum = _safe_min_max(value)

        print(f"Variable: {key}")
        print(f"Shape: {shape}")
        print(f"dtype: {dtype}")
        print(f"min: {minimum}")
        print(f"max: {maximum}")
        print("-" * 40)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "data" / "raw" / "pv_fault_dataset"

    for dataset_name in ("dataset_elec.mat", "dataset_amb.mat"):
        file_path = dataset_dir / dataset_name
        if not file_path.exists():
            raise FileNotFoundError(
                f"Expected dataset file is missing: {file_path}. "
                "Place the raw MAT files under ml-service/data/raw/pv_fault_dataset/"
            )
        print("=" * 80)
        inspect_mat_file(file_path)
        print("=" * 80)
