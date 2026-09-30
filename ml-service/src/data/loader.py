from __future__ import annotations

from pathlib import Path

from scipy.io import loadmat


def load_mat_file(path: str | Path):
    """Load a MATLAB .mat file and return the dictionary-like structure."""
    mat_path = Path(path)

    if not mat_path.exists():
        raise FileNotFoundError(
            f"MAT file not found: {mat_path}. Expected dataset under "
            "ml-service/data/raw/pv_fault_dataset/"
        )

    if not mat_path.is_file():
        raise ValueError(f"MAT path is not a file: {mat_path}")

    data = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    return data
