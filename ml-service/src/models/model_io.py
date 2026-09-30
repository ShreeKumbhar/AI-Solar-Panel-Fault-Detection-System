from __future__ import annotations

from pathlib import Path

import joblib


def save_model(model, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def load_model(path: str | Path):
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Model artifact not found: {source}")
    return joblib.load(source)
