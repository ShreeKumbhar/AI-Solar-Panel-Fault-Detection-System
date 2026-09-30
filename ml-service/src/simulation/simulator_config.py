from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42
    rows: int = 20
    columns: int = 25
    telemetry_samples: int = 96
    timestamp_start: str = "2025-01-01T00:00:00+00:00"
    sample_interval_minutes: int = 15
    nominal_panel_power_w: float = 330.0
    efficiency_min: float = 0.94
    efficiency_max: float = 1.06
    measurement_noise_std: float = 0.005
    source_dataset: str = "data/processed/clean_pv_data.csv"
    output_dir: str = "data/processed"
    report_dir: str = "data/reports"

    @property
    def panel_count(self) -> int:
        return self.rows * self.columns

    @property
    def source_path(self) -> Path:
        return Path(self.source_dataset)
