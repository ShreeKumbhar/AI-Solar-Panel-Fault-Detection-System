from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .panel_generator import generate_panel_metadata
from .simulator_config import SimulationConfig


def _select_source_rows(source: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    if source.empty:
        raise ValueError("Source PV dataset is empty.")
    sample_count = min(sample_count, len(source))
    positions = np.linspace(0, len(source) - 1, sample_count, dtype=int)
    return source.iloc[positions].reset_index(drop=True)


def generate_healthy_telemetry(source_path: str | Path, config: SimulationConfig, panels: pd.DataFrame | None = None) -> pd.DataFrame:
    source = pd.read_csv(source_path, usecols=["vdc1", "vdc2", "idc1", "idc2", "irradiance", "pv_temperature"])
    source.insert(0, "sample_index", np.arange(len(source), dtype=int))
    source_rows = _select_source_rows(source, config.telemetry_samples)
    panel_metadata = panels if panels is not None else generate_panel_metadata(config)
    rng = np.random.default_rng(config.seed + 1)
    timestamps = pd.date_range(
        start=pd.Timestamp(config.timestamp_start),
        periods=len(source_rows),
        freq=f"{config.sample_interval_minutes}min",
    )
    records = []
    for time_index, source_row in source_rows.iterrows():
        for panel in panel_metadata.itertuples(index=False):
            efficiency = panel.efficiency_factor
            noise = rng.normal(1.0, config.measurement_noise_std, size=4)
            voltage_scale = np.sqrt(efficiency)
            current_scale = np.sqrt(efficiency)
            vdc1 = max(0.0, float(source_row.vdc1) * voltage_scale * noise[0])
            vdc2 = max(0.0, float(source_row.vdc2) * voltage_scale * noise[1])
            idc1 = max(0.0, float(source_row.idc1) / 16.0 * current_scale * noise[2])
            idc2 = max(0.0, float(source_row.idc2) / 16.0 * current_scale * noise[3])
            records.append(
                {
                    "timestamp": timestamps[time_index],
                    "sample_index": int(source_row.sample_index),
                    "panel_id": panel.panel_id,
                    "row_index": panel.row_index,
                    "column_index": panel.column_index,
                    "string_id": panel.string_id,
                    "irradiance": max(0.0, float(source_row.irradiance)),
                    "pv_temperature": float(source_row.pv_temperature),
                    "vdc1": vdc1,
                    "vdc2": vdc2,
                    "idc1": idc1,
                    "idc2": idc2,
                    "actual_power_w": vdc1 * idc1 + vdc2 * idc2,
                    "fault_type": "NORMAL",
                    "fault_severity": 0.0,
                    "event_id": "",
                    "is_fault_injected": False,
                }
            )
    return pd.DataFrame(records)
