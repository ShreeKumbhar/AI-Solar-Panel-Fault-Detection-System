from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.simulation.fault_injector import build_fault_events, inject_faults
from src.simulation.farm_generator import generate_virtual_farm
from src.simulation.fault_injector import FAULT_TYPES
from src.simulation.panel_generator import generate_panel_metadata
from src.simulation.simulator_config import SimulationConfig
from src.simulation.telemetry_generator import generate_healthy_telemetry
from src.models.fault_classifier import CLASSIFIER_FEATURES


def test_topology_is_deterministic_and_complete():
    config = SimulationConfig(rows=20, columns=25, telemetry_samples=8)
    first = generate_panel_metadata(config)
    second = generate_panel_metadata(config)
    assert len(first) == 500
    assert first["panel_id"].tolist() == [f"P{i:04d}" for i in range(1, 501)]
    assert first.equals(second)
    assert first["string_id"].nunique() == 5
    assert first["row_index"].nunique() == 20
    assert first["column_index"].nunique() == 25
    assert first["efficiency_factor"].between(config.efficiency_min, config.efficiency_max).all()


def test_telemetry_has_one_record_per_panel_and_valid_values(tmp_path):
    config = SimulationConfig(rows=2, columns=3, telemetry_samples=5)
    panels = generate_panel_metadata(config)
    telemetry = generate_healthy_telemetry(Path("data/processed/clean_pv_data.csv").resolve(), config, panels)
    assert len(telemetry) == 30
    assert telemetry.groupby("panel_id").size().eq(5).all()
    assert telemetry["timestamp"].notna().all()
    numeric = telemetry.select_dtypes("number")
    assert np.isfinite(numeric.to_numpy()).all()
    assert (numeric[["irradiance", "vdc1", "vdc2", "idc1", "idc2", "actual_power_w"]] >= 0).all().all()


def test_full_generation_and_fault_events(tmp_path):
    config = SimulationConfig(rows=2, columns=3, telemetry_samples=12, source_dataset=str(Path("data/processed/clean_pv_data.csv").resolve()), output_dir="outputs", report_dir="reports")
    artifacts = generate_virtual_farm(config, tmp_path)
    assert all(path.exists() for path in artifacts.values())
    telemetry = __import__("pandas").read_csv(artifacts["telemetry"])
    events = pd.read_csv(artifacts["events"])
    assert set(events["fault_type"]) == set(FAULT_TYPES)
    assert (telemetry.loc[telemetry["is_fault_injected"], "fault_type"] != "NORMAL").all()
    assert telemetry.loc[telemetry["is_fault_injected"], "actual_power_w"].notna().all()


def test_fault_injection_changes_event_window_and_topology_is_not_ml_schema():
    config = SimulationConfig(rows=2, columns=3, telemetry_samples=12)
    panels = generate_panel_metadata(config)
    healthy = generate_healthy_telemetry(Path("data/processed/clean_pv_data.csv").resolve(), config, panels)
    timestamps = pd.DatetimeIndex(pd.to_datetime(healthy["timestamp"].drop_duplicates(), utc=True))
    events = build_fault_events(panels["panel_id"].tolist(), timestamps)
    faulty, _ = inject_faults(healthy, events)
    changed = faulty["actual_power_w"] != healthy["actual_power_w"]
    assert changed.any()
    assert {"panel_id", "row_index", "column_index", "string_id", "event_id"}.isdisjoint(CLASSIFIER_FEATURES)
