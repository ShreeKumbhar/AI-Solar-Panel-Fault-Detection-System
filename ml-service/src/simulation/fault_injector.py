from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FAULT_TYPES = ("SHADING", "DEGRADATION", "SHORT_CIRCUIT", "OPEN_CIRCUIT")


@dataclass(frozen=True)
class FaultEvent:
    event_id: str
    panel_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    fault_type: str
    severity: float


def build_fault_events(panel_ids: list[str], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    if len(timestamps) < 8:
        raise ValueError("At least eight telemetry timestamps are required for fault events.")
    selected = [panel_ids[index % len(panel_ids)] for index in [102, 236, 411, 57]]
    specifications = [
        ("E0001", selected[0], 8, 35, "SHADING", 0.45),
        ("E0002", selected[1], 22, 70, "DEGRADATION", 0.25),
        ("E0003", selected[2], 45, 78, "OPEN_CIRCUIT", 0.90),
        ("E0004", selected[3], 58, 90, "SHORT_CIRCUIT", 0.70),
    ]
    events = []
    for event_id, panel_id, start_index, end_index, fault_type, severity in specifications:
        start = min(start_index, len(timestamps) - 2)
        end = min(end_index, len(timestamps) - 1)
        events.append(
            {
                "event_id": event_id,
                "panel_id": panel_id,
                "start_time": timestamps[start],
                "end_time": timestamps[end],
                "fault_type": fault_type,
                "severity": severity,
            }
        )
    return pd.DataFrame(events)


def _apply_transformation(rows: pd.DataFrame, fault_type: str, severity: float) -> pd.DataFrame:
    transformed = rows.copy()
    if fault_type == "SHADING":
        transformed["idc1"] *= 1.0 - severity
        transformed["idc2"] *= 1.0 - severity
    elif fault_type == "DEGRADATION":
        reduction = 1.0 - 0.5 * severity
        transformed["idc1"] *= reduction
        transformed["idc2"] *= reduction
    elif fault_type == "OPEN_CIRCUIT":
        transformed["idc1"] *= 1.0 - severity
        transformed["idc2"] *= 1.0 - severity
    elif fault_type == "SHORT_CIRCUIT":
        transformed["vdc1"] *= 1.0 - 0.8 * severity
        transformed["idc1"] *= 1.0 + 0.3 * severity
    else:
        raise ValueError(f"Unsupported fault type: {fault_type}")
    transformed["vdc1"] = transformed["vdc1"].clip(lower=0.0)
    transformed["vdc2"] = transformed["vdc2"].clip(lower=0.0)
    transformed["idc1"] = transformed["idc1"].clip(lower=0.0)
    transformed["idc2"] = transformed["idc2"].clip(lower=0.0)
    transformed["actual_power_w"] = transformed["vdc1"] * transformed["idc1"] + transformed["vdc2"] * transformed["idc2"]
    return transformed


def inject_faults(healthy_telemetry: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"timestamp", "panel_id", "actual_power_w", "vdc1", "vdc2", "idc1", "idc2"}
    missing = sorted(required - set(healthy_telemetry.columns))
    if missing:
        raise ValueError(f"Telemetry is missing columns required for fault injection: {missing}")
    result = healthy_telemetry.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["event_id"] = result["event_id"].astype(str)
    for event in events.itertuples(index=False):
        mask = (
            (result["panel_id"] == event.panel_id)
            & (result["timestamp"] >= pd.Timestamp(event.start_time))
            & (result["timestamp"] < pd.Timestamp(event.end_time))
        )
        transformed = _apply_transformation(result.loc[mask], event.fault_type, event.severity)
        result.loc[mask, transformed.columns] = transformed
        result.loc[mask, "fault_type"] = event.fault_type
        result.loc[mask, "fault_severity"] = event.severity
        result.loc[mask, "event_id"] = event.event_id
        result.loc[mask, "is_fault_injected"] = True
    return result, events.copy()
