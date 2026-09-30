from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .fault_injector import build_fault_events, inject_faults
from .panel_generator import generate_panel_metadata
from .simulator_config import SimulationConfig
from .telemetry_generator import generate_healthy_telemetry


def generate_virtual_farm(config: SimulationConfig, project_root: str | Path = ".") -> dict[str, Path]:
    root = Path(project_root)
    source_path = root / config.source_path
    output_dir = root / config.output_dir
    report_dir = root / config.report_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    panels = generate_panel_metadata(config)
    healthy = generate_healthy_telemetry(source_path, config, panels)
    timestamps = pd.DatetimeIndex(pd.to_datetime(healthy["timestamp"].drop_duplicates(), utc=True))
    events = build_fault_events(panels["panel_id"].tolist(), timestamps)
    telemetry, events = inject_faults(healthy, events)

    panel_path = output_dir / "virtual_panels.csv"
    telemetry_path = output_dir / "virtual_telemetry.csv"
    events_path = output_dir / "virtual_fault_events.csv"
    panels.to_csv(panel_path, index=False)
    telemetry.to_csv(telemetry_path, index=False)
    events.to_csv(events_path, index=False)

    report = {
        "panel_count": len(panels),
        "rows": config.rows,
        "columns": config.columns,
        "panel_id_range": [panels["panel_id"].min(), panels["panel_id"].max()],
        "string_count": int(panels["string_id"].nunique()),
        "telemetry_record_count": len(telemetry),
        "time_range": [telemetry["timestamp"].min().isoformat(), telemetry["timestamp"].max().isoformat()],
        "fault_event_count": len(events),
        "fault_distribution": telemetry["fault_type"].value_counts().sort_index().to_dict(),
        "panel_efficiency_statistics": panels["efficiency_factor"].describe().round(6).to_dict(),
        "dataset_source": "clayton-h-costa/pv_fault_dataset via Phase 1 clean_pv_data.csv",
        "random_seed": config.seed,
        "simulation_configuration": config.__dict__,
        "timestamp_assumption": "Synthetic UTC timestamps are assigned because the public dataset exposes sample order but no timestamp column.",
        "ml_boundary": "panel_id, topology fields, string_id, event_id, and injected fault fields are simulation metadata, not ML features.",
    }
    report_path = report_dir / "virtual_farm_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return {"panels": panel_path, "telemetry": telemetry_path, "events": events_path, "report": report_path}
