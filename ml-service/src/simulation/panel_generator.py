from __future__ import annotations

import numpy as np
import pandas as pd

from .simulator_config import SimulationConfig


def generate_panel_metadata(config: SimulationConfig) -> pd.DataFrame:
    if config.rows < 1 or config.columns < 1:
        raise ValueError("Farm rows and columns must be positive.")
    if config.efficiency_min <= 0 or config.efficiency_min > config.efficiency_max:
        raise ValueError("Efficiency bounds must be positive and ordered.")

    rng = np.random.default_rng(config.seed)
    records = []
    for panel_number in range(config.panel_count):
        row_index = panel_number // config.columns + 1
        column_index = panel_number % config.columns + 1
        panel_id = f"P{panel_number + 1:04d}"
        records.append(
            {
                "panel_id": panel_id,
                "row_index": row_index,
                "column_index": column_index,
                "string_id": f"S{panel_number // 100 + 1:02d}",
                "nominal_power_w": config.nominal_panel_power_w,
                "efficiency_factor": float(rng.uniform(config.efficiency_min, config.efficiency_max)),
                "status": "HEALTHY",
            }
        )
    return pd.DataFrame(records)
