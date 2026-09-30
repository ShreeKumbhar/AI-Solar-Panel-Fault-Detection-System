from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.simulation.farm_generator import generate_virtual_farm
from src.simulation.simulator_config import SimulationConfig


if __name__ == "__main__":
    result = generate_virtual_farm(SimulationConfig(), project_root)
    print("Generated virtual farm artifacts:")
    for name, path in result.items():
        print(f"- {name}: {path}")
