# Phase 6: virtual 500-panel solar farm

The public PV dataset is a real two-string plant dataset, not synchronized monitoring data from 500 independently instrumented panels. This phase constructs a deterministic virtual topology of 20 rows by 25 columns, with IDs `P0001` through `P0500`.

Real Phase 1 measurements provide the environmental and operating backbone. Because the source has sample order but no timestamp column, the simulator selects evenly spaced source observations and assigns synthetic UTC timestamps at a configured interval. Panel efficiency variation, measurement noise, topology, and fault-event timing are controlled simulation additions.

Each virtual panel receives deterministic metadata and telemetry. Healthy telemetry scales the real electrical behavior to panel-level power and applies small efficiency/noise adjustments. Fault events are deterministic and cover SHADING, DEGRADATION, SHORT_CIRCUIT, and OPEN_CIRCUIT. Their transformations are deliberately simple, bounded assumptions based on the dataset's observed fault categories; they do not claim physical field fidelity.

Topology fields (`panel_id`, `row_index`, `column_index`, `string_id`) and synthetic event fields are localization/simulation metadata only. They must not be added to Phase 3-5 model feature schemas.

## Run

```bash
source .venv/bin/activate
python scripts/generate_virtual_farm.py
pytest -q
```

Outputs are written to `data/processed/virtual_panels.csv`, `data/processed/virtual_telemetry.csv`, `data/processed/virtual_fault_events.csv`, and `data/reports/virtual_farm_report.json`.

Due to the limited availability of publicly accessible datasets containing synchronized measurements at individual panel level, a virtual solar-farm topology was constructed. Real PV measurements were used as the underlying operating conditions, while panel-level variations and fault scenarios were generated through controlled data augmentation and fault injection.

These are simulation results. The topology is synthetic, event timing is controlled, and injected faults do not establish real-world field performance or prove that the public dataset represents a 500-panel farm.
