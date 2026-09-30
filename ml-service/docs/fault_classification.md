# Phase 5: fault classification

Phase 5 predicts one of the verified five dataset classes: `NORMAL`, `SHORT_CIRCUIT`, `DEGRADATION`, `OPEN_CIRCUIT`, or application-facing `SHADING` for the dataset's `Shadowing` label.

The classifier consumes raw PV measurements plus Phase 3 expected-power and Phase 4 residual/anomaly context. Fault labels, panel IDs, row/column IDs, and future observations are not classifier inputs. Causal historical features are rebuilt on the combined chronological stream before the original train/validation/test boundaries are restored.

Random Forest and XGBoost are compared using validation macro F1, then macro recall as tie-breaker. Class balancing is applied only to training data. The test set is held out until evaluation.

A validation-selected confidence threshold converts low-confidence Phase 4 anomalies to the application output `UNKNOWN_ANOMALY`. This is not a sixth training class and model probabilities are not guaranteed calibrated probabilities.

Run:

```bash
source .venv/bin/activate
python scripts/train_fault_classifier.py
python scripts/evaluate_fault_classifier.py
pytest -q
```

The public dataset's chronological training partition contains no Short-Circuit rows. That class remains in validation/test, but this means the classifier cannot learn Short-Circuit behavior from the mandated training split; this limitation is reported rather than addressed by leakage or random reshuffling.

This phase does not implement the virtual 500-panel farm or any application layer.
