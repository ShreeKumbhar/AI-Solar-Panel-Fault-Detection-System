package com.example.solarfarm.ml;

import jakarta.validation.constraints.*; import java.time.Instant;
public final class MlDtos { private MlDtos() {}
 public record MlTelemetry(String panel_id, Instant timestamp, @PositiveOrZero double vdc1, @PositiveOrZero double vdc2, @PositiveOrZero double idc1, @PositiveOrZero double idc2, @PositiveOrZero double irradiance, double pv_temperature) {}
 public record MlPrediction(String panel_id, Instant timestamp, double actual_power_w, double expected_power_w, double deviation_w, double deviation_percent, boolean anomaly, double anomaly_score, String fault_type, double confidence, String severity, String model_version) {}
 public record MlBatch(java.util.List<MlTelemetry> observations) {}
}
