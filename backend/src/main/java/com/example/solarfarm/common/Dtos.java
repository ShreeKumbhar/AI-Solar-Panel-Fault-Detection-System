package com.example.solarfarm.common;

import java.time.Instant; import jakarta.validation.constraints.*;
public final class Dtos { private Dtos() {}
 public record FarmResponse(Long id,String name,String description,int rows,int columns,int panelCount) {}
 public record PanelResponse(String panelId,int rowIndex,int columnIndex,String stringId,double nominalPowerW,double efficiencyFactor,String status) {}
 public record TelemetryRequest(@NotBlank String panelId,@NotNull Instant timestamp,@PositiveOrZero double vdc1,@PositiveOrZero double vdc2,@PositiveOrZero double idc1,@PositiveOrZero double idc2,@PositiveOrZero double irradiance,double pvTemperature) {}
 public record TelemetryResponse(TelemetryRequest telemetry, PredictionResponse prediction) {}
 public record PredictionResponse(Long id,String panelId,Instant timestamp,double actualPowerW,double expectedPowerW,double deviationW,double deviationPercent,boolean anomaly,double anomalyScore,String faultType,double confidence,String severity,String modelVersion) {}
 public record DashboardSummary(long totalPanels,long healthyPanels,long warningPanels,long faultPanels,long offlinePanels,double totalPowerW,double expectedPowerW,double powerLossW,long activeAlerts) {}
 public record FaultEventResponse(String eventId,String panelId,Instant startTime,Instant endTime,String faultType,double severity,String status) {}
 public record AlertResponse(Long id,String panelId,String faultType,String severity,String message,String status,Instant createdAt) {}
 public record LoginRequest(@NotBlank @Email String email,@NotBlank String password) {}
 public record LoginResponse(String token,String tokenType) {}
 public record ErrorResponse(Instant timestamp,int status,String error,String message,String path) {}
}
