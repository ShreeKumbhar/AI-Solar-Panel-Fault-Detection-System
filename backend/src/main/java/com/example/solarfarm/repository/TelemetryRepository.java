package com.example.solarfarm.repository;
import com.example.solarfarm.domain.*; import org.springframework.data.jpa.repository.JpaRepository; import java.time.Instant; import java.util.*;
public interface TelemetryRepository extends JpaRepository<Telemetry, Long> { List<Telemetry> findByPanel_PanelIdAndTimestampBetweenOrderByTimestampAsc(String panelId, Instant from, Instant to); Optional<Telemetry> findTopByPanelOrderByTimestampDesc(Panel panel); }
