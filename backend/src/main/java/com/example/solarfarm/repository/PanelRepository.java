package com.example.solarfarm.repository;
import com.example.solarfarm.domain.Panel; import com.example.solarfarm.domain.Enums.PanelStatus; import org.springframework.data.jpa.repository.JpaRepository; import java.util.*;
public interface PanelRepository extends JpaRepository<Panel, Long> { Optional<Panel> findByPanelId(String panelId); List<Panel> findByStatus(PanelStatus status); long countByStatus(PanelStatus status); }
