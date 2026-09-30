package com.example.solarfarm.repository;
import com.example.solarfarm.domain.*; import org.springframework.data.jpa.repository.JpaRepository; import java.util.*;
public interface PredictionRepository extends JpaRepository<Prediction, Long> { List<Prediction> findByPanel_PanelIdOrderByTimestampDesc(String panelId); Optional<Prediction> findTopByPanelOrderByTimestampDesc(Panel panel); }
