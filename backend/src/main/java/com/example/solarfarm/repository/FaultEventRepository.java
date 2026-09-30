package com.example.solarfarm.repository;
import com.example.solarfarm.domain.*; import org.springframework.data.jpa.repository.JpaRepository; import java.util.*;
public interface FaultEventRepository extends JpaRepository<FaultEvent, Long> { List<FaultEvent> findByPanel_PanelIdOrderByStartTimeDesc(String panelId); }
