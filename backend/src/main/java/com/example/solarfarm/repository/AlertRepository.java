package com.example.solarfarm.repository;
import com.example.solarfarm.domain.*; import com.example.solarfarm.domain.Enums.AlertStatus; import org.springframework.data.jpa.repository.JpaRepository; import java.util.*;
public interface AlertRepository extends JpaRepository<Alert, Long> { List<Alert> findByStatusOrderByCreatedAtDesc(AlertStatus status); long countByStatus(AlertStatus status); }
