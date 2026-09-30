package com.example.solarfarm.repository;
import com.example.solarfarm.domain.Farm; import org.springframework.data.jpa.repository.JpaRepository;
public interface FarmRepository extends JpaRepository<Farm, Long> {}
