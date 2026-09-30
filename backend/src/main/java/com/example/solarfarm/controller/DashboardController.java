package com.example.solarfarm.controller;
import com.example.solarfarm.common.Dtos; import com.example.solarfarm.service.DashboardService; import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api/v1/dashboard") public class DashboardController { private final DashboardService service; public DashboardController(DashboardService s){service=s;} @GetMapping("/summary") public Dtos.DashboardSummary summary(){return service.summary();} }
