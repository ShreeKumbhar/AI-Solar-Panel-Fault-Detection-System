package com.example.solarfarm.controller;
import com.example.solarfarm.common.Dtos; import com.example.solarfarm.service.TelemetryService; import org.springframework.web.bind.annotation.*; import java.util.*;
@RestController @RequestMapping("/api/v1/panels") public class PredictionController { private final TelemetryService service; public PredictionController(TelemetryService s){service=s;} @GetMapping("/{panelId}/predictions") public List<Dtos.PredictionResponse> predictions(@PathVariable String panelId){return service.predictions(panelId);} }
