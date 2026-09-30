package com.example.solarfarm.controller;
import org.springframework.web.bind.annotation.*; import java.util.*;
@RestController @RequestMapping("/api/v1/simulation") public class HealthController { private boolean running; @PostMapping("/start") public Map<String,Object> start(){running=true;return Map.of("running",true);} @PostMapping("/stop") public Map<String,Object> stop(){running=false;return Map.of("running",false);} @GetMapping("/status") public Map<String,Object> status(){return Map.of("running",running,"mode","manual-control-only");} }
