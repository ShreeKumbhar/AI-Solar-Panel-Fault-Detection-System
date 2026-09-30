package com.example.solarfarm.controller;
import com.example.solarfarm.common.Dtos; import com.example.solarfarm.service.FarmService; import org.springframework.web.bind.annotation.*; import java.util.*;
@RestController @RequestMapping("/api/v1/farms") public class FarmController { private final FarmService service; public FarmController(FarmService s){service=s;} @GetMapping public List<Dtos.FarmResponse> all(){return service.all();} @GetMapping("/{id}") public Dtos.FarmResponse one(@PathVariable Long id){return service.one(id);} }
