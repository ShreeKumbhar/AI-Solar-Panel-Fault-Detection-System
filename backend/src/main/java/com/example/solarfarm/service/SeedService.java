package com.example.solarfarm.service;

import com.example.solarfarm.domain.*;
import com.example.solarfarm.domain.Enums.*;
import com.example.solarfarm.repository.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;
import java.nio.file.*;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Component
public class SeedService implements CommandLineRunner {
    private static final DateTimeFormatter CSV_TIMESTAMP_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ssXXX");
    private final FarmService farms; private final PanelRepository panels; private final TelemetryRepository telemetry; private final FaultEventRepository events; private final ModelVersionRepository versions; private final UserRepository users; private final PasswordEncoder encoder;
    @Value("${app.seed-virtual-farm:false}") boolean enabled;
    public SeedService(FarmService f, PanelRepository p, TelemetryRepository t, FaultEventRepository e, ModelVersionRepository v, UserRepository u, PasswordEncoder pe){farms=f;panels=p;telemetry=t;events=e;versions=v;users=u;encoder=pe;}
    public void run(String... args)throws Exception{
        if(!enabled)return; Farm farm=farms.createDefault(); Path root=Path.of("../ml-service/data/processed"); Map<String,Panel> panelMap=new HashMap<>();
        if(panels.count()==0){try(var lines=Files.lines(root.resolve("virtual_panels.csv"))){lines.skip(1).forEach(line->{String[] x=line.split(",");Panel p=panels.save(new Panel(farm,x[0],Integer.parseInt(x[1]),Integer.parseInt(x[2]),x[3],Double.parseDouble(x[4]),Double.parseDouble(x[5]),PanelStatus.HEALTHY));panelMap.put(x[0],p);});}} else panels.findAll().forEach(p->panelMap.put(p.getPanelId(),p));
        if(telemetry.count()==0){List<Telemetry> batch=new ArrayList<>(1000);try(var lines=Files.lines(root.resolve("virtual_telemetry.csv"))){lines.skip(1).forEach(line->{String[] x=line.split(",");Panel p=panelMap.get(x[2]);if(p!=null){batch.add(new Telemetry(p,OffsetDateTime.parse(x[0],CSV_TIMESTAMP_FORMAT).toInstant(),Double.parseDouble(x[8]),Double.parseDouble(x[9]),Double.parseDouble(x[10]),Double.parseDouble(x[11]),Double.parseDouble(x[6]),Double.parseDouble(x[7]),Double.parseDouble(x[12])));if(batch.size()==1000){telemetry.saveAll(batch);batch.clear();}}});}if(!batch.isEmpty())telemetry.saveAll(batch);}
        if(events.count()==0){try(var lines=Files.lines(root.resolve("virtual_fault_events.csv"))){lines.skip(1).forEach(line->{String[] x=line.split(",");Panel p=panelMap.get(x[1]);if(p!=null)events.save(new FaultEvent(x[0],p,OffsetDateTime.parse(x[2],CSV_TIMESTAMP_FORMAT).toInstant(),OffsetDateTime.parse(x[3],CSV_TIMESTAMP_FORMAT).toInstant(),x[4],Double.parseDouble(x[5])));});}}
        if(versions.count()==0)versions.save(new ModelVersion("phase7-inference","phase7-inference-v1","Frozen FastAPI inference service")); if(users.findByEmail("admin@example.com").isEmpty())users.save(new User("admin@example.com",encoder.encode("change-me"),Role.ADMIN));
    }
}
