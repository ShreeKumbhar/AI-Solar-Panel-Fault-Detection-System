# Solar Farm Backend

Java 21 / Spring Boot backend for the virtual PV farm. It persists telemetry and predictions in PostgreSQL and calls the frozen FastAPI ML service over HTTP.

## Run

Set the variables in `.env.example`, start PostgreSQL, then run:

```bash
mvn spring-boot:run
```

Flyway creates the schema. Set `SEED_VIRTUAL_FARM=true` to import the Phase 6 CSV files once. The import is idempotent by farm/panel and event identifiers.

Swagger: `http://localhost:8080/swagger-ui.html`\nActuator: `http://localhost:8080/actuator/health`

Phase 8 does not implement React, streaming, Kafka, Redis, or deployment infrastructure.
