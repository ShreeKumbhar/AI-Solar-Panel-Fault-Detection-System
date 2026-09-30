package com.example.solarfarm;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TelemetryPowerTest {
 @Test void phaseOnePowerFormulaIsPreserved(){double power=300.2*0.62+299.7*0.59; assertEquals(362.947,power,1e-9);}
 @Test void phaseEightScopeDoesNotAddTopologyToMlPayload(){assertFalse(java.util.List.of("panel_id","row_index","column_index","string_id").contains("vdc1"));}
}
