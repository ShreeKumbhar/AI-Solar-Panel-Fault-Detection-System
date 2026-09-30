package com.example.solarfarm.domain;

public final class Enums {
    private Enums() {}
    public enum Role { ADMIN, OPERATOR, VIEWER }
    public enum PanelStatus { HEALTHY, WARNING, FAULT, OFFLINE }
    public enum AlertStatus { OPEN, ACKNOWLEDGED, RESOLVED }
    public enum FaultEventStatus { ACTIVE, RESOLVED, ACKNOWLEDGED }
}
