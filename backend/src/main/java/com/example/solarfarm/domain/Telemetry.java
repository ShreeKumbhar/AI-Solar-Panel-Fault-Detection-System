package com.example.solarfarm.domain;

import jakarta.persistence.*; import java.time.Instant;
@Entity @Table(name="telemetry") public class Telemetry {
 @Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id; @ManyToOne(fetch=FetchType.LAZY,optional=false) @JoinColumn(name="panel_id") private Panel panel; @Column(nullable=false) private Instant timestamp;
 private double vdc1,vdc2,idc1,idc2,irradiance,pvTemperature; @Column(name="actual_power_w",nullable=false) private double actualPowerW; @Column(name="created_at",nullable=false) private Instant createdAt=Instant.now();
 protected Telemetry(){} public Telemetry(Panel p,Instant t,double v1,double v2,double i1,double i2,double ir,double temp,double power){panel=p;timestamp=t;vdc1=v1;vdc2=v2;idc1=i1;idc2=i2;irradiance=ir;pvTemperature=temp;actualPowerW=power;}
 public Long getId(){return id;} public Panel getPanel(){return panel;} public Instant getTimestamp(){return timestamp;} public double getVdc1(){return vdc1;} public double getVdc2(){return vdc2;} public double getIdc1(){return idc1;} public double getIdc2(){return idc2;} public double getIrradiance(){return irradiance;} public double getPvTemperature(){return pvTemperature;} public double getActualPowerW(){return actualPowerW;}
}
