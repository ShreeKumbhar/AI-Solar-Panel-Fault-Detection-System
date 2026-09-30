package com.example.solarfarm.domain;

import jakarta.persistence.*;
import java.time.Instant;
import com.example.solarfarm.domain.Enums.PanelStatus;

@Entity @Table(name="panels", uniqueConstraints=@UniqueConstraint(columnNames={"farm_id","panel_id"}))
public class Panel {
 @Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id;
 @ManyToOne(fetch=FetchType.LAZY, optional=false) @JoinColumn(name="farm_id") private Farm farm;
 @Column(name="panel_id", nullable=false) private String panelId;
 @Column(name="row_index", nullable=false) private int rowIndex; @Column(name="column_index", nullable=false) private int columnIndex;
 @Column(name="string_id", nullable=false) private String stringId; @Column(name="nominal_power_w",nullable=false) private double nominalPowerW;
 @Column(name="efficiency_factor",nullable=false) private double efficiencyFactor; @Enumerated(EnumType.STRING) @Column(nullable=false) private PanelStatus status;
 @Column(name="created_at",nullable=false) private Instant createdAt=Instant.now(); @Column(name="updated_at",nullable=false) private Instant updatedAt=Instant.now();
 protected Panel() {} public Panel(Farm f,String pid,int r,int c,String s,double n,double e,PanelStatus st){farm=f;panelId=pid;rowIndex=r;columnIndex=c;stringId=s;nominalPowerW=n;efficiencyFactor=e;status=st;}
 public Long getId(){return id;} public Farm getFarm(){return farm;} public String getPanelId(){return panelId;} public int getRowIndex(){return rowIndex;} public int getColumnIndex(){return columnIndex;} public String getStringId(){return stringId;} public double getNominalPowerW(){return nominalPowerW;} public double getEfficiencyFactor(){return efficiencyFactor;} public PanelStatus getStatus(){return status;} public void setStatus(PanelStatus s){status=s;updatedAt=Instant.now();}
}
