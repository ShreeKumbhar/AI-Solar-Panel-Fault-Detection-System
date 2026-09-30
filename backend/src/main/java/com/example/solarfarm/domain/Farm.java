package com.example.solarfarm.domain;

import jakarta.persistence.*;
import java.time.Instant;

@Entity @Table(name="farms")
public class Farm {
 @Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id;
 @Column(nullable=false) private String name; private String description;
 @Column(name="rows", nullable=false) private int rows;
 @Column(name="columns", nullable=false) private int columns;
 @Column(name="panel_count", nullable=false) private int panelCount;
 @Column(name="created_at", nullable=false) private Instant createdAt=Instant.now();
 @Column(name="updated_at", nullable=false) private Instant updatedAt=Instant.now();
 protected Farm() {} public Farm(String name,String description,int rows,int columns,int panelCount){this.name=name;this.description=description;this.rows=rows;this.columns=columns;this.panelCount=panelCount;}
 public Long getId(){return id;} public String getName(){return name;} public String getDescription(){return description;} public int getRows(){return rows;} public int getColumns(){return columns;} public int getPanelCount(){return panelCount;}
}
