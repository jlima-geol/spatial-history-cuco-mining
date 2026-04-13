# Spatial History of Copper Mining — DRC / Zambia Copperbelt

> A multi-notebook geospatial analysis integrating remote sensing, terrain analysis, and historical mining data to map the environmental footprint of copper extraction in Central Africa.

![Synthesis Map](outputs/maps/synthesis/4_5_synthesis_map.png?v=2)


---

## Overview

The Copperbelt — stretching across southern DRC and northern Zambia — is one of the world's most significant copper-producing regions and one of its most environmentally stressed. This project builds a spatial analytical pipeline from raw satellite and geological data to a composite environmental impact assessment, grounding quantitative outputs in the physical and historical geography of the region.

The analysis covers **~210,000 km²**, integrates **132 MRDS mineral occurrences**, and spans a **15-year NDVI time series (2010–2025)** derived from MODIS MOD13Q1.

---

## Key Findings

- **Kolwezi district shows the strongest vegetation decline** in the region: median NDVI trend of −0.008 yr⁻¹, co-located with the densest cluster of active copper mines (Cluster C0, n=42).
- **Precambrian basement rocks host 0.70 mines per 1,000 km²** — by far the highest mineralisation density of any lithological unit, confirming structural geological control on deposit distribution.
- **Mine proximity to water is remarkably consistent**: median distance to nearest river/stream of just 1.5 km across all occurrences, with implications for acid mine drainage risk.
- **Spatial autocorrelation is significant** (Moran's I = 0.205, p = 0.026), confirming that mine clustering is non-random and district-level — not uniformly distributed along the belt.
- **Cluster C2** (near-road, high-density) shows large environmental impact score (0.278) but stable NDVI trend classification (0.0025).

---

## Pipeline Architecture

```
Notebook I  ─── Spatial Foundation
                 MRDS deposits · Infrastructure · SRTM terrain · Geology 
                 → copperbelt_base.gpkg

Notebook II ─── Mining Core Analysis
                 Mine buffers · KDE · DBSCAN clustering · Moran's I
                 accessibility · District-level statistics
                 → mining_analysis.gpkg

Notebook III ── Environmental Impact
                 Sentinel-2 spectral indices ·
                 MODIS NDVI trend (2010–2025) · 
                 Terrain analysis (SRTM) · 
                 Composite impact score
                 → environmental_impact.gpkg
```

Each notebook consumes outputs (.gpkg,.tif) from the prior stage. Most remote data is accessed via cloud-native APIs. Geology and Overture data required manual downloads .

---

## Data Sources

| Dataset | Source | Access |
|---|---|---|
| SRTM 30m DEM | NASA / USGS | `earthaccess` |
| MODIS MOD13Q1 NDVI | NASA LP DAAC | Microsoft Planetary Computer (STAC) |
| Sentinel-2 L2A | ESA Copernicus | CDSE Sentinel Hub Process API |
| Infrastructure | Overture | Overturemaps
| MRDS Mineral Deposits | USGS | Direct download → Parquet cache |
| Geology / Lithology | BGS / regional sources | Local ZIP fallback |
| Administrative boundaries | GADM | `geopandas` |

---

## Tech Stack

```
Python 3.10          geopandas · rioxarray · xarray · rasterio
pip 26.0.1           matplotlib · numpy 2.4.2 · pandas 3.0.0
pysheds 0.3.5        scipy 1.17.0 · scikit-learn (DBSCAN, KDE)
pystac-client        planetary-computer · odc-stac · requests 2.32.3 ·
earthaccess          
```

**CRS:** EPSG:32735 (UTM Zone 35S) for all projected analysis  
**Study bbox:** `[24.0°E, 13.7°S, 29.4°E, 10.0°S]`

---

## Repository Structure

```
spatial-history-cuco-mining/
├── notebooks/
│   ├── I_spatial_foundation.ipynb
│   ├── II_mining_core.ipynb
│   ├── III_environmental_impact.ipynb
│   └── IV_synthesis.ipynb
├── src/
│   └── compute_data.py
│   └── make_aoi.py
│   └── mapstyle.py
│   └── obtain_remote_data.py
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   └── maps/
│   └── figures/
├── environment-analysis.yml
├── environment-download.yml
├── LICENSE
└── README.md
```

---

## Selected Outputs

| Figure | Description |
|---|---|
| `I_01` | Infrastructure & mine logistics map |
| `I_02` | Spatial distribution of mines by ore type |
| `I_03` | SRTM terrain analysis (DEM, slope, TWI, roughness) |
| `II_01` | KDE mineralisation density + district counts |
| `II_02` | DBSCAN spatial clustering — 5 mining districts |
| `II_03` | Moran's I scatter plot — mine counts per district |
| `III_01` | Sentinel-2 NDVI composite |
| `III_02` | MODIS NDVI trend 2010–2025 |
| `III_03` | Composite environmental impact score |
| `III_04` | Synthesis map — clusters · NDVI trend · impact |

---

## Technical Report

For methodology, architectural decisions, and extended results, see [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md).

---

## Project Status
--> **This project is currently under active development.**  
New data, notebooks, and modules are being added progressively.

---

## Author
Jessica Torres, Ph.D.  
GIS & Data Analyst | Geoscientist

---

This study was built as a portfolio project demonstrating end-to-end geospatial analysis with Python, cloud-native remote sensing data access, and spatial statistics.  
**Stack focus:** remote sensing · GIS · spatial data science · environmental analysis · mining
