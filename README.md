# Mapping the spatial history of Copper and Cobalt mining in the DR Congo

![modis_ndvi](outputs/maps/env_impact/02_modis_ndvi_trend.png?v=2)

## Overview
This project sits at the intersection of the geochemistry of the Katanga Copperbelt and the 
tools of spatial analysis.<br>

My doctoral research (KU Leuven) focused on fluid geochemistry at the Katanga Copperbelt — 
the world's largest stratiform/stratabound copper-cobalt deposit. That work was analytical and 
geochemical. This project asks a different question: *what does the spatial and historical 
footprint of this mining region look like when mapped through open-source data?*<br>

It is also an exercise in building a reproducible, open GIS workflow — connecting scientific 
context to cartographic storytelling using only freely available tools and datasets.

## Objectives
- Map the distribution of copper mines and deposits using open-source datasets
- Analyze spatial relationships with geology, elevation, and infrastructure
- Produce clear cartographic outputs inspired by narrative geographies
- Demonstrate a reproducible GIS workflow using only open-source tools

## Data Sources
- SENTINEL-2 (Copernicus)
- LANDSAT (Microsoft Planetary Computer)
- MODIS NDVI (Microsoft Planetary Computer)
- Overture (for roads, railways, settlements)
- USGS MRDS (for mineral occurrences)
- SRTM DEM (NASADEM/SRTM)
- GADM (administrative boundaries; geodata.ucdavis.edu)

## Workflow
1. Data collection and preprocessing
2. Terrain and geological context integration
3. Spatial analysis (kernel density, clustering, distance metrics)
4. Cartographic design in QGIS
5. Narrative synthesis

## Tools
- Python (geopandas, matplotlib, numpy, pandas, pip, rasterio, rioxarray, scipy, shapely, xarray)
- QGIS
- Overture / Overpass API

## Results
- High-resolution maps in `outputs/maps/`
- Spatial analysis notebooks in `notebooks/`
- Reproducible scripts in `src/`

## Project Status
--> **This project is currently under active development.**  
New data, notebooks, and modules are being added progressively.

## Author
Jessica Torres, Ph.D.  
GIS & Data Analyst | Geoscientist
