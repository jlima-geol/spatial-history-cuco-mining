"""
obtain_remote_data.py
---------------------
Remote data access pipeline for the Spatial History of Copper Mining (DRC) project.

This module provides functions to load various datasets relevant to the project,
including satellite imagery (Sentinel-2, Landsat, MODIS), infrastructure (Overture Maps),
mineral occurrences (USGS MRDS), administrative boundaries (GADM), elevation data (SRTM/NASADEM),
and geological units (OneGeology WFS with fallback to USGS ScienceBase).

"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from pystac_client import Client

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

import xarray as xr
import rioxarray as rxr

import earthaccess
import pystac_client
import planetary_computer
import odc.stac

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonic bounding box for the Copperbelt region (DRC + Zambia).
# [minx, miny, maxx, maxy] = [lon_west, lat_south, lon_east, lat_north]
COPPERBELT_BBOX: list[float] = [24.0, -13.7, 29.4, -10.0]
CRS_PROJ                     = "EPSG:32735"
CRS_GEO                      = "EPSG:4326"

# Root directory of the project (folder containing /src, /data, /notebooks)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_TEMP      = PROJECT_ROOT / "data" / "temp"
MAPS_OUT       = PROJECT_ROOT / "outputs" / "maps"
FIGURES_OUT    = PROJECT_ROOT / "outputs" / "figures"

# Create the directories if they don't exist
for _d in (DATA_RAW, DATA_PROCESSED, DATA_TEMP, MAPS_OUT, FIGURES_OUT):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sentinel-2 (ESA / Copernicus — AWS STAC)
# ---------------------------------------------------------------------------

# -----------------------------------------
# Config
# -----------------------------------------

SENTINEL_DIR = DATA_RAW / "sentinel"

ESSENTIAL_BANDS = {
    "B02":  "blue",
    "B03": "green",
    "B04":   "red",
    "B08":   "nir",
    "B11": "swir16",
    "B12": "swir22",
    "SCL":   "scl",
}

STAC_URL = "https://earth-search.aws.element84.com/v1"


# -----------------------------------------
# Download Function
# -----------------------------------------

def download_sentinel_scene(
    bbox: list[float],
    start: str = "2023-01-01",
    end: str = "2023-12-31",
    band: str = "red",
):
    """
    Searches for a Sentinel-2 L2A scene in the AWS STAC and downloads the essential bands.
    Saves to DATA_RAW/sentinel/<scene_id>/.
    """

    catalog = Client.open(STAC_URL)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        max_items=1,
    )

    items = list(search.get_items())
    if not items:
        raise ValueError("No scene found.")

    item = items[0]
    scene_id = item.id
    scene_dir = SENTINEL_DIR / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene found: {scene_id}")
    print(f"Saving to: {scene_dir}")

    # -----------------------------------------
    # Download bands
    # -----------------------------------------

    for name, code in ESSENTIAL_BANDS.items():
        if code not in item.assets:
            print(f"  ⚠️ Band {code} ({name}) not available in this scene.")
            continue

        asset = item.assets[code]
        url = asset.href
        out_path = scene_dir / f"{scene_id}_{code}.tif"

        if out_path.exists():
            print(f"  ✓ {name} ({code}) already exists — skipping.")
            continue

        print(f"  Downloading {name} ({code})...")
        r = requests.get(url, stream=True)

        if r.status_code != 200:
            print(f"    ❌ Error downloading {code}: HTTP {r.status_code}")
            continue

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"    Saved: {out_path.name}")

    print("\n Download completed")
    print(f"Files saved to: {scene_dir}")

    return scene_dir

# -----------------------------------------
# Load Bands Function
# -----------------------------------------

def load_sentinel_bands(scene_dir: Path) -> dict[str, xr.DataArray] | None:
    """
    Load the essential bands of a Sentinel-2 scene saved in:
        DATA_RAW/sentinel/<scene_id>/

    Returns:
        dict {band_name: DataArray}
    """

    if not scene_dir.exists():
        print(f"[WARN] Scene directory not found: {scene_dir}")
        return None

    print(f"\n Loading bands from scene: {scene_dir.name}")

    bands: dict[str, xr.DataArray] = {}

    for name, code in ESSENTIAL_BANDS.items():
        pattern = f"*{code}*.tif"
        matches = sorted(scene_dir.glob(pattern))

        if not matches:
            print(f"  ⚠️ Band absent: {code} ({name})")
            continue

        tif = matches[0]
        print(f"  Opening {name} ({code}) → {tif.name}")

        da = rxr.open_rasterio(tif, masked=True).squeeze()

        # Reprojetar para o CRS do projeto
        da = da.rio.reproject(CRS_PROJ)

        bands[name] = da

        print(f"    shape={da.shape}, crs={da.rio.crs}")

    if not bands:
        print("[WARN] No bands loaded.")
        return None

    print("\n All essential bands loaded!")
    return bands


# ---------------------------------------------------------------------------
# Landsat Collection 2 Level-2 (Microsoft Planetary Computer)
# ---------------------------------------------------------------------------

def load_landsat(
    bbox: list[float] = COPPERBELT_BBOX,
    start: str = "2023-01-01",
    end: str = "2023-12-31",
    band: str = "red",
):
    """
    Load Landsat C2L2 via Microsoft Planetary Computer.

    Parameters
    ----------
    band : name of the asset (e.g., 'red', 'nir08', 'swir16', 'qa_pixel')
    """
    try:
        import planetary_computer as pc
        import rioxarray as rxr
        from pystac_client import Client

        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(
            collections=["landsat-c2-l2"], bbox=bbox,
            datetime=f"{start}/{end}", max_items=1,
        )

        item = next(search.get_items(), None)
        if item is None:
            raise ValueError("No Landsat item found for this bbox and period.")

        signed = pc.sign(item)
        print(f"  Landsat — band '{band}' loaded: {item.id}")
        return rxr.open_rasterio(signed.assets[band].href)

    except Exception as e:
        print(f"[ERROR] load_landsat(): {e}")
        raise


# ---------------------------------------------------------------------------
# MODIS (NASA Earthdata)
# ---------------------------------------------------------------------------
MODIS_CACHE   = DATA_PROCESSED / "modis" / "annual_composites"
MODIS_CACHE.mkdir(parents=True, exist_ok=True)

MODIS_YEARS   = list(range(2010, 2026))
MODIS_PRODUCT = "modis-13Q1-061"        # collection ID in Planetary Computer
MODIS_BAND    = "250m_16_days_NDVI"     # asset name confirmed
MODIS_NODATA  = -3000                   # fill value of the product


def _open_pc_catalog() -> pystac_client.Client:
    """Open the Planetary Computer catalog (no login required)."""
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

# -----------------------------------------
# Load Function
# -----------------------------------------

def load_modis_ndvi_year(
    year: int,
    bbox: list[float],
    collection: str = MODIS_PRODUCT,
    band: str = MODIS_BAND,
    target_crs: str = CRS_PROJ,
    resolution: int = 250,
) -> xr.DataArray:
    """
    Download all MOD13Q1 granules for the year via Planetary Computer (COG),
    stack them over time, apply quality mask, and return the annual median.

    No login required, no HDF4 required.
    """
    catalog = _open_pc_catalog()

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = list(search.get_items())

    if not items:
        raise ValueError(f"No granule found for {year} in bbox={bbox}")

    print(f"    {year}: {len(items)} granules found")

    # odc.stac.load creates mosaic + clip in bbox automaticaly.
    # Forcing output CRS since MODIS items have proj:code = EPSG:None.
    ds = odc.stac.load(
        items,
        bands=[band],
        bbox=bbox,
        crs=target_crs,
        resolution=resolution,
        nodata=MODIS_NODATA,
        groupby="solar_day",   # groups acquisitions from the same day before stacking
    )

    da = ds[band]              # DataArray with dim "time"

    # Scale: int × 10 000 → float NDVI [-1, 1]
    da = da.where(da > MODIS_NODATA).astype(float) / 10_000.0

    # Annual Median (robust to residual clouds)
    composite = da.median(dim="time").compute()
    composite = composite.rio.write_crs(target_crs)

    return composite

# -----------------------------------------
# Build annual composites function
# -----------------------------------------

def build_modis_annual_composites(
    years: list[int] = MODIS_YEARS,
    cache_dir: Path = MODIS_CACHE,
    bbox: list[float] = COPPERBELT_BBOX,
) -> dict[int, xr.DataArray]:
    """
    Build annual NDVI composites for each year in `years`.
    Cache in cache_dir/<year>.tif — only download missing years.
    """
    composites: dict[int, xr.DataArray] = {}

    for year in years:
        cache_path = cache_dir / f"{year}.tif"

        if cache_path.exists():
            print(f"  ✓ MODIS {year}: cache hit")
            composites[year] = rxr.open_rasterio(
                cache_path, masked=True
            ).squeeze()
            continue

        print(f"  → MODIS {year}: downloading via Planetary Computer...")
        try:
            composite = load_modis_ndvi_year(year=year, bbox=bbox)
            composite.rio.to_raster(cache_path)
            composites[year] = composite
            print(f"    ✓ {year}: saved to {cache_path.name}")

        except Exception as exc:
            print(f"  [WARN] MODIS {year} failed: {exc} — skipping")

    return composites



# ---------------------------------------------------------------------------
# Infrastructure and Roads (Overture Maps from local parquet files)
# ---------------------------------------------------------------------------

def load_overture_data(
    bbox: list[float] = COPPERBELT_BBOX,
    layer_type: str = "segment",
    path: Path | str | None = None,
):
    """
    Load Overture data (roads (segment), infrastructure, water)
    filtered by the bounding box.

    Parameters
    ----------
    layer_type : 'segment' | 'infrastructure' | 'water'
    path       : custom path for the parquet file (optional).
                 If None, uses DATA_RAW / copperbelt_{layer_type}.parquet
    """
    try:
        if path is None:
            path = DATA_RAW / f"copperbelt_{layer_type}.parquet"

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                f"Download the Overture data for the region and save it to {DATA_RAW}/"
            )

        minx, miny, maxx, maxy = bbox
        bbox_geom = box(minx, miny, maxx, maxy)

        gdf = gpd.read_parquet(path)
        gdf = gdf[gdf.intersects(bbox_geom)].copy()

        if layer_type == "segment" and "class" in gdf.columns:
            gdf = gdf[gdf["class"].isin(["road", "motorway", "primary", "secondary"])]

        if layer_type == "water" and "class" in gdf.columns:
            gdf = gdf[gdf["class"].isin(["stream", "river"])]

        print(f"  Overture '{layer_type}' — {len(gdf):,} features loaded.")
        return gdf.reset_index(drop=True)

    except Exception as e:
        print(f"[ERROR] load_overture_data(layer_type='{layer_type}'): {e}")
        raise


# ---------------------------------------------------------------------------
# MRDS — USGS Mineral Resources Data System  (cached parquet)
# ---------------------------------------------------------------------------

def load_mrds(
    bbox: list[float] = COPPERBELT_BBOX,
    cache_path: Path | str | None = None,
    force_download: bool = False,
):
    """
    Load mineral occurrences from the USGS MRDS filtered by the bounding box.

    The global CSV (~500 MB) is downloaded only once and saved in cache
    as a filtered Parquet file for the region. In subsequent executions, it reads
    directly from the local cache.

    Parameters
    ----------
    cache_path      : path to the cache parquet file (optional).
                      Default: DATA_RAW / 'mrds_copperbelt.parquet'
    force_download  : True to ignore the cache and download again.
    """
    try:
        if cache_path is None:
            cache_path = DATA_RAW / "mrds_copperbelt.parquet"

        cache_path = Path(cache_path)

        if cache_path.exists() and not force_download:
            print(f"  MRDS — reading local cache: {cache_path.name}")
            return gpd.read_parquet(cache_path)

        print("  MRDS — downloading global CSV (~500 MB). This takes a few minutes...")
        url = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
        response = requests.get(url, timeout=300)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open("mrds.csv") as f:
                df = pd.read_csv(
                    f,
                    low_memory=False,
                    usecols=[
                        "latitude", "longitude", "site_name",
                        "com_type", "dep_type", "ore",
                        "orebody_fm", "model", "hrock_unit",
                    ],
                )

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326",
        )

        minx, miny, maxx, maxy = bbox
        gdf_filtered = gdf.cx[minx:maxx, miny:maxy].reset_index(drop=True)

        # Save filtered cache
        gdf_filtered.to_parquet(cache_path)
        print(f"  MRDS — {len(gdf_filtered):,} occurrences saved to cache: {cache_path.name}")
        return gdf_filtered

    except Exception as e:
        print(f"[ERRO] load_mrds(): {e}")
        raise


# ---------------------------------------------------------------------------
# GADM — Administrative Boundaries
# ---------------------------------------------------------------------------

def load_full_copperbelt_admin(
    bbox: list[float] = COPPERBELT_BBOX,
    level: int = 2,
):
    """
    Download and merge GADM administrative boundaries for DRC and Zambia.

    Parameters
    ----------
    level : 1 = Provinces | 2 = Districts/Municipalities
    """
    try:
        countries = ["COD", "ZMB"]
        adms = []
        minx, miny, maxx, maxy = bbox

        for iso in countries:
            print(f"  GADM — downloading {iso} level {level}...")
            url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso}_shp.zip"
            layer = f"gadm41_{iso}_{level}"

            try:
                gdf = gpd.read_file(url, layer=layer)
                gdf = gdf.cx[minx:maxx, miny:maxy]
                adms.append(gdf)
                print(f"    {iso}: {len(gdf)} units loaded.")
            except Exception as e:
                print(f"    [WARNING] Error downloading {iso}: {e}")

        if not adms:
            raise RuntimeError("No GADM data loaded.")

        return pd.concat(adms, ignore_index=True)

    except Exception as e:
        print(f"[ERROR] load_full_copperbelt_admin(): {e}")
        raise


# ---------------------------------------------------------------------------
# SRTM / NASADEM (NASA Earthdata)
# ---------------------------------------------------------------------------

def load_srtm(
    bbox: list[float] = COPPERBELT_BBOX,
    output_path: Path | str | None = None,
    earthaccess_strategy: str = "interactive",
):
    """
    Load elevation data from NASADEM/SRTM via NASA earthaccess.
    Reads from cached GeoTIFF on subsequent calls.

    Parameters
    ----------
    output_path          : path to save the final GeoTIFF (optional).
                           Default: DATA_RAW / 'srtm_copperbelt.tif'
    earthaccess_strategy : 'netrc' | 'environment' | 'interactive'
    """
    try:
        if output_path is None:
            output_path = DATA_RAW / "srtm_copperbelt.tif"

        output_path = Path(output_path)

        if output_path.exists():
            print(f"  SRTM — reading existing file: {output_path.name}")
            raster = rxr.open_rasterio(output_path)
            # Ensure it's a DataArray/Dataset, not a list (handle multi-band if needed)
            if isinstance(raster, list):
                raster = raster[0]  # Select first band if list
            return raster.squeeze()

        earthaccess.login(strategy=earthaccess_strategy)

        results = earthaccess.search_data(
            short_name="NASADEM_HGT",
            bounding_box=tuple(bbox),
        )

        if not results:
            raise ValueError("No SRTM data found.")

        temp_dir = DATA_TEMP / "srtm"
        temp_dir.mkdir(parents=True, exist_ok=True)
        downloaded = earthaccess.download(results, local_path=str(temp_dir))

        datasets = []
        for file_path in downloaded:
            p = Path(file_path)
            if p.suffix == ".zip":
                with zipfile.ZipFile(p, "r") as z:
                    content = [n for n in z.namelist() if n.endswith((".hgt", ".tif"))][0]
                    z.extract(content, path=temp_dir)
                    datasets.append(rxr.open_rasterio(temp_dir / content))
            else:
                datasets.append(rxr.open_rasterio(p))

        full_dem = xr.combine_by_coords(datasets).squeeze().drop_vars("band", errors="ignore")
        full_dem = full_dem.rio.clip_box(*bbox)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_dem.rio.to_raster(str(output_path))
        print(f"  SRTM — DEM saved to: {output_path.name}")
        return full_dem

    except Exception as e:
        print(f"[ERROR] load_srtm(): {e}")
        raise


# ---------------------------------------------------------------------------
# Geology — local ZIP files  (RECOMMENDED)
# ---------------------------------------------------------------------------

def load_geology_from_local_zips(
    bbox: list[float] = COPPERBELT_BBOX,
    zip_paths: "list[Path | str] | None" = None,
    cache_path: "Path | str | None" = None,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """
    Load geology from manually downloaded ZIP files containing shapefiles.
    These were downloaded from two sources:
    https://github.com/Heed725/Africa_Geology_Data_Shapefile?tab=readme-ov-file
    https://www.sciencebase.gov/catalog/item/60d0ff26d34e86b938aab404


    Works with ZIPs from OneGeology, CGMW, USGS, or any standard shapefile ZIP.

    The function automatically:
      - Scans DATA_RAW for any ZIP whose name contains 'geo' (if zip_paths=None)
      - Extracts and reads all .shp files found inside each ZIP
      - Reprojects to EPSG:4326 if needed
      - Clips to the bounding box
      - Merges all sources into one GeoDataFrame
      - Saves a GeoJSON cache — subsequent calls are instant

    Parameters
    ----------
    zip_paths    : explicit list of ZIP paths (optional). If None, auto-detects
                   ZIPs in data/raw/ whose name contains 'geo' or 'geol'.
    cache_path   : output GeoJSON. Default: DATA_RAW/geology_copperbelt.geojson
    force_reload : True to ignore cache and re-extract.

    Example
    -------
    # Auto-detect ZIPs (recommended):
    geo = load_geology_from_local_zips(COPPERBELT_BBOX)

    # Explicit paths:
    geo = load_geology_from_local_zips(COPPERBELT_BBOX, zip_paths=[
        "data/raw/Africa_Geology_Data_Shapefile.zip",
        "data/raw/geo7_2ag.zip",
    ])
    """
    try:
        if cache_path is None:
            cache_path = DATA_RAW / "geology_copperbelt.geojson"
        cache_path = Path(cache_path)

        if cache_path.exists() and not force_reload:
            print(f"  Geology — reading local cache: {cache_path.name}")
            return gpd.read_file(cache_path)

        # Auto-detect ZIPs if not specified
        if zip_paths is None:
            zip_paths = [
                p for p in DATA_RAW.glob("*.zip")
                if any(kw in p.name.lower() for kw in ("geo", "geol", "geologi"))
            ]
            if not zip_paths:
                raise FileNotFoundError(
                    f"No geology ZIPs found in {DATA_RAW}.\n"
                    "Place your downloaded ZIPs there, or pass zip_paths= explicitly."
                )

        zip_paths = [Path(p) for p in zip_paths]
        print(f"  Geology — {len(zip_paths)} ZIP(s): {[Path(p).name for p in zip_paths]}")

        minx, miny, maxx, maxy = bbox
        bbox_geom = box(minx, miny, maxx, maxy)
        all_gdfs = []
        extract_base = DATA_TEMP / "geology_zips"

        for zip_path in zip_paths:
            if not zip_path.exists():
                print(f"    [WARNING] Not found, skipping: {zip_path}")
                continue

            extract_dir = extract_base / zip_path.stem
            extract_dir.mkdir(parents=True, exist_ok=True)

            print(f"    Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

            shp_files = list(extract_dir.rglob("*.shp"))
            if not shp_files:
                print(f"    [WARNING] No .shp inside {zip_path.name}.")
                continue

            for shp in shp_files:
                try:
                    gdf = gpd.read_file(shp)

                    # Normalise CRS to EPSG:4326
                    if gdf.crs is None:
                        print(f"    [WARNING] {shp.name} has no CRS — assuming EPSG:4326.")
                        gdf = gdf.set_crs("EPSG:4326")
                    elif gdf.crs.to_epsg() != 4326:
                        gdf = gdf.to_crs("EPSG:4326")

                    gdf = gdf[gdf.geometry.notna()]
                    gdf = gdf[gdf.intersects(bbox_geom)].copy()

                    if len(gdf) == 0:
                        print(f"    [INFO] {shp.name}: no features within bbox.")
                        continue

                    gdf["_source_zip"] = zip_path.name
                    gdf["_source_shp"] = shp.name
                    all_gdfs.append(gdf)
                    print(f"    {shp.name}: {len(gdf):,} features.")

                except Exception as e:
                    print(f"    [WARNING] Could not read {shp.name}: {e}")

        if not all_gdfs:
            raise RuntimeError(
                "No geology features found within the bbox after processing all ZIPs.\n"
                f"Expected coverage: lon {minx}–{maxx}, lat {miny}–{maxy} (Copperbelt)."
            )

        merged = gpd.GeoDataFrame(
            pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326"
        )
        merged.to_file(cache_path, driver="GeoJSON")
        print(f"  Geology — {len(merged):,} features merged → {cache_path.name}")
        print(f"  Geology — columns: {list(merged.columns)}")
        return merged

    except Exception as e:
        print(f"[ERROR] load_geology_from_local_zips(): {e}")
        raise


# ---------------------------------------------------------------------------
# Geology — cloud WFS fallback  (tries local ZIPs first)
# ---------------------------------------------------------------------------

def download_geology_data(
    bbox: list[float] = COPPERBELT_BBOX,
    cache_path: "Path | str | None" = None,
    force_download: bool = False,
    max_features: int = 5000,
) -> gpd.GeoDataFrame:
    """
    Load geology data, preferring local ZIPs over network requests.

    Order of precedence:
      1. Cached GeoJSON (DATA_RAW/geology_copperbelt.geojson)
      2. Local ZIP files in DATA_RAW/ whose name contains 'geo'
      3. OneGeology WFS (BRGM server)
      4. USGS ScienceBase download

    If you already have ZIPs downloaded, this function will use them automatically.
    You can also call load_geology_from_local_zips() directly for more control.
    """
    try:
        if cache_path is None:
            cache_path = DATA_RAW / "geology_copperbelt.geojson"
        cache_path = Path(cache_path)

        if cache_path.exists() and not force_download:
            print(f"  Geology — reading local cache: {cache_path.name}")
            return gpd.read_file(cache_path)

        # Check for local ZIPs first
        local_zips = [
            p for p in DATA_RAW.glob("*.zip")
            if any(kw in p.name.lower() for kw in ("geo", "geol", "geologi"))
        ]
        if local_zips:
            print(f"  Geology — {len(local_zips)} local ZIP(s) found. Using local files.")
            return load_geology_from_local_zips(
                bbox=bbox, zip_paths=local_zips,
                cache_path=cache_path, force_reload=force_download,
            )

        minx, miny, maxx, maxy = bbox

        # Attempt 1: OneGeology WFS (BRGM)
        print("  Geology — trying OneGeology WFS (BRGM)...")
        try:
            resp = requests.get(
                "https://ogc.brgm.fr/geol500k/ows",
                params={
                    "service": "WFS", "version": "2.0.0", "request": "GetFeature",
                    "typeName": "ms:World_CGMW_5M_GeologicalUnitsOnshore",
                    "outputFormat": "application/json",
                    "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:4326",
                    "srsName": "EPSG:4326", "count": max_features,
                },
                timeout=120,
            )
            if resp.status_code == 200 and "json" in resp.headers.get("content-type", ""):
                gdf = gpd.read_file(io.BytesIO(resp.content))
                if len(gdf) > 0:
                    gdf = gdf[gdf.intersects(box(minx, miny, maxx, maxy))].reset_index(drop=True)
                    gdf.to_file(cache_path, driver="GeoJSON")
                    print(f"  Geology — {len(gdf):,} units saved: {cache_path.name}")
                    return gdf
        except requests.RequestException as e:
            print(f"  [WARNING] OneGeology WFS failed: {e}")

        # Attempt 2: USGS ScienceBase
        print("  Geology — trying USGS ScienceBase...")
        sb = requests.get(
            "https://www.sciencebase.gov/catalog/item/5888bf4fe4b0001f0f93c43a/files",
            params={"format": "json"}, timeout=60,
        )
        sb.raise_for_status()
        shp_url = next(
            (f["url"] for f in sb.json() if f.get("name", "").endswith(".zip")), None
        )
        if shp_url is None:
            raise RuntimeError(
                "ScienceBase ZIP not found. Download manually from:\n"
                "https://www.sciencebase.gov/catalog/item/5888bf4fe4b0001f0f93c43a\n"
                "Then call load_geology_from_local_zips()."
            )

        zip_path = DATA_TEMP / "geology_africa.zip"
        zip_path.write_bytes(requests.get(shp_url, timeout=300).content)
        return load_geology_from_local_zips(
            bbox=bbox, zip_paths=[zip_path],
            cache_path=cache_path, force_reload=True,
        )

    except Exception as e:
        print(f"[ERROR] download_geology_data(): {e}")
        raise


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== MRDS (cached) ===")
    mrds = load_mrds(COPPERBELT_BBOX)
    print(mrds[["site_name", "ore"]].head())

    print("\n=== Geology from local ZIPs ===")
    geo = load_geology_from_local_zips(COPPERBELT_BBOX)
    print(geo.columns.tolist())
    print(geo.head(3))
