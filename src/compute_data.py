"""
compute_data.py
---------------------
Remote and local data access pipeline for the Spatial History of Copper Mining (DRC) project.
All data is loaded directly from the cloud — no manual downloads required.

This module provides functions to compute remote sensing data (Sentinel-2, Landsat, MODIS), infrastructure (Overture Maps).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr

from typing import Dict, Optional
from rasterio.crs import CRS

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar

from scipy import ndimage
from scipy.stats import linregress
from shapely.geometry import box, shape

from obtain_remote_data import build_modis_annual_composites

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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
# Sentinel spectral indices
# ---------------------------------------------------------------------------

def compute_spectral_indices(
    bands: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    """
    Compute NDVI, NBR, NDWI from Sentinel-2 surface reflectance bands.

    All bands must share the same grid (reproject if needed before calling).
    Returns values clipped to [-1, 1].
    """
    indices: dict[str, xr.DataArray] = {}

    def _norm_diff(a: xr.DataArray, b: xr.DataArray, name: str) -> xr.DataArray:
        nd = (a - b) / (a + b)
        return nd.clip(-1.0, 1.0).rename(name)

    # NDVI — vegetated = high positive
    if "nir" in bands and "red" in bands:
        indices["ndvi"] = _norm_diff(bands["nir"], bands["red"], "ndvi")
        print(f"  ✓ NDVI: range [{float(indices['ndvi'].min()):.3f}, {float(indices['ndvi'].max()):.3f}]")

    # NBR — burned / disturbed = low negative
    if "nir" in bands and "swir1" in bands:
        indices["nbr"] = _norm_diff(bands["nir"], bands["swir1"], "nbr")
        print(f"  ✓ NBR:  range [{float(indices['nbr'].min()):.3f}, {float(indices['nbr'].max()):.3f}]")

    # NDWI (Gao) — water-stressed vegetation / flooded = positive
    if "nir" in bands and "swir1" in bands:
        indices["ndwi"] = _norm_diff(bands["nir"], bands["swir1"], "ndwi")
        # Note: NBR and NDWI share the same NIR/SWIR1 bands; NDWI uses the
        # same formula but is interpreted differently in context.
        # For open-water detection, substitute Green (B03) for NIR if available.
        print(f"  ✓ NDWI: computed (Gao 1996 formulation)")

    return indices


# ---------------------------------------------------------------------------
# Classify degradation from NDVI
# ---------------------------------------------------------------------------

NDVI_THRESHOLDS = [0.1, 0.3, 0.5]
NDVI_CLASS_LABELS = ["Water/Bare", "Sparse/Degraded", "Moderate", "Dense Woodland"]
NDVI_CLASS_COLORS = ["#1772d4", "#f2d326", "#27ae60", "#084822"]


def classify_ndvi(ndvi: xr.DataArray, thresholds: list[float]) -> xr.DataArray:
    """Assign integer class (0-3) based on NDVI thresholds."""
    t1, t2, t3 = thresholds
    classified = xr.where(ndvi < t1, 0,
                 xr.where(ndvi < t2, 1,
                 xr.where(ndvi < t3, 2, 3)))
    return classified.rename("ndvi_class")


# ---------------------------------------------------------------------------
# MODIS long-term vegetation trends (2010-2025)
# ---------------------------------------------------------------------------

def compute_ndvi_trend(
    composites: dict[int, xr.DataArray],
    min_years: int = 10,
) -> xr.DataArray | None:
    """
    Per-pixel OLS slope of NDVI ~ year over the available composite stack.

    Parameters
    ----------
    composites : {year: DataArray}  — all DataArrays must share the same grid.
    min_years  : minimum number of valid years required per pixel.

    Returns
    -------
    DataArray of OLS slope [NDVI yr⁻¹], masked where < min_years valid obs.
    """
    if len(composites) < min_years:
        print(f"  [WARN] Only {len(composites)} composites — trend unreliable.")
        return None

    years_sorted = sorted(composites.keys())

    def strip_band_coord(da):
        return da.drop_vars("band") if "band" in da.coords else da

    # Stack into a single DataArray along a 'year' dimension
    stack = xr.concat(
        [strip_band_coord(composites[y]).expand_dims(year=[y]) for y in years_sorted],
        dim="year",
    )

    print(f"  Stack shape: {stack.shape}  ({len(years_sorted)} years × H × W)")

    # Vectorised OLS via numpy (faster than applying linregress pixel-by-pixel)
    y_vals  = np.array(years_sorted, dtype=float)
    y_mean  = y_vals.mean()
    ss_x    = ((y_vals - y_mean) ** 2).sum()

    data    = stack.values                      # shape: (T, H, W)
    valid   = np.isfinite(data)                 # bool mask (T, H, W)
    n_valid = valid.sum(axis=0)                 # (H, W)

    # Masked mean per pixel
    data_filled = np.where(valid, data, 0.0)
    sum_y   = data_filled.sum(axis=0)
    n       = np.maximum(n_valid, 1)
    y_px_mean = sum_y / n                       # (H, W)

    # Covariance (year - mean_year) × (ndvi - mean_ndvi)
    cov = np.einsum(
        "t,thw->hw",
        y_vals - y_mean,
        np.where(valid, data - y_px_mean[None, :, :], 0.0),
    )
    slope_vals = cov / ss_x                     # (H, W)

    # Mask pixels with insufficient data
    slope_vals = np.where(n_valid >= min_years, slope_vals, np.nan)

    # Wrap back into DataArray using spatial coords from first composite
    ref = composites[years_sorted[0]]
    slope_da = xr.DataArray(
        slope_vals,
        coords={k: ref.coords[k] for k in ("y", "x") if k in ref.coords},
        dims=["y", "x"],
        attrs={"long_name": "NDVI trend (OLS slope)", "units": "NDVI yr⁻¹"},
    )
    slope_da.rio.write_crs(CRS_PROJ, inplace=True)

    # Summary
    finite = slope_vals[np.isfinite(slope_vals)]
    pct_neg = (finite < 0).sum() / len(finite) * 100
    print(f"  ✓ Trend map: {slope_da.shape}")
    print(f"    slope range: {finite.min():.4f} - {finite.max():.4f} NDVI yr⁻¹")
    print(f"    declining pixels: {pct_neg:.1f}%")

    return slope_da

# ---------------------------------------------------------------------------
# Identify persistent decline zones
# ---------------------------------------------------------------------------

DECLINE_THRESHOLD = -0.003  # NDVI yr⁻¹

def vectorise_decline_zones(
    trend: xr.DataArray,
    threshold: float,
    crs: str,
    min_area_ha: float = 100.0,
) -> gpd.GeoDataFrame | None:
    """
    Convert raster decline mask to polygon GeoDataFrame.
    Removes small patches below min_area_ha.
    """
    try:
        import rasterio.features

        decline_mask = (trend < threshold).values.astype(np.uint8)
        transform = trend.rio.transform()

        shapes = rasterio.features.shapes(decline_mask, transform=transform)

        polys = [
            {"geometry": shape(geom), "value": val}
            for geom, val in shapes
            if val == 1
        ]

        if not polys:
            print("  [INFO] No decline zones above threshold.")
            return None

        gdf = gpd.GeoDataFrame(polys, geometry="geometry", crs=crs)

        # Area filter
        gdf["area_ha"] = gdf.geometry.area / 10_000
        gdf = gdf[gdf["area_ha"] >= min_area_ha].reset_index(drop=True)

        print(f"  ✓ Decline zones: {len(gdf):,} polygons ≥ {min_area_ha} ha")
        print(f"    total area: {gdf['area_ha'].sum():,.0f} ha")

        return gdf

    except Exception as e:
        print(f"  [ERROR] vectorise_decline_zones: {e}")
        return None

    

# ---------------------------------------------------------------------------
# SRTM — erosion proxies & drainage patterns
# ---------------------------------------------------------------------------

# Slope

def compute_slope(dem: xr.DataArray) -> xr.DataArray:
    """
    Compute slope (degrees) using Horn's 3x3 finite-difference method.
    Cell resolution is read from the DEM transform.
    """

    # --- Validation ---
    if dem.rio.crs is None:
        raise ValueError("DEM must have a CRS before computing slope.")

    if dem.rio.resolution() is None:
        raise ValueError("DEM must have valid resolution metadata.")

    res_x, res_y = map(abs, dem.rio.resolution())
    elev = dem.values.astype("float64")

    # Replace NaNs temporarily to avoid propagation
    mask = ~np.isfinite(elev)
    elev_filled = np.where(mask, np.nanmean(elev), elev)

    # Horn kernel (classic GIS slope)
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]) / (8 * res_x)

    kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]) / (8 * res_y)

    dz_dx = ndimage.convolve(elev_filled, kernel_x, mode="nearest")
    dz_dy = ndimage.convolve(elev_filled, kernel_y, mode="nearest")

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    slope_deg[mask] = np.nan

    da = xr.DataArray(
        slope_deg,
        coords=dem.coords,
        dims=dem.dims,
        attrs={"long_name": "Slope", "units": "degrees"},
    )
    da.rio.write_crs(dem.rio.crs, inplace=True)

    print(f"  ✓ Slope computed: {np.nanmin(slope_deg):.1f}° - {np.nanmax(slope_deg):.1f}°")
    return da


# Topographic Wetness Index

def compute_twi(dem: xr.DataArray) -> xr.DataArray:
    """
    Compute a simplified Topographic Wetness Index (TWI).
    Uses Horn slope + smoothed inverse-slope proxy for upslope area.
    """

    if dem.rio.crs is None:
        raise ValueError("DEM must have a CRS.")

    res_x = abs(float(dem.rio.resolution()[0]))
    elev = dem.values.astype("float64")

    # Fill NaNs
    mask = ~np.isfinite(elev)
    elev_filled = np.where(mask, np.nanmean(elev), elev)

    # Horn slope
    kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) / (8 * res_x)
    kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) / (8 * res_x)

    dz_dx = ndimage.convolve(elev_filled, kernel_x, mode="nearest")
    dz_dy = ndimage.convolve(elev_filled, kernel_y, mode="nearest")

    slope = np.sqrt(dz_dx**2 + dz_dy**2)
    slope = np.maximum(slope, 1e-6)

    # Flow accumulation proxy
    flow_acc = ndimage.gaussian_filter(1.0 / slope, sigma=5)

    twi = np.log(flow_acc / np.tan(np.arctan(slope)))
    twi[mask] = np.nan

    da = xr.DataArray(
        twi,
        coords=dem.coords,
        dims=dem.dims,
        attrs={"long_name": "Topographic Wetness Index"},
    )
    da.rio.write_crs(dem.rio.crs, inplace=True)

    print(f"  ✓ TWI computed: {np.nanmin(twi):.2f} - {np.nanmax(twi):.2f}")
    return da


# Terrain Roughness

def compute_roughness(dem: xr.DataArray, window: int = 3) -> xr.DataArray:
    """
    Terrain roughness: local elevation standard deviation.
    Uses reflect padding to avoid border artifacts.
    """

    elev = dem.values.astype("float64")
    mask = ~np.isfinite(elev)
    elev_filled = np.where(mask, np.nanmean(elev), elev)

    mean_sq = ndimage.uniform_filter(elev_filled**2, size=window, mode="reflect")
    sq_mean = ndimage.uniform_filter(elev_filled, size=window, mode="reflect")**2

    rough = np.sqrt(np.maximum(mean_sq - sq_mean, 0))
    rough[mask] = np.nan

    da = xr.DataArray(
        rough,
        coords=dem.coords,
        dims=dem.dims,
        attrs={"long_name": "Terrain Roughness", "units": "m"},
    )
    da.rio.write_crs(dem.rio.crs, inplace=True)

    print(f"  ✓ Roughness: {np.nanmin(rough):.2f} - {np.nanmax(rough):.2f} m")
    return da


# Drainage Network Extraction

def extract_drainage_network(
    dem: xr.DataArray,
    catchment_area_km2: float = 50.0,
    target_resolution: float = 90.0,
) -> gpd.GeoDataFrame | None:
    # use pip install pysheds==0.3.5 to avoid incompatibility with numpy 2.0+ (as of June 2024)

    try:
        from rasterio.enums import Resampling
        from pysheds.grid import Grid
        from pysheds.view import Raster
        import numpy as np
        from pathlib import Path
        import tempfile

        # --- 1. Downsample ---
        orig_res = abs(float(dem.rio.resolution()[0]))
        if orig_res < target_resolution:
            dem = dem.rio.reproject(
                dem.rio.crs,
                resolution=target_resolution,
                resampling=Resampling.bilinear,
            )
            print(f"  ✓ Downsampled: {orig_res:.1f} m → {target_resolution:.1f} m")

        res_m = abs(float(dem.rio.resolution()[0]))

        # --- 2. Write temp ---
        tmp = Path(tempfile.mktemp(suffix=".tif"))
        dem.astype("float32").rio.to_raster(tmp)

        # --- 3. Load grid ---
        grid = Grid.from_raster(str(tmp))
        dem_arr = grid.read_raster(str(tmp))

        # --- 4. Hydrological conditioning ---
        dem_f64 = Raster(dem_arr.astype(np.float64), dem_arr.viewfinder)
        pit_filled = grid.fill_pits(dem_f64)
        flooded    = grid.fill_depressions(pit_filled)
        inflated   = grid.resolve_flats(flooded)

        # --- 5. Flow direction + accumulation ---
        fdir = grid.flowdir(inflated)
        acc  = grid.accumulation(fdir)

        # --- 6. Threshold independet-resolution ---
        cell_area_km2   = (res_m / 1000) ** 2
        threshold_cells = int(catchment_area_km2 / cell_area_km2)
        print(f"  ✓ Threshold: {catchment_area_km2} km² = {threshold_cells:,} cells @ {res_m:.0f} m")

        # --- 7. Extract drainage network ---
        branches = grid.extract_river_network(fdir, acc > threshold_cells)
        gdf = gpd.GeoDataFrame.from_features(branches["features"], crs=dem.rio.crs)

        tmp.unlink(missing_ok=True)

        if gdf.empty:
            print("  ✓ No streams detected.")
            return None

        print(f"  ✓ Drainage network extracted: {len(gdf):,} segments")
        return gdf

    except ImportError as e:
        print(f"  [ERROR] Dependency missing: {e}")
        return None
    except Exception as e:
        import traceback
        print(f"  [ERROR] extract_drainage_network: {e}")
        traceback.print_exc()
        return None
    

# ---------------------------------------------------------------------------
# Composite environmental impact score
# ---------------------------------------------------------------------------

IMPACT_WEIGHTS = {
    "ndvi":  0.40,
    "nbr":   0.25,
    "slope": 0.20,
    "twi":   0.15,
}


def _normalise_0_1(arr: np.ndarray) -> np.ndarray:
    """Min-max normalisation, ignoring NaN."""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def compute_impact_score(
    ndvi: Optional[xr.DataArray],
    nbr: Optional[xr.DataArray],
    slope: Optional[xr.DataArray],
    twi: Optional[xr.DataArray],
    *,
    ndvi_trend: Optional[xr.DataArray] = None,
    weights: Dict[str, float] = IMPACT_WEIGHTS,
    reference: Optional[xr.DataArray] = None,
) -> Optional[xr.DataArray]:
    """
    Composite environmental impact score [0-1].

    Parameters
    ----------
    ndvi : xr.DataArray or None
        Vegetation index (low = high impact → inverted).
    nbr : xr.DataArray or None
        Burn ratio (low = disturbed → inverted).
    slope : xr.DataArray or None
        Terrain slope (high = erosion risk → not inverted).
    twi : xr.DataArray or None
        Topographic wetness index (high = drainage concentration → not inverted).
    ndvi_trend : xr.DataArray or None
        Optional fallback vegetation proxy (negative trend = high impact).
    weights : dict
        Component weights before renormalisation.
    reference : xr.DataArray or None
        Grid to which all components are reprojected.

    Returns
    -------
    xr.DataArray or None
        Composite impact score in [0, 1], or None if no components available.
    """

    components: Dict[str, np.ndarray] = {}
    available_w: Dict[str, float] = {}

    def _prep(da: xr.DataArray, invert: bool, name: str) -> np.ndarray:
        """Reproject, normalise, optionally invert."""
        if reference is not None and da.shape != reference.shape:
            da = da.rio.reproject_match(reference)

        arr = da.values.astype(float)
        arr = _normalise_0_1(arr)
        return (1.0 - arr) if invert else arr

    # -------------------------
    # NDVI or fallback NDVI trend
    # -------------------------
    if ndvi is not None:
        components["ndvi"] = _prep(ndvi, invert=True, name="ndvi")
        available_w["ndvi"] = weights["ndvi"]

    elif ndvi_trend is not None:
        components["ndvi"] = _prep(ndvi_trend, invert=True, name="ndvi_trend")
        available_w["ndvi"] = weights["ndvi"]
        print("  [INFO] Using NDVI trend as vegetation component.")

    # -------------------------
    # NBR
    # -------------------------
    if nbr is not None:
        components["nbr"] = _prep(nbr, invert=True, name="nbr")
        available_w["nbr"] = weights["nbr"]

    # -------------------------
    # Slope
    # -------------------------
    if slope is not None:
        components["slope"] = _prep(slope, invert=False, name="slope")
        available_w["slope"] = weights["slope"]

    # -------------------------
    # TWI
    # -------------------------
    if twi is not None:
        components["twi"] = _prep(twi, invert=False, name="twi")
        available_w["twi"] = weights["twi"]

    # -------------------------
    # No components?
    # -------------------------
    if not components:
        print("  [WARN] No components available — impact score cannot be computed.")
        return None

    # -------------------------
    # Renormalise weights
    # -------------------------
    total_w = sum(available_w.values())
    norm_w = {k: v / total_w for k, v in available_w.items()}

    print(f"  Components used: {list(norm_w.keys())}")
    print(f"  Normalised weights: { {k: f'{v:.2f}' for k, v in norm_w.items()} }")

    # -------------------------
    # Compute weighted score
    # -------------------------
    ref_arr = next(iter(components.values()))
    score = np.zeros_like(ref_arr, dtype=float)
    valid_mask = np.ones_like(ref_arr, dtype=bool)

    for name, arr in components.items():
        if arr.shape != ref_arr.shape:
            print(f"  [WARN] Shape mismatch for {name} — skipping.")
            continue

        nan_mask = ~np.isfinite(arr)
        arr_filled = np.where(nan_mask, 0.0, arr)

        score += norm_w[name] * arr_filled
        valid_mask &= ~nan_mask

    score[~valid_mask] = np.nan

    # -------------------------
    # Build DataArray
    # -------------------------
    ref_da = None
    for candidate in (reference, ndvi, slope, twi):
        if candidate is not None:
            ref_da = candidate
            break

    if ref_da is None:
        raise ValueError("No valid reference DataArray found for coordinates.")

    score_da = xr.DataArray(
        score,
        coords=ref_da.coords,
        dims=ref_da.dims,
        attrs={
            "long_name": "Composite Environmental Impact Score",
            "units": "0-1",
        },
    )

    score_da.rio.write_crs(CRS_PROJ, inplace=True)

    finite = score[np.isfinite(score)]
    print(f"  ✓ Impact score: range {finite.min():.3f} - {finite.max():.3f}")

    return score_da



# ---------------------------------------------------------------------------
# Spatial join: mining features × environmental impact
# ---------------------------------------------------------------------------

def extract_zonal_stats(
    gdf: gpd.GeoDataFrame,
    raster: xr.DataArray,
    column_name: str,
    buffer_m: float = 5_000.0,
) -> gpd.GeoDataFrame:
    """
    For each feature in gdf, extract mean raster value within buffer_m.
    Adds a new column `column_name` to gdf copy.
    """
    from rasterio.mask import mask as rio_mask
    import rasterio

    result = gdf.copy()
    values = []

    # Write raster to a temp in-memory file for rasterio.mask
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    raster.rio.to_raster(tmp_path)

    try:
        with rasterio.open(tmp_path) as src:
            for geom in gdf.geometry.buffer(buffer_m):
                try:
                    out_image, _ = rio_mask(src, [geom.__geo_interface__], crop=True, nodata=np.nan)
                    valid = out_image[out_image != src.nodata]
                    valid = valid[np.isfinite(valid)]
                    values.append(float(np.mean(valid)) if len(valid) > 0 else np.nan)
                except Exception:
                    values.append(np.nan)
    finally:
        os.unlink(tmp_path)

    result[column_name] = values
    n_valid = sum(1 for v in values if np.isfinite(v))
    print(f"  ✓ {column_name}: {n_valid}/{len(gdf)} features with valid values")
    return result

