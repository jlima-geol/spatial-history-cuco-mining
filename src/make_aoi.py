import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

def create_aoi():
    # Create are of interest (AOI) as a bounding box (minx, miny, maxx, maxy)
    minx, miny = 24.0, -13.5
    maxx, maxy = 29.0, -10.0

    geom = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame({"name": ["Copperbelt_AOI"]}, geometry=[geom], crs="EPSG:4326")

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf.to_file(out_dir / "aoi.shp")
    gdf.to_file(out_dir / "aoi.geojson", driver="GeoJSON")

    print("✔ AOI created in data/processed/aoi.shp and data/processed/aoi.geojson")

if __name__ == "__main__":
    create_aoi()

