"""
mapstyle.py
---------------------
This module provides the map style for maps used in the 
Spatial History of Copper Mining (DRC) project. It defines a 
consistent color palette and styling for all maps used in the
Environmental Impact notebbok.
"""

# %%
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import FancyArrowPatch

# ── Visual palette ──────────────────────
PALETTE = {
    "boundary":   "#2c2c2c",
    "mine":       "#270a07",
    "cluster":    "#e67e22",
    "water":      "#2980b9",
    "vegetation": "#27ae60",
    "degraded":   "#362c2a",
    "neutral":    "#bdc3c7",
    "highlight":  "#8e44ad",
}


def add_map_furniture(ax, gdf_ref, title="", legend=True):
    """Adds title, scale, north arrow, and axes labels adapted to the CRS."""
    # --- Detects if the CRS is geographic (degrees) or projected (meters) ---
    def is_geographic(gdf):
        """Returns True if the CRS is geographic (degrees)."""
        if gdf.crs is None:
            return False
        return gdf.crs.is_geographic

    geographic = is_geographic(gdf_ref)

    # --- Axis labels ---
    if geographic:
        ax.set_xlabel("Longitude (\u00B0)", fontsize=11)
        ax.set_ylabel("Latitude (\u00B0)", fontsize=11)
    else:
        ax.set_xlabel("Easting (m)", fontsize=10)
        ax.set_ylabel("Northing (m)", fontsize=10)

    # --- Title ---
    ax.set_title(title, fontsize=14, fontweight="bold", pad=6) # adjust pad according to the map layout (2, 4, 10)

    # --- Scale ---
    if geographic:
        # 1 degree latitude ≈ 111 km
        dx = 111_000  
        units = "m"
    else:
        dx = 1
        units = "m"

    scalebar = ScaleBar(
        dx=dx,
        units=units,
        location="lower left",
        box_alpha=0.8,
        scale_loc="top",
        length_fraction=0.2,
        font_properties={"size": 10},
    )
    scalebar.set_zorder(200)
    ax.add_artist(scalebar)

    # --- North arrow ---
    ax.annotate("N", xy=(0.06, 0.90), xycoords="axes fraction", # use 0.90 for maps in env_impact maps; 0.95 for other maps
                fontsize=18, ha="center", va="bottom", 
                fontweight="bold", zorder=200)

    arrow = FancyArrowPatch(
        (0.06, 0.77), (0.06, 0.90),  # use 0.77 and 0.90 for maps in env_impact maps; 0.88 and 0.95 for other maps
        transform=ax.transAxes,
        arrowstyle='-|>',
        mutation_scale=25,
        color='black',
        zorder=200
    )
    ax.add_patch(arrow)

    # --- Legend ---
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right", frameon=True, fontsize=9)
