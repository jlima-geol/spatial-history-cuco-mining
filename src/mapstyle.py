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


def add_map_furniture(
        ax,
        gdf_ref, 
        title="",
        title_fontsize: int = 14,
        label_fontsize: int = 10,
        labelticksize: int = 9,
        legend=True):
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
        ax.set_xlabel("Longitude (\u00B0)", fontsize=label_fontsize, labelpad=7) # label_fontsize=11
        ax.set_ylabel("Latitude (\u00B0)", fontsize=label_fontsize, labelpad=7)
    else:
        ax.set_xlabel("Easting (m)", fontsize=label_fontsize, labelpad=7) # label_fontsize=10
        ax.set_ylabel("Northing (m)", fontsize=label_fontsize, labelpad=7)

    ax.tick_params(axis="x", labelsize=labelticksize)
    ax.tick_params(axis="y", labelsize=labelticksize)

    # --- Title ---
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=7) # adjust pad according to the map layout (2, 4, 10)

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

def add_panel_furniture(
    ax,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    title_fontsize: int = 14,
    label_fontsize: int = 10,
    labelticksize: int = 9,
    legend: bool = False,
    legend_loc: str = "upper right",
    legend_fontsize: int = 10,
    zero_line: str | None = None,
    grid: bool = False,
    rotate_xticks: int = 0,
    rotate_yticks: int = 0,
) -> None:
    """
    Applies consistent styling to a statistical panel (bar, scatter, line).

    Removes the top and right spines, sets title and axis labels, and
    optionally adds a reference line at zero, a legend, a light grid,
    and rotated x-tick labels.

    Parameters
    ----------
    ax            : Matplotlib Axes to style.
    title         : Panel title.
    xlabel        : X-axis label (empty string = no label).
    ylabel        : Y-axis label (empty string = no label).
    title_fontsize: Font size for the title (default 11).
    label_fontsize: Font size for axis labels (default 10).
    labelticksize : Font size for axis tick labels (default 10).
    legend        : If True, draws the axes legend with auto-collected handles.
    legend_loc    : Legend location string (default "upper right").
    legend_fontsize: Font size for legend entries (default 9).
    zero_line     : Draw a dashed reference line at 0 on the given axis.
                    Pass "x" for a vertical line, "y" for a horizontal line,
                    or None (default) for no line.
    grid          : If True, adds a light horizontal grid (alpha=0.25).
    """
    # Spines
    ax.spines[["top", "right"]].set_visible(False)

    # Title and labels
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=5)
    
    ax.tick_params(axis="x", labelsize=labelticksize)
    ax.tick_params(axis="y", labelsize=labelticksize)

    # Zero reference line
    if zero_line == "y":
        ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=1)
    elif zero_line == "x":
        ax.axvline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=1)

    # Light grid
    if grid:
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.25, color="#888888")
        ax.set_axisbelow(True)

    # Legend
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc=legend_loc,
                      fontsize=legend_fontsize, frameon=True)
            

