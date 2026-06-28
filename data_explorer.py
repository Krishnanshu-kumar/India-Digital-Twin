"""
data_explorer.py
================
Generate exploratory visualizations from the harmonized Pan-India climate data.

Creates publication-quality plots covering:
  1. Spatial rainfall map (monsoon season)
  2. 10-year temperature trend (annual mean)
  3. Monthly rainfall climatology (seasonal cycle)
  4. Heatwave risk map (extreme tmax days)
  5. Multi-variable correlation dashboard
  6. Year-over-year monsoon comparison

All plots saved to ./plots/
"""

import os
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
HARMONIZED_DIR = BASE_DIR / "data" / "harmonized"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("explorer")

# India approximate boundary for masking
INDIA_LAT = (6.5, 37.5)
INDIA_LON = (68.0, 97.5)

# Styling
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 11,
})


def load_all_years(start=2014, end=2023):
    """Load all harmonized datasets into dictionaries keyed by year."""
    imd_data = {}
    mosdac_data = {}
    nices_data = {}

    for year in range(start, end + 1):
        year_dir = HARMONIZED_DIR / str(year)
        imd_path = year_dir / f"imd_grid_{year}.nc"
        mosdac_path = year_dir / f"insat_L2B_{year}.nc"
        nices_path = year_dir / f"nices_ecv_{year}.nc"

        if imd_path.exists():
            imd_data[year] = xr.open_dataset(str(imd_path))
        if mosdac_path.exists():
            mosdac_data[year] = xr.open_dataset(str(mosdac_path))
        if nices_path.exists():
            nices_data[year] = xr.open_dataset(str(nices_path))

    log.info(f"Loaded {len(imd_data)} years of IMD, {len(mosdac_data)} MOSDAC, {len(nices_data)} NICES")
    return imd_data, mosdac_data, nices_data


# ===================================================================
# PLOT 1: Monsoon Rainfall Spatial Map
# ===================================================================
def plot_monsoon_rainfall(imd_data):
    """Mean monsoon (Jun-Sep) rainfall averaged over all years."""
    log.info("Plotting monsoon rainfall spatial map ...")

    all_monsoon = []
    for year, ds in imd_data.items():
        rain = ds["rain"]
        # Filter Jun-Sep (months 6-9)
        times = rain.coords["time"].values
        months = np.array([np.datetime64(t, "M").astype(int) % 12 + 1
                           if hasattr(t, 'astype') else int(str(t)[5:7])
                           for t in times])
        # Use pandas for reliable month extraction
        import pandas as pd
        months = pd.DatetimeIndex(times).month
        monsoon_mask = (months >= 6) & (months <= 9)
        monsoon_rain = rain.values[monsoon_mask]
        # Sum over monsoon season (total mm)
        seasonal_total = np.nansum(monsoon_rain, axis=0)
        all_monsoon.append(seasonal_total)

    mean_monsoon = np.nanmean(np.stack(all_monsoon), axis=0)

    fig, ax = plt.subplots(figsize=(10, 12))

    lat = imd_data[2023]["rain"].coords["latitude"].values
    lon = imd_data[2023]["rain"].coords["longitude"].values

    # Custom colormap: dry → wet
    colors_rain = ["#2c1810", "#8B4513", "#CD853F", "#FFD700",
                   "#90EE90", "#32CD32", "#006400", "#00CED1",
                   "#1E90FF", "#0000CD", "#4B0082"]
    cmap_rain = mcolors.LinearSegmentedColormap.from_list("monsoon", colors_rain, N=256)

    im = ax.pcolormesh(lon, lat, mean_monsoon, cmap=cmap_rain,
                       vmin=0, vmax=2500, shading="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Total Rainfall (mm)", fontsize=13, color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#8b949e")

    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title("Mean Monsoon Rainfall (Jun–Sep)\n2014–2023 Average",
                 fontsize=16, fontweight="bold", color="#58a6ff", pad=15)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = PLOTS_DIR / "01_monsoon_rainfall_spatial.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  → Saved {path.name}")


# ===================================================================
# PLOT 2: 10-Year Temperature Trend
# ===================================================================
def plot_temperature_trend(imd_data):
    """Annual mean tmax and tmin across India over 10 years."""
    log.info("Plotting 10-year temperature trend ...")

    years = sorted(imd_data.keys())
    annual_tmax = []
    annual_tmin = []
    annual_tmax_std = []
    annual_tmin_std = []

    for year in years:
        ds = imd_data[year]
        tmax_vals = ds["tmax"].values
        tmin_vals = ds["tmin"].values
        annual_tmax.append(np.nanmean(tmax_vals))
        annual_tmin.append(np.nanmean(tmin_vals))
        annual_tmax_std.append(np.nanstd(np.nanmean(tmax_vals, axis=(1, 2))))
        annual_tmin_std.append(np.nanstd(np.nanmean(tmin_vals, axis=(1, 2))))

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(years,
                     np.array(annual_tmax) - np.array(annual_tmax_std),
                     np.array(annual_tmax) + np.array(annual_tmax_std),
                     alpha=0.2, color="#ff6b6b")
    ax.fill_between(years,
                     np.array(annual_tmin) - np.array(annual_tmin_std),
                     np.array(annual_tmin) + np.array(annual_tmin_std),
                     alpha=0.2, color="#4ecdc4")

    ax.plot(years, annual_tmax, "o-", color="#ff6b6b", linewidth=2.5,
            markersize=8, label=f"Max Temp (μ={np.mean(annual_tmax):.1f}°C)", zorder=5)
    ax.plot(years, annual_tmin, "s-", color="#4ecdc4", linewidth=2.5,
            markersize=8, label=f"Min Temp (μ={np.mean(annual_tmin):.1f}°C)", zorder=5)

    # Trend lines
    z_max = np.polyfit(years, annual_tmax, 1)
    z_min = np.polyfit(years, annual_tmin, 1)
    ax.plot(years, np.polyval(z_max, years), "--", color="#ff6b6b", alpha=0.6, linewidth=1.5)
    ax.plot(years, np.polyval(z_min, years), "--", color="#4ecdc4", alpha=0.6, linewidth=1.5)

    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Temperature (°C)", fontsize=13)
    ax.set_title("Pan-India Annual Mean Temperature Trend\n2014–2023 (IMD Gridded Data)",
                 fontsize=15, fontweight="bold", color="#58a6ff", pad=15)
    ax.legend(fontsize=11, facecolor="#161b22", edgecolor="#30363d", loc="center left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years)

    # Annotate trend
    trend_max = z_max[0] * 10  # per decade
    trend_min = z_min[0] * 10
    ax.text(0.98, 0.05, f"Tmax trend: {trend_max:+.2f}°C/decade\nTmin trend: {trend_min:+.2f}°C/decade",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", edgecolor="#30363d", alpha=0.9),
            color="#e6edf3")

    fig.tight_layout()
    path = PLOTS_DIR / "02_temperature_trend_10yr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  → Saved {path.name}")


# ===================================================================
# PLOT 3: Monthly Rainfall Climatology (Seasonal Cycle)
# ===================================================================
def plot_seasonal_rainfall(imd_data):
    """Monthly mean rainfall climatology showing the monsoon cycle."""
    log.info("Plotting seasonal rainfall climatology ...")
    import pandas as pd

    monthly_means = {m: [] for m in range(1, 13)}

    for year, ds in imd_data.items():
        rain = ds["rain"]
        times = rain.coords["time"].values
        months = pd.DatetimeIndex(times).month

        for m in range(1, 13):
            mask = months == m
            if mask.any():
                monthly_means[m].append(np.nanmean(rain.values[mask]))

    months_list = list(range(1, 13))
    means = [np.mean(monthly_means[m]) for m in months_list]
    stds = [np.std(monthly_means[m]) for m in months_list]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Gradient bars
    colors = ["#2c3e50", "#2c3e50", "#e67e22", "#e67e22", "#e74c3c",
              "#3498db", "#2980b9", "#2980b9", "#3498db", "#e67e22",
              "#2c3e50", "#2c3e50"]

    bars = ax.bar(month_names, means, color=colors, edgecolor="#30363d",
                  linewidth=0.8, alpha=0.85, width=0.7)
    ax.errorbar(month_names, means, yerr=stds, fmt="none", ecolor="#8b949e",
                capsize=4, capthick=1.5, linewidth=1.5)

    # Highlight monsoon
    ax.axvspan(4.5, 8.5, alpha=0.08, color="#3498db", label="Monsoon (Jun–Sep)")

    ax.set_xlabel("Month", fontsize=13)
    ax.set_ylabel("Mean Daily Rainfall (mm/day)", fontsize=13)
    ax.set_title("Pan-India Monthly Rainfall Climatology\n2014–2023 (IMD 0.25° Grid)",
                 fontsize=15, fontweight="bold", color="#58a6ff", pad=15)
    ax.legend(fontsize=11, facecolor="#161b22", edgecolor="#30363d")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / "03_seasonal_rainfall_cycle.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  → Saved {path.name}")


# ===================================================================
# PLOT 4: Extreme Heat Risk Map
# ===================================================================
def plot_heatwave_risk(imd_data):
    """Spatial map of days where tmax > 42°C (extreme heat threshold)."""
    log.info("Plotting heatwave risk map ...")

    all_extreme_days = []
    for year, ds in imd_data.items():
        tmax = ds["tmax"].values
        extreme = np.nansum(tmax > 42.0, axis=0)
        all_extreme_days.append(extreme)

    mean_extreme = np.mean(np.stack(all_extreme_days), axis=0)

    lat = imd_data[2023]["tmax"].coords["latitude"].values
    lon = imd_data[2023]["tmax"].coords["longitude"].values

    fig, ax = plt.subplots(figsize=(10, 12))

    cmap_heat = mcolors.LinearSegmentedColormap.from_list(
        "heat", ["#0d1117", "#1a1a2e", "#16213e", "#e94560",
                 "#ff6b6b", "#ffd93d", "#ffffff"], N=256
    )

    im = ax.pcolormesh(lon, lat, mean_extreme, cmap=cmap_heat,
                       vmin=0, vmax=30, shading="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Avg Days/Year with Tmax > 42°C", fontsize=13, color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#8b949e")

    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title("Extreme Heat Risk Map (Tmax > 42°C)\n2014–2023 Annual Average",
                 fontsize=16, fontweight="bold", color="#ff6b6b", pad=15)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = PLOTS_DIR / "04_heatwave_risk_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  → Saved {path.name}")


# ===================================================================
# PLOT 5: Multi-Variable Dashboard
# ===================================================================
def plot_dashboard(imd_data, mosdac_data, nices_data):
    """6-panel dashboard showing all key variables for a single year (2023)."""
    log.info("Plotting multi-variable dashboard ...")

    year = 2023
    ds_imd = imd_data[year]
    ds_mosdac = mosdac_data[year]
    ds_nices = nices_data[year]

    lat_imd = ds_imd["rain"].coords["latitude"].values
    lon_imd = ds_imd["rain"].coords["longitude"].values
    lat_m = ds_mosdac["lst"].coords["latitude"].values
    lon_m = ds_mosdac["lst"].coords["longitude"].values
    lat_n = ds_nices["soil_moisture"].coords["latitude"].values
    lon_n = ds_nices["soil_moisture"].coords["longitude"].values

    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    panels = [
        (gs[0, 0], np.nanmean(ds_imd["rain"].values, axis=0), lat_imd, lon_imd,
         "Mean Daily Rainfall (mm)", "YlGnBu", 0, 15),
        (gs[0, 1], np.nanmean(ds_imd["tmax"].values, axis=0), lat_imd, lon_imd,
         "Mean Max Temperature (°C)", "RdYlBu_r", 20, 42),
        (gs[1, 0], np.nanmean(ds_imd["tmin"].values, axis=0), lat_imd, lon_imd,
         "Mean Min Temperature (°C)", "cool", 5, 28),
        (gs[1, 1], np.nanmean(ds_mosdac["lst"].values, axis=0), lat_m, lon_m,
         "Land Surface Temp (K) — MOSDAC", "inferno", 295, 330),
        (gs[2, 0], np.nanmean(ds_nices["soil_moisture"].values, axis=0), lat_n, lon_n,
         "Soil Moisture (m³/m³) — NICES", "BrBG", 0.05, 0.45),
        (gs[2, 1], np.nanmean(ds_nices["albedo"].values, axis=0), lat_n, lon_n,
         "Surface Albedo — NICES", "Greys_r", 0.1, 0.35),
    ]

    for gs_pos, data, lat, lon, title, cmap, vmin, vmax in panels:
        ax = fig.add_subplot(gs_pos)
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#8b949e")
        ax.set_title(title, fontsize=13, fontweight="bold", color="#58a6ff", pad=8)
        ax.set_aspect("equal")
        ax.set_xlabel("Lon (°E)", fontsize=9)
        ax.set_ylabel("Lat (°N)", fontsize=9)

    fig.suptitle("Pan-India Climate Dashboard — 2023\nIMD + MOSDAC + NICES Harmonized Data",
                 fontsize=18, fontweight="bold", color="#e6edf3", y=0.98)

    path = PLOTS_DIR / "05_multi_variable_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  → Saved {path.name}")


# ===================================================================
# PLOT 6: Monsoon Year-over-Year Comparison
# ===================================================================
def plot_monsoon_comparison(imd_data):
    """Heatmap of total monsoon rainfall by year."""
    log.info("Plotting monsoon year-over-year comparison ...")
    import pandas as pd

    years = sorted(imd_data.keys())
    monsoon_totals = []

    for year in years:
        ds = imd_data[year]
        rain = ds["rain"]
        times = rain.coords["time"].values
        months = pd.DatetimeIndex(times).month
        monsoon_mask = (months >= 6) & (months <= 9)
        total = np.nanmean(np.nansum(rain.values[monsoon_mask], axis=0))
        monsoon_totals.append(total)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Color based on relative wetness
    mean_total = np.mean(monsoon_totals)
    colors = []
    for t in monsoon_totals:
        ratio = (t - mean_total) / mean_total
        if ratio > 0.05:
            colors.append("#3498db")  # wet
        elif ratio < -0.05:
            colors.append("#e74c3c")  # dry
        else:
            colors.append("#95a5a6")  # normal

    bars = ax.bar([str(y) for y in years], monsoon_totals, color=colors,
                  edgecolor="#30363d", linewidth=0.8, width=0.65)

    ax.axhline(y=mean_total, color="#f39c12", linestyle="--", linewidth=2,
               label=f"10-yr Mean: {mean_total:.0f} mm", alpha=0.8)

    # Annotate each bar
    for bar, total in zip(bars, monsoon_totals):
        pct = ((total - mean_total) / mean_total) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{pct:+.1f}%", ha="center", va="bottom", fontsize=9,
                color="#e6edf3", fontweight="bold")

    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Total Monsoon Rainfall (mm)", fontsize=13)
    ax.set_title("Pan-India Monsoon (JJAS) Rainfall — Year-over-Year\n🔵 Wet  ⚫ Normal  🔴 Dry",
                 fontsize=15, fontweight="bold", color="#58a6ff", pad=15)
    ax.legend(fontsize=11, facecolor="#161b22", edgecolor="#30363d")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / "06_monsoon_yoy_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  → Saved {path.name}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  Pan-India Climate Digital Twin — Data Explorer     ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("")

    imd_data, mosdac_data, nices_data = load_all_years()

    plot_monsoon_rainfall(imd_data)
    plot_temperature_trend(imd_data)
    plot_seasonal_rainfall(imd_data)
    plot_heatwave_risk(imd_data)
    plot_dashboard(imd_data, mosdac_data, nices_data)
    plot_monsoon_comparison(imd_data)

    log.info("")
    log.info("=" * 60)
    log.info(f"All plots saved to: {PLOTS_DIR.resolve()}")
    log.info("=" * 60)

    # Close all datasets
    for ds_dict in [imd_data, mosdac_data, nices_data]:
        for ds in ds_dict.values():
            ds.close()


if __name__ == "__main__":
    main()
