import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import geopandas as gpd
from pathlib import Path
import time

from future_simulator import FutureClimateSimulator

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISRO Climate Digital Twin — Future Simulation",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium dark theme with glassmorphism
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global dark background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Month banner */
    .month-banner {
        background: linear-gradient(90deg, #238636, #1f6feb, #8957e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 6px;
        margin: 0.5rem 0 0.3rem 0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    .month-subtitle {
        text-align: center;
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }

    /* KPI cards */
    .kpi-container {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        justify-content: center;
        margin-bottom: 1rem;
    }

    .kpi-card {
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(48, 54, 61, 0.5);
        border-radius: 12px;
        padding: 0.7rem 1.2rem;
        min-width: 140px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    }

    .kpi-value {
        font-size: 1.5rem;
        font-weight: 800;
        font-family: 'Inter', monospace;
    }

    .kpi-label {
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2px;
    }

    .kpi-delta {
        font-size: 0.75rem;
        margin-top: 2px;
    }

    .delta-up { color: #f85149; }
    .delta-down { color: #3fb950; }
    .delta-neutral { color: #8b949e; }

    /* Map panel title */
    .map-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #e6edf3;
        text-align: center;
        padding: 0.4rem 0;
        margin-bottom: 0.3rem;
        font-family: 'Inter', sans-serif;
    }

    .map-source {
        font-size: 0.6rem;
        color: #484f58;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: -0.2rem;
        margin-bottom: 0.3rem;
    }

    /* Section headers */
    .section-header {
        font-size: 0.75rem;
        font-weight: 700;
        color: #58a6ff;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 0.5rem 0 0.3rem 0;
        border-bottom: 1px solid rgba(48, 54, 61, 0.4);
        margin-bottom: 0.5rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.95);
        border-right: 1px solid rgba(48, 54, 61, 0.5);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Year badge */
    .year-badge {
        background: linear-gradient(135deg, #1f6feb, #8957e5);
        color: white;
        font-size: 1.1rem;
        font-weight: 800;
        padding: 0.3rem 1.2rem;
        border-radius: 20px;
        display: inline-block;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Inter', monospace;
    }

    /* Progress bar for slideshow */
    .progress-bar-container {
        width: 100%;
        height: 4px;
        background: rgba(48, 54, 61, 0.5);
        border-radius: 2px;
        margin: 0.3rem 0 1rem 0;
        overflow: hidden;
    }

    .progress-bar-fill {
        height: 100%;
        border-radius: 2px;
        background: linear-gradient(90deg, #238636, #1f6feb, #8957e5);
        transition: width 0.3s ease;
    }

    /* Streamlit element overrides */
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #c9d1d9 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
HARMONIZED_DIR = BASE_DIR / "data" / "harmonized"
GEOJSON_PATH = BASE_DIR / "data" / "india_states.geojson"

MONTH_NAMES = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
]

MONTH_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Color schemes per variable (for heatmaps)
COLOR_RANGES = {
    "tmax": [
        [13, 8, 135], [75, 3, 161], [126, 3, 168], [171, 35, 149],
        [204, 71, 120], [229, 107, 93], [248, 149, 64], [253, 196, 42],
        [240, 249, 33]
    ],
    "tmin": [
        [49, 54, 149], [69, 117, 180], [116, 173, 209], [171, 217, 233],
        [224, 243, 248], [255, 255, 191], [254, 224, 144], [253, 174, 97],
        [244, 109, 67]
    ],
    "rain": [
        [255, 255, 204], [199, 233, 180], [127, 205, 187],
        [65, 182, 196], [29, 145, 192], [34, 94, 168], [12, 44, 132]
    ],
    "lst": [
        [0, 0, 4], [30, 10, 80], [80, 20, 140], [160, 40, 120],
        [210, 70, 80], [240, 130, 40], [250, 200, 30], [255, 255, 180]
    ],
    "imc": [
        [240, 249, 232], [204, 235, 197], [168, 221, 181],
        [123, 204, 196], [78, 179, 211], [43, 140, 190], [8, 88, 158]
    ],
    "sst": [
        [13, 8, 135], [56, 15, 160], [114, 27, 155], [166, 54, 130],
        [207, 85, 98], [237, 130, 60], [248, 183, 32], [240, 249, 33]
    ],
    "olr": [
        [255, 255, 178], [254, 217, 118], [254, 178, 76], [253, 141, 60],
        [252, 78, 42], [227, 26, 28], [177, 0, 38]
    ],
    "soil_moisture": [
        [140, 81, 10], [191, 129, 45], [223, 194, 125], [246, 232, 195],
        [199, 234, 229], [128, 205, 193], [53, 151, 143], [1, 102, 94]
    ],
    "albedo": [
        [255, 255, 255], [224, 224, 224], [192, 192, 192],
        [160, 160, 160], [128, 128, 128], [96, 96, 96], [48, 48, 48]
    ],
}

# KPI accent colors per variable
KPI_COLORS = {
    "tmax": "#ff6b6b", "tmin": "#4ecdc4", "rain": "#3498db",
    "lst": "#e67e22", "imc": "#2ecc71", "sst": "#9b59b6",
    "olr": "#f1c40f", "soil_moisture": "#1abc9c", "albedo": "#95a5a6",
}

# Page groupings for the multi-page layout
PAGE_CATEGORIES = {
    "🌡️ TEMPERATURE": ["tmax", "tmin", "lst"],
    "🌧️ PRECIPITATION & RADIATION": ["rain", "imc", "olr"],
    "🌱 SURFACE & OCEAN": ["soil_moisture", "albedo", "sst"],
}


# ─────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_simulator():
    """Load and initialize the future climate simulator."""
    sim = FutureClimateSimulator(HARMONIZED_DIR)
    n_imd, n_mos, n_nic = sim.load_all_historical()
    if n_imd > 0 or n_mos > 0 or n_nic > 0:
        sim.compute_climatology()
    return sim


@st.cache_resource
def load_india_boundary():
    """Load India boundary for clipping points."""
    if GEOJSON_PATH.exists():
        gdf = gpd.read_file(GEOJSON_PATH)
        # Simplify the geometry to avoid GEOS bad allocation errors
        gdf["geometry"] = gdf.geometry.simplify(0.05)
        polygon = gdf.geometry.unary_union
        return polygon.buffer(0.5)
    return None


@st.cache_data
def compute_future_projection(_sim, target_year, variables_tuple):
    """Compute and cache future projection for a year and specific variables."""
    return _sim.project_future_year(target_year, variables_to_project=list(variables_tuple))


# ─────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def create_map_df(data_2d, lat, lon, var_name, india_boundary=None):
    """Convert 2D array to DataFrame for PyDeck, clipped to India."""
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    df = pd.DataFrame({
        "lon": lon_grid.flatten(),
        "lat": lat_grid.flatten(),
        var_name: data_2d.flatten(),
    })
    df = df.dropna(subset=[var_name])

    if india_boundary is not None:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        df = gdf[gdf.geometry.within(india_boundary)].drop(columns=["geometry"])

    return df


def render_map(df, var_name, color_range=None):
    """Render a PyDeck heatmap for a climate variable."""
    if color_range is None:
        color_range = COLOR_RANGES.get(var_name, COLOR_RANGES["tmax"])

    heatmap = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=["lon", "lat"],
        get_weight=var_name,
        opacity=0.45,
        radiusPixels=40,
        colorRange=color_range,
    )

    tooltip_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_radius=15000,
        get_fill_color=[0, 0, 0, 0],
        pickable=True,
    )

    view = pdk.ViewState(
        longitude=82.0, latitude=22.0, zoom=3.8, pitch=0, bearing=0
    )

    layers = [heatmap, tooltip_layer]

    # Add GeoJSON boundary if available
    if GEOJSON_PATH.exists():
        geo_layer = pdk.Layer(
            "GeoJsonLayer",
            data=str(GEOJSON_PATH),
            opacity=1.0,
            stroked=True,
            filled=False,
            extruded=False,
            get_line_color=[255, 255, 255, 120],
            line_width_min_pixels=1,
        )
        layers.append(geo_layer)

    tooltip_text = f"Lat: {{lat}}\nLon: {{lon}}\n{var_name}: {{{var_name}}}"

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=pdk.map_styles.DARK,
        tooltip={"text": tooltip_text},
    )


def format_delta(current, baseline, units, var_name):
    """Format a delta value with color coding."""
    delta = current - baseline
    if abs(delta) < 0.001:
        return f'<span class="delta-neutral">+0.00 {units}</span>'

    # For some vars, increase is "bad" (warm = red), for others "good"
    warming_vars = {"tmax", "tmin", "lst", "sst", "olr"}
    if var_name in warming_vars:
        cls = "delta-up" if delta > 0 else "delta-down"
    else:
        cls = "delta-down" if delta > 0 else "delta-up"

    sign = "+" if delta > 0 else ""
    return f'<span class="{cls}">{sign}{delta:.2f} {units}</span>'


# ─────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────
def main():
    # Load simulator
    simulator = load_simulator()
    india_boundary = load_india_boundary()

    available_vars = simulator.get_available_vars()

    if not available_vars:
        st.error("No harmonized data found. Please run the data pipeline first.")
        st.code(
            "python data_downloader.py --start-year 2014 --end-year 2023\n"
            "python data_harmonizer.py --start-year 2014 --end-year 2023",
            language="bash",
        )
        st.stop()

    lat, lon = simulator.get_lat_lon()

    # ── Sidebar ──────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">🌍</div>
        <div style="font-size: 1.1rem; font-weight: 800; color: #e6edf3;
                    letter-spacing: 1px; margin-top: 0.3rem;">
            ISRO CLIMATE<br>DIGITAL TWIN
        </div>
        <div style="font-size: 0.65rem; color: #484f58; margin-top: 0.2rem;
                    text-transform: uppercase; letter-spacing: 2px;">
            Future Simulation Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Page selection
    st.sidebar.markdown(
        '<div class="section-header">Dashboard View</div>',
        unsafe_allow_html=True,
    )
    selected_page = st.sidebar.radio(
        "Select Category",
        list(PAGE_CATEGORIES.keys()),
        index=0,
    )
    selected_vars = PAGE_CATEGORIES[selected_page]

    st.sidebar.markdown("---")

    # Mode selection
    mode = st.sidebar.radio(
        "Mode",
        ["Future Simulation", "Historical Baseline"],
        index=0,
    )

    st.sidebar.markdown("---")

    if mode == "Future Simulation":
        st.sidebar.markdown(
            '<div class="section-header">Projection Settings</div>',
            unsafe_allow_html=True,
        )
        target_year = st.sidebar.slider("Target Year", 2025, 2050, 2030)

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            '<div class="section-header">Slideshow Controls</div>',
            unsafe_allow_html=True,
        )

        auto_play = st.sidebar.checkbox("Auto-Play Slideshow", value=False)
        speed = st.sidebar.slider("Speed (seconds/month)", 0.5, 3.0, 1.5, 0.5)

        # Month selection — use session_state for auto-play
        if "sim_month" not in st.session_state:
            st.session_state.sim_month = 1

        current_month = st.sidebar.slider(
            "Month", 1, 12, st.session_state.sim_month,
            format="%d",
            help="Select month (1=Jan, 12=Dec)",
        )
        st.session_state.sim_month = current_month

        st.sidebar.markdown("---")

        # Quick nav buttons
        nav_cols = st.sidebar.columns(3)
        with nav_cols[0]:
            if st.button("Prev", use_container_width=True):
                st.session_state.sim_month = max(1, current_month - 1)
                st.rerun()
        with nav_cols[1]:
            if st.button("Reset", use_container_width=True):
                st.session_state.sim_month = 1
                st.rerun()
        with nav_cols[2]:
            if st.button("Next", use_container_width=True):
                st.session_state.sim_month = min(12, current_month + 1)
                st.rerun()

        baseline_year = 2023  # unused in this mode, but avoids NameError

    else:
        st.sidebar.markdown(
            '<div class="section-header">Historical Settings</div>',
            unsafe_allow_html=True,
        )
        baseline_year = st.sidebar.slider("Year", 2014, 2023, 2023)
        current_month = st.sidebar.slider("Month", 1, 12, 6)
        target_year = None
        auto_play = False
        speed = 1.5

    # Data sources info
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="section-header">Data Sources</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(f"""
    <div style="font-size: 0.75rem; color: #8b949e; line-height: 1.6;">
        <strong style="color:#58a6ff">IMD</strong> — Rain, Tmax, Tmin<br>
        <strong style="color:#da3633">MOSDAC</strong> — LST, Rainfall, SST, OLR<br>
        <strong style="color:#3fb950">NICES</strong> — Soil Moisture, Albedo<br>
        <br>
        <em>{len(available_vars)} / 9 variables loaded</em>
    </div>
    """, unsafe_allow_html=True)

    # ── Main Content ─────────────────────────────────────────────────
    month_idx = current_month - 1  # 0-indexed

    if mode == "Future Simulation":
        _render_future_mode(
            simulator, india_boundary, selected_vars, lat, lon,
            target_year, month_idx, current_month, auto_play, speed, selected_page
        )
    else:
        _render_historical_mode(
            simulator, india_boundary, selected_vars, lat, lon,
            baseline_year, month_idx, current_month, selected_page
        )

    # ── Footer ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem; color: #30363d;
                font-size: 0.7rem; border-top: 1px solid rgba(48,54,61,0.3);
                margin-top: 2rem;">
        ISRO Hackathon &bull; Pan-India Climate Digital Twin &bull;
        IMD + MOSDAC + NICES Harmonized Data &bull;
        Physics-Informed Simulation Engine
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# FUTURE SIMULATION RENDERER
# ─────────────────────────────────────────────────────────────────────
def _render_future_mode(
    simulator, india_boundary, available_vars, lat, lon,
    target_year, month_idx, current_month, auto_play, speed, selected_page
):
    with st.spinner(f"Computing projections and rendering maps for {selected_page}..."):
        projections = compute_future_projection(simulator, target_year, tuple(available_vars))

    # Header
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 0.5rem;">
        <div class="year-badge">PROJECTED YEAR: {target_year}</div>
    </div>
    <div class="month-banner">{MONTH_NAMES[month_idx]}</div>
    <div class="month-subtitle">
        Simulated future climate pattern &bull; Based on 2014-2023 trend extrapolation
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    progress_pct = (current_month / 12) * 100
    st.markdown(f"""
    <div class="progress-bar-container">
        <div class="progress-bar-fill" style="width: {progress_pct}%;"></div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    _render_kpi_cards(simulator, available_vars, projections, month_idx, future=True)

    # 3x3 Map Grid
    _render_map_grid(simulator, india_boundary, lat, lon, available_vars,
                     projections, month_idx, future=True)

    # Auto-play logic
    if auto_play and current_month < 12:
        time.sleep(speed)
        st.session_state.sim_month = current_month + 1
        st.rerun()


# ─────────────────────────────────────────────────────────────────────
# HISTORICAL BASELINE RENDERER
# ─────────────────────────────────────────────────────────────────────
def _render_historical_mode(
    simulator, india_boundary, available_vars, lat, lon,
    baseline_year, month_idx, current_month, selected_page
):
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 0.5rem;">
        <div class="year-badge">HISTORICAL: {baseline_year}</div>
    </div>
    <div class="month-banner">{MONTH_NAMES[month_idx]}</div>
    <div class="month-subtitle">
        Historical climatology from harmonized IMD + MOSDAC + NICES data
    </div>
    """, unsafe_allow_html=True)

    progress_pct = (current_month / 12) * 100
    st.markdown(f"""
    <div class="progress-bar-container">
        <div class="progress-bar-fill" style="width: {progress_pct}%;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Build baseline dict in same format as projections
    baselines = {}
    for var in available_vars:
        b = simulator.get_baseline_monthly(var)
        if b is not None:
            baselines[var] = b

    _render_kpi_cards(simulator, available_vars, baselines, month_idx, future=False)
    _render_map_grid(simulator, india_boundary, lat, lon, available_vars,
                     baselines, month_idx, future=False)


# ─────────────────────────────────────────────────────────────────────
# SHARED RENDERERS
# ─────────────────────────────────────────────────────────────────────
def _render_kpi_cards(simulator, available_vars, data_dict, month_idx, future):
    kpi_html = '<div class="kpi-container">'
    for var in available_vars:
        meta = simulator.get_var_meta(var)
        if meta is None or var not in data_dict:
            continue

        _, display_name, units, _, _, _ = meta
        val = np.nanmean(data_dict[var][month_idx])
        color = KPI_COLORS.get(var, "#58a6ff")

        if future:
            baseline = simulator.get_baseline_monthly(var)
            baseline_val = np.nanmean(baseline[month_idx]) if baseline is not None else val
            delta_html = format_delta(val, baseline_val, units, var)
        else:
            delta_html = f'<span class="delta-neutral">{units}</span>'

        kpi_html += f"""
<div class="kpi-card">
    <div class="kpi-value" style="color: {color};">{val:.1f}</div>
    <div class="kpi-label">{display_name}</div>
    <div class="kpi-delta">{delta_html}</div>
</div>
"""
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)


def _render_map_grid(simulator, india_boundary, lat, lon, available_vars,
                     data_dict, month_idx, future):
    source_colors = {"IMD": "#58a6ff", "MOSDAC": "#da3633", "NICES": "#3fb950"}

    cols = st.columns(3)
    for col_idx, var in enumerate(available_vars):
        with cols[col_idx]:
            if var not in data_dict:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; padding:3rem;">
                    <div style="color:#484f58; font-size:0.85rem;">
                        {var} data not available
                    </div>
                </div>
                """, unsafe_allow_html=True)
                continue

            meta = simulator.get_var_meta(var)
            _, display_name, units, _, _, _ = meta
            source_label = meta[0].upper()
            src_color = source_colors.get(source_label, "#8b949e")

            st.markdown(f"""
            <div class="map-title">{display_name}
                <span style="font-size:0.65rem; color:{src_color};"> ({units})</span>
            </div>
            <div class="map-source" style="color:{src_color};">&bull; {source_label}</div>
            """, unsafe_allow_html=True)

            data_2d = data_dict[var][month_idx]
            df = create_map_df(data_2d, lat, lon, var, india_boundary)

            if len(df) > 0:
                deck = render_map(df, var)
                st.pydeck_chart(deck, use_container_width=True)

                mean_val = np.nanmean(data_2d)
                min_val = np.nanmin(data_2d)
                max_val = np.nanmax(data_2d)

                stats_parts = [
                    f"Min: {min_val:.1f}",
                    f"Mean: {mean_val:.1f}",
                    f"Max: {max_val:.1f}",
                ]

                if future:
                    baseline = simulator.get_baseline_monthly(var)
                    if baseline is not None:
                        delta = mean_val - np.nanmean(baseline[month_idx])
                        stats_parts.append(f"Delta: {delta:+.2f}")

                stats_html = " &middot; ".join(
                    f"<span>{s}</span>" for s in stats_parts
                )
                st.markdown(f"""
                <div style="display:flex; justify-content:space-around;
                            font-size:0.7rem; color:#8b949e; padding:0 0.3rem;">
                    {stats_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No data for this variable/month")


if __name__ == "__main__":
    main()
