import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import geopandas as gpd
from pathlib import Path
from simulation_engine import ClimateSimulator

st.set_page_config(page_title="ISRO Climate Digital Twin", layout="wide", page_icon="🌍")

# Define paths
BASE_DIR = Path(__file__).parent
HARMONIZED_DIR = BASE_DIR / "data" / "harmonized"

@st.cache_resource
def load_simulator():
    return ClimateSimulator(HARMONIZED_DIR)

simulator = load_simulator()

# --- Sidebar Controls ---
st.sidebar.title("🌍 Simulation Controls")
st.sidebar.markdown("Configure the initial state and apply counterfactual scenarios.")

year = st.sidebar.slider("Baseline Year", 2014, 2023, 2023)
month = st.sidebar.slider("Baseline Month", 1, 12, 6)

st.sidebar.markdown("---")
st.sidebar.subheader("Counterfactual Scenarios")

scenario = st.sidebar.selectbox("Select Scenario", [
    "None (Baseline only)",
    "Urban Heat Island (Delhi NCR)",
    "Deforestation (Western Ghats)",
    "Custom Perturbation"
])

magnitude = 0.0
lat_range = [0, 0]
lon_range = [0, 0]
var_to_perturb = "tmax"
perturb_type = "add"

if scenario == "Urban Heat Island (Delhi NCR)":
    magnitude = st.sidebar.slider("Temperature Increase (°C)", 0.0, 5.0, 2.0, 0.5)
    lat_range = [28.0, 29.0]
    lon_range = [76.5, 77.5]
    var_to_perturb = "tmax"
    perturb_type = "add"
    st.sidebar.info("Simulates localized heating due to urban expansion.")

elif scenario == "Deforestation (Western Ghats)":
    magnitude = st.sidebar.slider("Rainfall Reduction Multiplier", 0.1, 1.0, 0.7, 0.1)
    lat_range = [10.0, 16.0]
    lon_range = [74.0, 77.0]
    var_to_perturb = "rain"
    perturb_type = "multiply"
    st.sidebar.info("Simulates reduced rainfall due to forest cover loss.")
    
elif scenario == "Custom Perturbation":
    var_to_perturb = st.sidebar.selectbox("Variable", ["tmax", "rain"])
    perturb_type = st.sidebar.selectbox("Operation", ["add", "multiply"])
    magnitude = st.sidebar.number_input("Magnitude", value=1.0)
    st.sidebar.markdown("Bounding Box")
    lat_range = st.sidebar.slider("Latitude", 6.5, 40.0, (20.0, 25.0))
    lon_range = st.sidebar.slider("Longitude", 66.5, 100.0, (75.0, 80.0))

run_sim = st.sidebar.button("🚀 Run Simulation", use_container_width=True)

# --- Main Dashboard ---
st.title("AI-Powered Pan-India Climate Digital Twin")
st.markdown("Developed for the ISRO Hackathon. Interactive physics-informed simulation engine.")

# Load Baseline Data
try:
    base_tmax, base_rain, lat, lon = simulator.load_base_state(year, month)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

@st.cache_resource
def load_india_boundary():
    gdf = gpd.read_file(BASE_DIR / "data" / "india_states.geojson")
    india_polygon = gdf.geometry.unary_union
    # 0.5 degree buffer gives a healthy margin around the landmass
    return india_polygon.buffer(0.5)

india_boundary = load_india_boundary()

# Helper function to convert 2D arrays to PyDeck friendly dataframes
def create_map_df(data_array, lat_array, lon_array, var_name):
    # Flatten arrays
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    df = pd.DataFrame({
        'lon': lon_grid.flatten(),
        'lat': lat_grid.flatten(),
        var_name: data_array.flatten()
    })
    # Drop NaNs
    df = df.dropna()
    
    # Trim to India with margin
    gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    df = gdf_points[gdf_points.geometry.within(india_boundary)].drop(columns=['geometry'])
    return df

def get_color_range(var_name):
    if var_name == "tmax":
        return [
            [49, 54, 149], [69, 117, 180], [116, 173, 209], [171, 217, 233],
            [224, 243, 248], [255, 255, 191], [254, 224, 144], [253, 174, 97],
            [244, 109, 67], [215, 48, 39], [165, 0, 38]
        ]
    else: # rain
        return [
            [255, 255, 204], [199, 233, 180], [127, 205, 187],
            [65, 182, 196], [29, 145, 192], [34, 94, 168], [12, 44, 132]
        ]

def render_map(df, var_name, elevation_scale=0):
    if var_name == "tmax":
        # Plasma / Inferno smooth gradient
        color_range = [
            [13, 8, 135], [75, 3, 161], [126, 3, 168], [171, 35, 149],
            [204, 71, 120], [229, 107, 93], [248, 149, 64], [253, 196, 42],
            [240, 249, 33]
        ]
    else:
        # YlGnBu smooth gradient for rain
        color_range = [
            [255, 255, 204], [199, 233, 180], [127, 205, 187],
            [65, 182, 196], [29, 145, 192], [34, 94, 168], [12, 44, 132]
        ]

    # Smooth, continuous heatmap
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=["lon", "lat"],
        get_weight=var_name,
        opacity=0.4,       # High transparency to clearly see the map details underneath
        radiusPixels=45,   # Smooth interpolation radius
        colorRange=color_range,
    )
    
    # Invisible layer just for tooltips
    tooltip_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_radius=15000,
        get_fill_color=[0, 0, 0, 0], # Completely transparent
        pickable=True,
    )

    view_state = pdk.ViewState(
        longitude=82.0,
        latitude=21.0,
        zoom=4.0,
        pitch=0,
        bearing=0
    )

    geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        data=str(BASE_DIR / "data" / "india_states.geojson"),
        opacity=1.0,
        stroked=True,
        filled=False,
        extruded=False,
        get_line_color=[255, 255, 255, 255], 
        line_width_min_pixels=1.5,
    )

    tooltip_text = "Lat: {lat}\nLon: {lon}\n" + var_name + ": {" + var_name + "}"
    return pdk.Deck(layers=[heatmap_layer, geojson_layer, tooltip_layer], initial_view_state=view_state, map_style=pdk.map_styles.DARK, tooltip={"text": tooltip_text})


# Execute Simulation if requested
if run_sim and scenario != "None (Baseline only)":
    with st.spinner("Running Advection-Diffusion Physics Engine..."):
        # 1. Apply initial perturbation
        if var_to_perturb == "tmax":
            tmax_init = simulator.apply_perturbation(base_tmax, perturb_type, magnitude, lat_range, lon_range)
            # 2. Run simulation (simplified wind field towards East)
            u_wind = np.ones_like(base_tmax) * 2.0  # 2 m/s East
            v_wind = np.ones_like(base_tmax) * 0.5  # 0.5 m/s North
            sim_tmax = simulator.run_advection_diffusion(tmax_init, u_wind, v_wind, diff_coeff=50000, time_steps=24, dt=3600)
            
            df_base = create_map_df(base_tmax, lat, lon, "tmax")
            df_sim = create_map_df(sim_tmax, lat, lon, "tmax")
            
            st.success(f"Simulation Complete: 24-hour propagation of {scenario}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Baseline Max Temp")
                st.pydeck_chart(render_map(df_base, "tmax", elevation_scale=10000))
            with col2:
                st.subheader("Simulated Max Temp (+24 hrs)")
                st.pydeck_chart(render_map(df_sim, "tmax", elevation_scale=10000))
                
        elif var_to_perturb == "rain":
            rain_init = simulator.apply_perturbation(base_rain, perturb_type, magnitude, lat_range, lon_range)
            # Rain doesn't advect the same way, but we show the perturbed state
            df_base = create_map_df(base_rain, lat, lon, "rain")
            df_sim = create_map_df(rain_init, lat, lon, "rain")
            
            st.success(f"Scenario Applied: {scenario}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Baseline Rainfall")
                st.pydeck_chart(render_map(df_base, "rain", elevation_scale=2000))
            with col2:
                st.subheader("Counterfactual Rainfall")
                st.pydeck_chart(render_map(df_sim, "rain", elevation_scale=2000))

else:
    # Show baseline only
    st.info("Showing historical baseline. Select a scenario and run simulation to see counterfactuals.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Mean Max Temperature (°C)")
        df_tmax = create_map_df(base_tmax, lat, lon, "tmax")
        st.pydeck_chart(render_map(df_tmax, "tmax", elevation_scale=10000))
        
    with col2:
        st.subheader("Monthly Mean Rainfall (mm/day)")
        df_rain = create_map_df(base_rain, lat, lon, "rain")
        st.pydeck_chart(render_map(df_rain, "rain", elevation_scale=2000))

