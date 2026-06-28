import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

class ClimateSimulator:
    def __init__(self, harmonized_data_path):
        """Initialize the simulator with harmonized data."""
        self.data_path = harmonized_data_path
        self.ds_imd = None
        self.ds_mosdac = None
        
        # Grid parameters (0.25 deg ~ 27.75 km)
        self.dx = 27750.0 
        self.dy = 27750.0
        
    def load_base_state(self, year=2023, month=6):
        """Load a monthly mean state to act as our simulation baseline."""
        import pandas as pd
        imd_path = self.data_path / str(year) / f"imd_grid_{year}.nc"
        mosdac_path = self.data_path / str(year) / f"insat_L2B_{year}.nc"
        
        if imd_path.exists():
            ds = xr.open_dataset(imd_path)
            # Use pandas DatetimeIndex to get the month
            times = pd.DatetimeIndex(ds.time.values)
            mask = times.month == month
            self.base_tmax = ds['tmax'].values[mask].mean(axis=0)
            self.base_rain = ds['rain'].values[mask].mean(axis=0)
            self.lat = ds.latitude.values
            self.lon = ds.longitude.values
        else:
            raise FileNotFoundError(f"IMD data for {year} not found.")
            
        return self.base_tmax, self.base_rain, self.lat, self.lon

    def apply_perturbation(self, var_map, perturbation_type, magnitude, lat_range, lon_range):
        """Apply a perturbation to a specific region."""
        perturbed = var_map.copy()
        
        # Find indices for the bounding box
        lat_idx = np.where((self.lat >= lat_range[0]) & (self.lat <= lat_range[1]))[0]
        lon_idx = np.where((self.lon >= lon_range[0]) & (self.lon <= lon_range[1]))[0]
        
        if len(lat_idx) == 0 or len(lon_idx) == 0:
            return perturbed
            
        lat_start, lat_end = lat_idx[0], lat_idx[-1]
        lon_start, lon_end = lon_idx[0], lon_idx[-1]
        
        # Need to handle reverse sorted lats if applicable
        if lat_start > lat_end:
            lat_start, lat_end = lat_end, lat_start
            
        if perturbation_type == "add":
            perturbed[lat_start:lat_end+1, lon_start:lon_end+1] += magnitude
        elif perturbation_type == "multiply":
            perturbed[lat_start:lat_end+1, lon_start:lon_end+1] *= magnitude
            
        return perturbed

    def run_advection_diffusion(self, T, u, v, diff_coeff, time_steps, dt=3600):
        """
        Run a 2D advection-diffusion simulation on the temperature field.
        T: Initial temperature field (2D array)
        u, v: Wind velocity fields (2D arrays, m/s)
        diff_coeff: Diffusion coefficient (m^2/s)
        time_steps: Number of steps to simulate
        dt: Time step in seconds
        """
        T_new = T.copy()
        
        # Handle NaNs (e.g., ocean areas in IMD data)
        mask = np.isnan(T)
        T_new = np.nan_to_num(T_new, nan=np.nanmean(T))

        for step in range(time_steps):
            # Spatial derivatives (Central difference)
            # Pad boundaries to handle edges
            T_pad = np.pad(T_new, 1, mode='edge')
            
            dTdx = (T_pad[1:-1, 2:] - T_pad[1:-1, :-2]) / (2 * self.dx)
            dTdy = (T_pad[2:, 1:-1] - T_pad[:-2, 1:-1]) / (2 * self.dy)
            
            d2Tdx2 = (T_pad[1:-1, 2:] - 2*T_new + T_pad[1:-1, :-2]) / (self.dx**2)
            d2Tdy2 = (T_pad[2:, 1:-1] - 2*T_new + T_pad[:-2, 1:-1]) / (self.dy**2)
            
            # Advection term: - (u * dT/dx + v * dT/dy)
            advection = -(u * dTdx + v * dTdy)
            
            # Diffusion term: D * (d2T/dx2 + d2T/dy2)
            diffusion = diff_coeff * (d2Tdx2 + d2Tdy2)
            
            # Update T
            T_new += (advection + diffusion) * dt
            
        # Re-apply NaN mask
        T_new[mask] = np.nan
        return T_new
