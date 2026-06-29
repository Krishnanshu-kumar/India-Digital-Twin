"""
future_simulator.py
====================
Physics-informed future climate projection engine for the Pan-India Digital Twin.

Projects future weather patterns by:
  1. Computing 10-year trends from historical harmonized data (2014-2023)
  2. Extracting monthly climatology (mean + std per month)
  3. Extrapolating forward with trend + seasonal pattern + variability
  4. Applying advection-diffusion physics for spatial propagation

Supports ALL 9 climate variables:
  IMD:    rain, tmax, tmin
  MOSDAC: lst, imc (rainfall), sst, olr
  NICES:  soil_moisture, albedo
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter


class FutureClimateSimulator:
    """Projects future monthly climate fields from historical data."""

    # Variable metadata: name → (source, display_name, units, colormap, vmin, vmax)
    VAR_META = {
        # IMD variables
        "tmax":          ("imd", "Max Temperature", "°C", "RdYlBu_r", 15, 48),
        "tmin":          ("imd", "Min Temperature", "°C", "cool", 2, 32),
        "rain":          ("imd", "Rainfall", "mm/day", "YlGnBu", 0, 25),
        # MOSDAC variables
        "lst":           ("mosdac", "Land Surface Temp", "°C", "inferno", 12, 62),
        "imc":           ("mosdac", "Satellite Rainfall", "mm/hr", "GnBu", 0, 15),
        "sst":           ("mosdac", "Sea Surface Temp", "°C", "plasma", 22, 37),
        "olr":           ("mosdac", "Outgoing Longwave Rad.", "W/m²", "YlOrRd_r", 160, 310),
        # NICES variables
        "soil_moisture": ("nices", "Soil Moisture", "m³/m³", "BrBG", 0.03, 0.50),
        "albedo":        ("nices", "Surface Albedo", "", "Greys_r", 0.08, 0.45),
    }

    def __init__(self, harmonized_dir: str | Path):
        self.harmonized_dir = Path(harmonized_dir)
        self.dx = 27750.0  # 0.25 deg ≈ 27.75 km
        self.dy = 27750.0

        # Loaded data caches
        self._imd_data = {}
        self._mosdac_data = {}
        self._nices_data = {}

        # Computed climatology caches
        self._climatology = {}   # var → (12, nlat, nlon) monthly means
        self._clim_std = {}      # var → (12, nlat, nlon) monthly stds
        self._trends = {}        # var → (nlat, nlon) per-decade trend
        self._lat = None
        self._lon = None

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def load_all_historical(self, start_year=2014, end_year=2023):
        """Load all harmonized datasets eagerly into memory."""
        for year in range(start_year, end_year + 1):
            year_dir = self.harmonized_dir / str(year)

            imd_path = year_dir / f"imd_grid_{year}.nc"
            mosdac_path = year_dir / f"insat_L2B_{year}.nc"
            nices_path = year_dir / f"nices_ecv_{year}.nc"

            if imd_path.exists() and imd_path.stat().st_size > 500:
                try:
                    ds = xr.open_dataset(str(imd_path))
                    self._imd_data[year] = ds.load()  # eager load into memory
                    ds.close()
                except Exception:
                    pass

            if mosdac_path.exists() and mosdac_path.stat().st_size > 500:
                try:
                    ds = xr.open_dataset(str(mosdac_path))
                    self._mosdac_data[year] = ds.load()
                    ds.close()
                except Exception:
                    pass

            if nices_path.exists() and nices_path.stat().st_size > 500:
                try:
                    ds = xr.open_dataset(str(nices_path))
                    self._nices_data[year] = ds.load()
                    ds.close()
                except Exception:
                    pass

        # Extract lat/lon from IMD (primary grid)
        if self._imd_data:
            sample_ds = next(iter(self._imd_data.values()))
            self._lat = sample_ds["latitude"].values
            self._lon = sample_ds["longitude"].values

        n_imd = len(self._imd_data)
        n_mos = len(self._mosdac_data)
        n_nic = len(self._nices_data)
        return n_imd, n_mos, n_nic

    # ------------------------------------------------------------------
    # CLIMATOLOGY & TRENDS
    # ------------------------------------------------------------------
    def compute_climatology(self):
        """Compute monthly climatology (mean & std) and linear trends."""
        self._compute_imd_climatology()
        self._compute_mosdac_climatology()
        self._compute_nices_climatology()
        
        # Clear raw data caches to aggressively free up memory.
        # We only need the computed climatology and trends going forward.
        self._imd_data.clear()
        self._mosdac_data.clear()
        self._nices_data.clear()
        
        import gc
        gc.collect()

    def _compute_imd_climatology(self):
        """Monthly climatology for rain, tmax, tmin from daily IMD data."""
        for var in ["rain", "tmax", "tmin"]:
            monthly_stacks = {m: [] for m in range(1, 13)}
            annual_means = []
            years_list = []

            for year in sorted(self._imd_data.keys()):
                ds = self._imd_data[year]
                if var not in ds:
                    continue

                da = ds[var]
                times = pd.DatetimeIndex(da.coords["time"].values)
                months = times.month
                values = da.values

                for m in range(1, 13):
                    mask = months == m
                    if mask.any():
                        monthly_mean = np.nanmean(values[mask], axis=0)
                        monthly_stacks[m].append(monthly_mean)

                annual_means.append(np.nanmean(values, axis=0))
                years_list.append(year)

            # Compute climatology: mean and std across years for each month
            clim = np.zeros((12, *monthly_stacks[1][0].shape))
            clim_std = np.zeros_like(clim)
            for m in range(1, 13):
                if monthly_stacks[m]:
                    stack = np.stack(monthly_stacks[m], axis=0)
                    clim[m - 1] = np.nanmean(stack, axis=0)
                    clim_std[m - 1] = np.nanstd(stack, axis=0)

            self._climatology[var] = clim
            self._clim_std[var] = clim_std

            # Compute linear trend (per decade)
            if len(annual_means) >= 3:
                self._trends[var] = self._compute_linear_trend(
                    np.stack(annual_means, axis=0),
                    np.array(years_list)
                )

    def _compute_mosdac_climatology(self):
        """Monthly climatology for MOSDAC satellite variables."""
        # MOSDAC data is already monthly (indexed by 'month' dim)
        for var in ["lst", "imc", "sst", "olr"]:
            monthly_stacks = {m: [] for m in range(1, 13)}
            annual_means = []
            years_list = []

            for year in sorted(self._mosdac_data.keys()):
                ds = self._mosdac_data[year]
                if var not in ds:
                    continue

                da = ds[var]
                values = da.values
                if var in ["lst", "sst"]:
                    values = values - 273.15

                # MOSDAC uses 'month' coordinate (1-12)
                if "month" in da.dims:
                    month_coords = da.coords["month"].values
                    for i, m in enumerate(month_coords):
                        m = int(m)
                        if 1 <= m <= 12:
                            monthly_stacks[m].append(values[i])
                elif "time" in da.dims:
                    # Fallback: treat as time-indexed
                    times = pd.DatetimeIndex(da.coords["time"].values)
                    for i, m in enumerate(times.month):
                        monthly_stacks[m].append(values[i])

                annual_means.append(np.nanmean(values, axis=0))
                years_list.append(year)

            # Build climatology
            if not any(monthly_stacks[m] for m in range(1, 13)):
                continue

            sample = next(v[0] for v in monthly_stacks.values() if v)
            clim = np.full((12, *sample.shape), np.nan)
            clim_std = np.full_like(clim, np.nan)

            for m in range(1, 13):
                if monthly_stacks[m]:
                    stack = np.stack(monthly_stacks[m], axis=0)
                    clim[m - 1] = np.nanmean(stack, axis=0)
                    clim_std[m - 1] = np.nanstd(stack, axis=0)

            self._climatology[var] = clim
            self._clim_std[var] = clim_std

            if len(annual_means) >= 3:
                self._trends[var] = self._compute_linear_trend(
                    np.stack(annual_means, axis=0),
                    np.array(years_list)
                )

    def _compute_nices_climatology(self):
        """Monthly climatology for NICES ECV variables."""
        for var in ["soil_moisture", "albedo"]:
            monthly_stacks = {m: [] for m in range(1, 13)}
            annual_means = []
            years_list = []

            for year in sorted(self._nices_data.keys()):
                ds = self._nices_data[year]
                if var not in ds:
                    continue

                da = ds[var]
                values = da.values

                if "month" in da.dims:
                    month_coords = da.coords["month"].values
                    for i, m in enumerate(month_coords):
                        m = int(m)
                        if 1 <= m <= 12:
                            monthly_stacks[m].append(values[i])

                annual_means.append(np.nanmean(values, axis=0))
                years_list.append(year)

            if not any(monthly_stacks[m] for m in range(1, 13)):
                continue

            sample = next(v[0] for v in monthly_stacks.values() if v)
            clim = np.full((12, *sample.shape), np.nan)
            clim_std = np.full_like(clim, np.nan)

            for m in range(1, 13):
                if monthly_stacks[m]:
                    stack = np.stack(monthly_stacks[m], axis=0)
                    clim[m - 1] = np.nanmean(stack, axis=0)
                    clim_std[m - 1] = np.nanstd(stack, axis=0)

            self._climatology[var] = clim
            self._clim_std[var] = clim_std

            if len(annual_means) >= 3:
                self._trends[var] = self._compute_linear_trend(
                    np.stack(annual_means, axis=0),
                    np.array(years_list)
                )

    def _compute_linear_trend(self, annual_stack, years):
        """Compute pixel-wise linear trend (value per decade)."""
        n_years = len(years)
        if n_years < 2:
            return np.zeros(annual_stack.shape[1:])

        x = years - years.mean()
        trend = np.zeros(annual_stack.shape[1:])

        for i in range(annual_stack.shape[1]):
            for j in range(annual_stack.shape[2]):
                y = annual_stack[:, i, j]
                valid = ~np.isnan(y)
                if valid.sum() >= 2:
                    coeffs = np.polyfit(x[valid], y[valid], 1)
                    trend[i, j] = coeffs[0] * 10  # per decade
                else:
                    trend[i, j] = 0.0

        return trend

    # ------------------------------------------------------------------
    # FUTURE PROJECTION
    # ------------------------------------------------------------------
    def project_future_year(self, target_year: int, baseline_end=2023, variables_to_project=None, scenario="Moderate"):
        """
        Project monthly climate fields for a future year.

        Returns dict: var_name → (12, nlat, nlon) array of projected values.
        """
        results = {}
        years_ahead = target_year - baseline_end
        decades_ahead = years_ahead / 10.0

        for var in self.VAR_META:
            if variables_to_project is not None and var not in variables_to_project:
                continue
            if var not in self._climatology:
                continue

            clim = self._climatology[var]          # (12, nlat, nlon)
            clim_std = self._clim_std[var]          # (12, nlat, nlon)
            trend = self._trends.get(var, np.zeros(clim.shape[1:]))

            projected = np.zeros_like(clim)

            for m in range(12):
                # Base: climatological mean for this month
                base = clim[m].copy()

                # Trend contribution: linear extrapolation
                trend_signal = trend * decades_ahead

                # Apply scenario multiplier
                if scenario == "Extreme":
                    # Exponential growth for extreme scenario (RCP 8.5 simulation)
                    # For a variable that is decreasing (like soil moisture), this accelerates the decrease
                    trend_signal = np.sign(trend_signal) * (np.abs(trend_signal) * (1.0 + 0.8 * decades_ahead))

                # Seasonal modulation of trend
                # Temperature trends are stronger in summer, rainfall in monsoon
                if var in ["tmax", "tmin", "lst"]:
                    # Pre-monsoon amplification (Apr-Jun)
                    seasonal_mod = 1.0 + 0.3 * np.exp(-0.5 * ((m - 4) / 1.5) ** 2)
                elif var in ["rain", "imc"]:
                    # Monsoon intensification (Jun-Sep)
                    seasonal_mod = 1.0 + 0.5 * np.exp(-0.5 * ((m - 7) / 1.5) ** 2)
                elif var == "soil_moisture":
                    # Follows rainfall pattern
                    seasonal_mod = 1.0 + 0.3 * np.exp(-0.5 * ((m - 8) / 2.0) ** 2)
                elif var == "olr":
                    # Inverse monsoon relationship
                    seasonal_mod = 1.0 - 0.2 * np.exp(-0.5 * ((m - 7) / 2.0) ** 2)
                else:
                    seasonal_mod = 1.0

                # Apply trend with seasonal modulation
                projected_field = base + trend_signal * seasonal_mod

                # Add climate variability (scaled noise from historical std)
                np.random.seed(target_year * 100 + m)
                volatility = 0.3
                if scenario == "Extreme":
                    volatility = 1.0 + (decades_ahead * 0.5)  # Severe localized extremes
                
                noise = np.random.randn(*base.shape) * clim_std[m] * volatility
                noise = gaussian_filter(noise, sigma=2)  # spatial smoothing
                projected_field += noise

                # Apply physical constraints
                projected_field = self._apply_constraints(projected_field, var)

                # Apply advection-diffusion smoothing for spatial coherence
                projected_field = self._apply_spatial_physics(projected_field, var, m)

                projected[m] = projected_field

            results[var] = projected

        return results

    def _apply_constraints(self, field, var):
        """Apply physical constraints to keep values realistic."""
        if var == "rain":
            field = np.maximum(field, 0.0)
        elif var == "imc":
            field = np.maximum(field, 0.0)
        elif var == "soil_moisture":
            field = np.clip(field, 0.01, 0.55)
        elif var == "albedo":
            field = np.clip(field, 0.05, 0.50)
        elif var == "olr":
            field = np.clip(field, 100, 350)
        elif var in ["tmax", "tmin", "lst", "sst"]:
            field = np.clip(field, -10, 75)
        return field

    def _apply_spatial_physics(self, field, var, month):
        """
        Apply advection-diffusion smoothing to ensure spatial coherence.
        Uses simplified wind patterns for India.
        """
        # Only apply to fields that benefit from spatial smoothing
        if var in ["albedo"]:
            return field

        mask = np.isnan(field)
        if mask.all():
            return field

        field_clean = np.nan_to_num(field, nan=np.nanmean(field))

        # Monsoon months: stronger SW winds (Jun-Sep)
        is_monsoon = 5 <= month <= 8
        sigma = 1.5 if is_monsoon else 1.0

        # Gaussian diffusion as a proxy for advection-diffusion
        smoothed = gaussian_filter(field_clean, sigma=sigma)

        # Blend: mostly original with some smoothing for coherence
        blend = 0.85 * field_clean + 0.15 * smoothed

        # Re-apply NaN mask
        blend[mask] = np.nan
        return blend

    # ------------------------------------------------------------------
    # BASELINE RETRIEVAL
    # ------------------------------------------------------------------
    def get_baseline_monthly(self, var):
        """Get the historical monthly climatology for a variable."""
        return self._climatology.get(var, None)

    def get_lat_lon(self):
        """Get the coordinate arrays."""
        return self._lat, self._lon

    def get_available_vars(self):
        """Return list of variables that have computed climatology."""
        return [v for v in self.VAR_META if v in self._climatology]

    def get_var_meta(self, var):
        """Get metadata for a variable."""
        return self.VAR_META.get(var, None)

    def get_trend_summary(self):
        """Get a summary of trends for all variables."""
        summary = {}
        for var, trend in self._trends.items():
            meta = self.VAR_META[var]
            summary[var] = {
                "display_name": meta[1],
                "units": meta[2],
                "mean_trend_per_decade": float(np.nanmean(trend)),
                "max_trend_per_decade": float(np.nanmax(trend)),
                "min_trend_per_decade": float(np.nanmin(trend)),
            }
        return summary

    def close(self):
        """Close all open datasets."""
        for ds_dict in [self._imd_data, self._mosdac_data, self._nices_data]:
            for ds in ds_dict.values():
                try:
                    ds.close()
                except Exception:
                    pass
