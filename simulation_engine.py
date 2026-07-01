"""
simulation_engine.py
Interactive "what-if" climate scenario engine for the Pan-India Digital Twin.

Lets a user take a real historical monthly baseline field (e.g. June 2023
Tmax), apply a localized perturbation to a region (e.g. "+3°C over
Rajasthan"), and propagate that perturbation outward using a 2D
advection-diffusion model so neighbouring regions feel a physically
plausible spillover effect rather than a hard-edged box.

This is intentionally simple (constant wind field, isotropic diffusion) —
it is a scenario-exploration tool, not a numerical weather model — but the
finite-difference stepping is numerically stable (CFL-checked) so results
don't blow up for arbitrary user-chosen parameters.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger("simulation_engine")
SUPPORTED_VARS = {
    "tmax": (-10.0, 60.0),
    "tmin": (-15.0, 45.0),
    "rain": (0.0, 500.0),
}


class ClimateSimulator:
    """Loads a historical baseline field and runs perturbation + advection-diffusion scenarios on it."""

    def __init__(self, harmonized_data_path):
        """Initialize the simulator with harmonized data."""
        self.data_path = Path(harmonized_data_path)

        # Grid parameters (0.25 deg ~ 27.75 km)
        self.dx = 27750.0
        self.dy = 27750.0

        self.lat = None
        self.lon = None
        self._base_fields = {}  # var - 2D array for the currently loaded year/month

    def load_base_state(self, year=2023, month=6, variables=("tmax", "rain")):
        """

        Returns a dict {var: 2D array}, plus sets self.lat / self.lon.
        Unknown variable names are skipped with a warning rather than raising,
        so the UI can request a flexible set without crashing.
        """
        imd_path = self.data_path / str(year) / f"imd_grid_{year}.nc"
        if not imd_path.exists():
            raise FileNotFoundError(f"IMD data for {year} not found at {imd_path}.")

        ds = xr.open_dataset(imd_path)
        try:
            times = pd.DatetimeIndex(ds.time.values)
            mask = times.month == month
            if not mask.any():
                raise ValueError(f"No data found for month={month} in year={year}.")

            self.lat = ds.latitude.values
            self.lon = ds.longitude.values

            fields = {}
            for var in variables:
                if var not in SUPPORTED_VARS:
                    log.warning(f"'{var}' is not a supported simulation variable — skipping.")
                    continue
                if var not in ds:
                    log.warning(f"'{var}' not present in {imd_path.name} — skipping.")
                    continue
                fields[var] = ds[var].values[mask].mean(axis=0)

            self._base_fields = fields
            return fields, self.lat, self.lon
        finally:
            ds.close()

    def apply_perturbation(self, var_map, perturbation_type, magnitude, lat_range, lon_range):
        """Apply a perturbation to a rectangular lat/lon region of a field."""
        perturbed = var_map.copy()

        lo, hi = min(lat_range), max(lat_range)
        lat_idx = np.where((self.lat >= lo) & (self.lat <= hi))[0]
        lo, hi = min(lon_range), max(lon_range)
        lon_idx = np.where((self.lon >= lo) & (self.lon <= hi))[0]

        if len(lat_idx) == 0 or len(lon_idx) == 0:
            return perturbed
        lat_start, lat_end = lat_idx[0], lat_idx[-1]
        lon_start, lon_end = lon_idx[0], lon_idx[-1]

        region = perturbed[lat_start:lat_end + 1, lon_start:lon_end + 1]
        if perturbation_type == "add":
            region += magnitude
        elif perturbation_type == "multiply":
            region *= magnitude
        else:
            raise ValueError(f"Unknown perturbation_type: {perturbation_type!r}")

        return perturbed

   
    def run_advection_diffusion(self, T, u, v, diff_coeff, time_steps, dt=3600,
                                 cfl_safety=0.4):
        """
        T: Initial field (2D array)
        u, v: Wind velocity (scalar or 2D arrays, m/s)
        diff_coeff: Diffusion coefficient (m^2/s)
        time_steps: Number of *requested* steps worth of physical time to simulate
        dt: Requested time step in seconds

        The explicit finite-difference scheme used here is only numerically
        stable below a CFL-derived maximum timestep. Rather than letting the
        field blow up (NaNs/infs) for aggressive user-chosen parameters, this
        method automatically computes a stable sub-step and runs enough
        sub-steps to cover the same total physical time (time_steps * dt).
        """
        T_new = T.copy()
        mask = np.isnan(T)
        T_new = np.nan_to_num(T_new, nan=np.nanmean(T))

        u_max = np.max(np.abs(u)) if np.ndim(u) else abs(u)
        v_max = np.max(np.abs(v)) if np.ndim(v) else abs(v)

        diff_limit = (self.dx ** 2) / (4.0 * diff_coeff) if diff_coeff > 0 else np.inf
        adv_limit = self.dx / max(u_max + v_max, 1e-9)
        stable_dt = cfl_safety * min(diff_limit, adv_limit)

        total_time = time_steps * dt
        n_substeps = max(1, int(np.ceil(total_time / stable_dt)))
        sub_dt = total_time / n_substeps

        if n_substeps > time_steps:
            log.info(
                f"run_advection_diffusion: requested dt={dt}s was CFL-unstable "
                f"(stable dt≈{stable_dt:.1f}s) — auto-using {n_substeps} sub-steps "
                f"of {sub_dt:.1f}s to cover the same {total_time/3600:.1f}h of physical time."
            )

        for _ in range(n_substeps):
            T_pad = np.pad(T_new, 1, mode='edge')

            dTdx = (T_pad[1:-1, 2:] - T_pad[1:-1, :-2]) / (2 * self.dx)
            dTdy = (T_pad[2:, 1:-1] - T_pad[:-2, 1:-1]) / (2 * self.dy)

            d2Tdx2 = (T_pad[1:-1, 2:] - 2 * T_new + T_pad[1:-1, :-2]) / (self.dx ** 2)
            d2Tdy2 = (T_pad[2:, 1:-1] - 2 * T_new + T_pad[:-2, 1:-1]) / (self.dy ** 2)

            advection = -(u * dTdx + v * dTdy)
            diffusion = diff_coeff * (d2Tdx2 + d2Tdy2)

            T_new = T_new + (advection + diffusion) * sub_dt

        T_new[mask] = np.nan
        return T_new

    def apply_constraints(self, field, var):
        """Clip a field back to physically plausible bounds for the given variable."""
        bounds = SUPPORTED_VARS.get(var)
        if bounds is None:
            return field
        lo, hi = bounds
        clipped = np.clip(field, lo, hi)
        if var == "rain":
            clipped = np.maximum(clipped, 0.0)
        return clipped