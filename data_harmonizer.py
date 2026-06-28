"""
data_harmonizer.py
==================
Transforms raw downloaded data (IMD .nc/.grd, MOSDAC .h5, NICES .nc) into
unified, analysis-ready NetCDF files on a common 0.25° EPSG:4326 grid.

Outputs are stored year-wise: ./data/harmonized/{year}/

Pipeline steps:
  1. Read raw files from ./data/raw/{source}/{year}/
  2. Clean missing values (-999.0, 99.9 → NaN)
  3. Regrid all datasets to a common 0.25° lat/lon grid
  4. Merge into consolidated xarray Datasets per year
  5. Write compressed NetCDF4 files to ./data/harmonized/{year}/
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
HARMONIZED_DIR = BASE_DIR / "data" / "harmonized"

# Target grid: 0.25° resolution over India
TARGET_LAT = np.arange(6.5, 40.25, 0.25)
TARGET_LON = np.arange(66.5, 100.25, 0.25)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("harmonizer")


# ===================================================================
# 1. IMD HARMONIZER (YEAR-WISE)
# ===================================================================
def harmonize_imd(year: int) -> xr.Dataset | None:
    """
    Parse raw IMD data for a single year and produce a unified NetCDF.

    Looks for data in: data/raw/imd/{year}/
    Supports imdlib NetCDF snapshots and raw .grd binary files.
    """
    log.info(f"  ── IMD ──")

    imd_year_dir = RAW_DIR / "imd" / str(year)
    if not imd_year_dir.exists():
        log.warning(f"    Directory not found: {imd_year_dir}")
        return None

    datasets = {}
    specs = {
        "rain": {
            "nlon": 129, "nlat": 135,
            "lon": np.linspace(66.5, 100.0, 129),
            "lat": np.linspace(6.5, 40.0, 135),
            "missing": -999.0,
            "units": "mm/day",
            "long_name": "Daily Rainfall",
        },
        "tmax": {
            "nlon": 31, "nlat": 31,
            "lon": np.linspace(67.5, 97.5, 31),
            "lat": np.linspace(7.5, 37.5, 31),
            "missing": 99.9,
            "units": "°C",
            "long_name": "Daily Maximum Temperature",
        },
        "tmin": {
            "nlon": 31, "nlat": 31,
            "lon": np.linspace(67.5, 97.5, 31),
            "lat": np.linspace(7.5, 37.5, 31),
            "missing": 99.9,
            "units": "°C",
            "long_name": "Daily Minimum Temperature",
        },
    }

    for var, spec in specs.items():
        # Strategy 1: NetCDF snapshot from imdlib
        nc_path = imd_year_dir / f"imd_{var}_{year}.nc"
        if nc_path.exists():
            ds_imd = xr.open_dataset(str(nc_path))

            # Normalize dimension names
            rename_map = {}
            for dim in ds_imd.dims:
                dl = dim.lower()
                if "lat" in dl and dim != "latitude":
                    rename_map[dim] = "latitude"
                elif "lon" in dl and dim != "longitude":
                    rename_map[dim] = "longitude"
            if rename_map:
                ds_imd = ds_imd.rename(rename_map)

            data_vars = list(ds_imd.data_vars)
            if not data_vars:
                continue
            da = ds_imd[data_vars[0]]

            # Mask missing values
            da = da.where(~np.isclose(da, spec["missing"], atol=0.1))
            if var == "rain":
                da = da.where(da >= 0)

            da.attrs.update({
                "units": spec["units"],
                "long_name": spec["long_name"],
                "source": "IMD via imdlib",
            })

            # Regrid if needed
            src_lat = da.coords["latitude"].values
            src_lon = da.coords["longitude"].values
            if len(src_lon) != len(TARGET_LON) or len(src_lat) != len(TARGET_LAT):
                da = _regrid_to_target(da, src_lat, src_lon)

            datasets[var] = da
            log.info(f"    {var}: {da.shape} [{float(np.nanmin(da.values)):.1f} – {float(np.nanmax(da.values)):.1f}]")
            continue

        # Strategy 2: Raw .grd binary
        grd_path = imd_year_dir / var / f"{year}.grd"
        if not grd_path.exists():
            grd_path = imd_year_dir / f"{var}_{year}.grd"
        if not grd_path.exists():
            log.warning(f"    {var}: no data found")
            continue

        nlon, nlat = spec["nlon"], spec["nlat"]
        record_bytes = nlon * nlat * 4
        n_days = grd_path.stat().st_size // record_bytes

        raw = np.fromfile(str(grd_path), dtype="<f4")
        raw = raw.reshape(n_days, nlat, nlon, order="F").astype(np.float64)
        raw[np.isclose(raw, spec["missing"], atol=0.1)] = np.nan
        if var == "rain":
            raw[raw < 0] = np.nan

        time_coords = [np.datetime64(f"{year}-01-01") + np.timedelta64(i, "D")
                       for i in range(n_days)]
        da = xr.DataArray(
            data=raw, dims=["time", "latitude", "longitude"],
            coords={"time": time_coords, "latitude": spec["lat"], "longitude": spec["lon"]},
            attrs={"units": spec["units"], "long_name": spec["long_name"], "source": "IMD"},
        )

        if nlon != len(TARGET_LON) or nlat != len(TARGET_LAT):
            da = _regrid_to_target(da, spec["lat"], spec["lon"])
        datasets[var] = da
        log.info(f"    {var}: {da.shape}")

    if not datasets:
        return None

    ds = xr.Dataset(
        {name: da for name, da in datasets.items()},
        attrs={
            "title": f"IMD Harmonized Daily Gridded Data — {year}",
            "institution": "India Meteorological Department",
            "year": year,
            "conventions": "CF-1.8",
            "spatial_resolution": "0.25 degrees",
            "crs": "EPSG:4326",
            "created": datetime.now(tz=timezone.utc).isoformat(),
        },
    )
    return ds


# ===================================================================
# 2. MOSDAC HARMONIZER (YEAR-WISE)
# ===================================================================
def harmonize_mosdac(year: int) -> xr.Dataset | None:
    """Parse MOSDAC HDF5 files for a single year."""
    log.info(f"  ── MOSDAC ──")

    import h5py

    mosdac_year_dir = RAW_DIR / "mosdac" / str(year)
    if not mosdac_year_dir.exists():
        log.warning(f"    Directory not found: {mosdac_year_dir}")
        return None

    h5_files = sorted(mosdac_year_dir.glob("*.h5"))
    if not h5_files:
        log.warning(f"    No HDF5 files found for {year}")
        return None

    # Group by product
    products: dict[str, list] = {}
    for fp in h5_files:
        parts = fp.stem.split("_")
        if len(parts) >= 3:
            pid = "_".join(parts[:3])
        else:
            pid = parts[0]
        products.setdefault(pid, []).append(fp)

    all_vars = {}

    for pid, files in products.items():
        monthly_arrays = []
        months = []

        for fp in sorted(files):
            try:
                with h5py.File(str(fp), "r") as hf:
                    lat = hf["Latitude"][:]
                    lon = hf["Longitude"][:]

                    skip = {"Latitude", "Longitude", "lat", "lon"}
                    data_keys = [k for k in hf.keys() if k not in skip]
                    if not data_keys:
                        continue

                    var_key = data_keys[0]
                    data = hf[var_key][:].astype(np.float64)
                    units = hf[var_key].attrs.get("units", b"unknown")
                    if isinstance(units, bytes):
                        units = units.decode()

                    missing = hf[var_key].attrs.get("missing_value", -999.0)
                    data[np.isclose(data, float(missing), atol=0.5)] = np.nan

                    # Extract month from attributes or filename
                    month_val = int(hf.attrs.get("Month", 0))
                    if not month_val:
                        date_str = hf.attrs.get("Date", b"").decode() if isinstance(
                            hf.attrs.get("Date", b""), bytes) else str(hf.attrs.get("Date", ""))
                        if len(date_str) >= 6:
                            month_val = int(date_str[4:6])

                    regridded = _regrid_2d(data, lat, lon, TARGET_LAT, TARGET_LON)
                    monthly_arrays.append(regridded)
                    months.append(month_val if month_val else len(months) + 1)

            except Exception as e:
                log.warning(f"    Error reading {fp.name}: {e}")

        if monthly_arrays:
            stacked = np.stack(monthly_arrays, axis=0)
            clean_name = pid.lower().replace("3rimg_l2b_", "")

            da = xr.DataArray(
                data=stacked,
                dims=["month", "latitude", "longitude"],
                coords={"month": months, "latitude": TARGET_LAT, "longitude": TARGET_LON},
                attrs={"units": units, "long_name": pid, "source": "MOSDAC INSAT-3D/3DR"},
            )
            all_vars[clean_name] = da
            log.info(f"    {clean_name}: {da.shape}")

    if not all_vars:
        return None

    ds = xr.Dataset(
        all_vars,
        attrs={
            "title": f"MOSDAC INSAT-3D/3DR Harmonized L2B — {year}",
            "institution": "Space Applications Centre, ISRO",
            "year": year,
            "satellite": "INSAT-3DR",
            "temporal_resolution": "Monthly Composite",
            "conventions": "CF-1.8",
            "spatial_resolution": "0.25 degrees (regridded)",
            "crs": "EPSG:4326",
            "created": datetime.now(tz=timezone.utc).isoformat(),
        },
    )
    return ds


# ===================================================================
# 3. NICES HARMONIZER (YEAR-WISE)
# ===================================================================
def harmonize_nices(year: int) -> xr.Dataset | None:
    """Load and validate NICES ECV NetCDF for a single year."""
    log.info(f"  ── NICES ──")

    nices_year_dir = RAW_DIR / "nices" / str(year)
    if not nices_year_dir.exists():
        log.warning(f"    Directory not found: {nices_year_dir}")
        return None

    nc_files = sorted(nices_year_dir.glob("*.nc"))
    if not nc_files:
        log.warning(f"    No NetCDF files for {year}")
        return None

    datasets = []
    for fp in nc_files:
        ds = xr.open_dataset(str(fp))
        rename_map = {}
        for dim in ds.dims:
            dl = dim.lower()
            if "lat" in dl and dim != "latitude":
                rename_map[dim] = "latitude"
            elif "lon" in dl and dim != "longitude":
                rename_map[dim] = "longitude"
        if rename_map:
            ds = ds.rename(rename_map)
        datasets.append(ds)

    merged = xr.merge(datasets, compat="override")
    merged.attrs["year"] = year
    log.info(f"    Variables: {list(merged.data_vars)}")
    return merged


# ===================================================================
# YEAR-WISE ORCHESTRATOR
# ===================================================================
def harmonize_year(year: int):
    """Harmonize all data sources for a single year and write output."""
    year_out_dir = HARMONIZED_DIR / str(year)
    year_out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"{'─' * 50}")
    log.info(f"  Year {year}")
    log.info(f"{'─' * 50}")

    # IMD
    imd_out = year_out_dir / f"imd_grid_{year}.nc"
    ds_imd = harmonize_imd(year)
    if ds_imd is not None:
        ds_imd.to_netcdf(str(imd_out), engine="netcdf4",
                         encoding={v: {"zlib": True, "complevel": 4} for v in ds_imd.data_vars})
        size_mb = imd_out.stat().st_size / (1024 * 1024)
        log.info(f"    ✓ {imd_out.name} ({size_mb:.1f} MB)")
    else:
        log.warning(f"    ✗ No IMD data for {year}")

    # MOSDAC
    mosdac_out = year_out_dir / f"insat_L2B_{year}.nc"
    ds_mosdac = harmonize_mosdac(year)
    if ds_mosdac is not None:
        ds_mosdac.to_netcdf(str(mosdac_out), engine="netcdf4",
                            encoding={v: {"zlib": True, "complevel": 4} for v in ds_mosdac.data_vars})
        size_mb = mosdac_out.stat().st_size / (1024 * 1024)
        log.info(f"    ✓ {mosdac_out.name} ({size_mb:.1f} MB)")
    else:
        log.warning(f"    ✗ No MOSDAC data for {year}")

    # NICES
    nices_out = year_out_dir / f"nices_ecv_{year}.nc"
    ds_nices = harmonize_nices(year)
    if ds_nices is not None:
        ds_nices.to_netcdf(str(nices_out), engine="netcdf4",
                           encoding={v: {"zlib": True, "complevel": 4} for v in ds_nices.data_vars})
        size_mb = nices_out.stat().st_size / (1024 * 1024)
        log.info(f"    ✓ {nices_out.name} ({size_mb:.1f} MB)")
    else:
        log.warning(f"    ✗ No NICES data for {year}")


# ===================================================================
# REGRIDDING UTILITIES
# ===================================================================
def _regrid_to_target(da: xr.DataArray, src_lat, src_lon) -> xr.DataArray:
    """Regrid a (time × lat × lon) DataArray to the target 0.25° grid."""
    target_lat_grid, target_lon_grid = np.meshgrid(TARGET_LAT, TARGET_LON, indexing="ij")
    target_points = np.column_stack([target_lat_grid.ravel(), target_lon_grid.ravel()])

    regridded_slices = []
    for t in range(da.shape[0]):
        filled = _fill_nan_nearest(da.values[t])
        interp = RegularGridInterpolator(
            (src_lat, src_lon), filled,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        result = interp(target_points).reshape(len(TARGET_LAT), len(TARGET_LON))
        regridded_slices.append(result)

    return xr.DataArray(
        data=np.stack(regridded_slices, axis=0),
        dims=["time", "latitude", "longitude"],
        coords={"time": da.coords["time"].values, "latitude": TARGET_LAT, "longitude": TARGET_LON},
        attrs=da.attrs,
    )


def _regrid_2d(data_2d, src_lat, src_lon, tgt_lat, tgt_lon):
    """Regrid a single 2D array."""
    filled = _fill_nan_nearest(data_2d)
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        filled = filled[::-1, :]
    if src_lon[0] > src_lon[-1]:
        src_lon = src_lon[::-1]
        filled = filled[:, ::-1]

    interp = RegularGridInterpolator(
        (src_lat, src_lon), filled,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    tgt_lat_grid, tgt_lon_grid = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
    points = np.column_stack([tgt_lat_grid.ravel(), tgt_lon_grid.ravel()])
    return interp(points).reshape(len(tgt_lat), len(tgt_lon))


def _fill_nan_nearest(arr):
    """Fill NaN values with nearest non-NaN neighbor."""
    from scipy.ndimage import distance_transform_edt
    mask = np.isnan(arr)
    if not mask.any():
        return arr.copy()
    filled = arr.copy()
    _, indices = distance_transform_edt(mask, return_distances=True, return_indices=True)
    filled[mask] = arr[indices[0][mask], indices[1][mask]]
    return filled


# ===================================================================
# SUMMARY
# ===================================================================
def print_summary(start_year, end_year):
    """Print a summary of all harmonized year-wise files."""
    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║         HARMONIZED DATASET SUMMARY (Year-wise)      ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    grand_total = 0

    for year in range(start_year, end_year + 1):
        year_dir = HARMONIZED_DIR / str(year)
        if not year_dir.exists():
            continue

        files = sorted(year_dir.glob("*.nc"))
        year_size = sum(f.stat().st_size for f in files)
        grand_total += year_size

        file_names = [f.name for f in files]
        log.info(f"\n  📂 {year}/  ({year_size / (1024*1024):.1f} MB)")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            try:
                ds = xr.open_dataset(str(f))
                vars_str = ", ".join(ds.data_vars)
                dims = dict(ds.sizes)
                log.info(f"     📄 {f.name:30s} {size_mb:6.1f} MB  │ {vars_str}")
                ds.close()
            except Exception:
                log.info(f"     📄 {f.name:30s} {size_mb:6.1f} MB")

    log.info(f"\n  ── Target Grid: 0.25° × 0.25° │ {len(TARGET_LAT)}×{len(TARGET_LON)} │ EPSG:4326")
    log.info(f"  ── Grand Total: {grand_total / (1024*1024):.1f} MB ({grand_total / (1024*1024*1024):.2f} GB)")


# ===================================================================
# MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Pan-India Climate Digital Twin — Data Harmonizer"
    )
    parser.add_argument("--start-year", type=int, default=2014,
                        help="Start year (default: 2014)")
    parser.add_argument("--end-year", type=int, default=2023,
                        help="End year (default: 2023)")
    args = parser.parse_args()

    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  Pan-India Climate Digital Twin — Data Harmonizer   ║")
    log.info(f"║  Period: {args.start_year} – {args.end_year}  ({args.end_year - args.start_year + 1} years)                        ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("")

    for year in range(args.start_year, args.end_year + 1):
        harmonize_year(year)
        log.info("")

    print_summary(args.start_year, args.end_year)

    log.info("")
    log.info("=" * 60)
    log.info("HARMONIZATION COMPLETE")
    log.info(f"Output: {HARMONIZED_DIR.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
