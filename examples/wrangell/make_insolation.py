"""Build the gridded monthly solar potential field for a domain.

Uses GLARE's SolarPotential to compute terrain-corrected insolation accounting
for slope, aspect, and self-shadowing from the DEM, then decomposes the diurnal
cycle into mean/cos/sin Fourier modes per month.

Output: {domain_path}/model_inputs/gridded_insolation.nc
"""
import argparse
from pathlib import Path

import geopandas
import numpy as np
import xarray as xr
from glare import SolarPotential


GRID_RESOLUTION_M = 90.0
TIMEZONE = "America/Anchorage"
YEAR = 2012

def _domain_centroid_latlon(domain_path: Path) -> tuple[float, float]:
    """Return (latitude, longitude) of the centroid of the domain outline,
    in EPSG:4326 degrees. Used to anchor solar geometry.
    """
    outline = geopandas.read_file(domain_path / 'outline.kml')
    if outline.crs is not None and outline.crs.to_epsg() != 4326:
        outline = outline.to_crs(4326)
    centroid = outline.geometry.unary_union.centroid
    return float(centroid.y), float(centroid.x)


domain_path = Path('./model_inputs')
dem_path = domain_path / 'gridded_dem.nc'
output_path = domain_path / 'gridded_insolation.nc'

dem = xr.load_dataset(dem_path)
latitude, longitude = _domain_centroid_latlon(domain_path)

solar = SolarPotential(
    dem=dem,
    latitude=latitude,
    longitude=longitude,
    grid_resolution=GRID_RESOLUTION_M,
    timezone=TIMEZONE,
)

mean, cos_mode, sin_mode = solar.compute_solar_potential_fourier_decomposition(YEAR)

months = np.arange(0, 12, dtype=np.float32) / 12
coords = {"t": months, "y": dem.y, "x": dem.x}
dims = ["t", "y", "x"]

mean_da = xr.DataArray(
    mean.get(), dims=dims, coords=coords,
    attrs={
        "units": "dimensionless (intensity relative to continuous orthogonal sunlight)",
        "long_name": "Monthly average daily solar potential (incidence-weighted, shadow-masked)",
    },
)
cos_da = xr.DataArray(
    cos_mode.get(), dims=dims, coords=coords,
    attrs={
        "units": "dimensionless (intensity relative to continuous orthogonal sunlight)",
        "long_name": "cos mode of diurnal variability in insolation",
    },
)
sin_da = xr.DataArray(
    sin_mode.get(), dims=dims, coords=coords,
    attrs={
        "units": "dimensionless (intensity relative to continuous orthogonal sunlight)",
        "long_name": "sin mode of diurnal variability in insolation",
    },
)

# Carry only the projection metadata from the DEM; downstream merge
# provides the elevation and mask fields.
insolation_ds = dem.copy()
insolation_ds["monthly_solar_potential_mean"] = mean_da
insolation_ds["monthly_solar_potential_cos"] = cos_da
insolation_ds["monthly_solar_potential_sin"] = sin_da
insolation_ds.attrs["source"] = f"GLARE SolarPotential, year {YEAR}"
insolation_ds.attrs["centroid_lat"] = latitude
insolation_ds.attrs["centroid_lon"] = longitude
insolation_ds.attrs["grid_resolution"] = f"{GRID_RESOLUTION_M} m"

for name in ('elevation', 'domain_mask', 'rgi_mask',
	 'topography', 'bathymetry', 'bathymetry_mask'):
    del insolation_ds[name]

insolation_ds.to_netcdf(output_path)
