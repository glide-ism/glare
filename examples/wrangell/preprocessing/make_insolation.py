"""
Compute monthly and daily solar potential for Wrangell ice cap.

Uses GLARE's SolarPotential class to compute terrain-corrected insolation
accounting for slope, aspect, and self-shadowing from the DEM.
"""

import datetime

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LightSource

from glare import SolarPotential

# Configuration
DEM_PATH = '../data/cop30/cop90_reprojected.nc'
GRID_RESOLUTION = 90.0  # meters
LATITUDE = 61.0
LONGITUDE = -143.0
TIMEZONE = "America/Anchorage"
YEAR = 2023
MONTH = 4
NUM_DAYS = 30  # April has 30 days

# Load DEM
print(f"Loading DEM from {DEM_PATH}")
dem = xr.load_dataset(DEM_PATH)
print(f"DEM shape: {dem.elevation.shape}")

# Initialize solar potential calculator
print("Initializing SolarPotential calculator...")
solar = SolarPotential(
    dem=dem,
    latitude=LATITUDE,
    longitude=LONGITUDE,
    grid_resolution=GRID_RESOLUTION,
    timezone=TIMEZONE,
)

# Accumulate solar potential over the month
print(f"Computing daily solar potential for April {YEAR}...")
accumulated_sunlight_hours = cp.zeros_like(solar.z)
daily_average_potential = cp.zeros_like(solar.z)

for day in range(1, NUM_DAYS + 1):
    daily_sunlight = cp.zeros_like(solar.z)
    daily_radiation = cp.zeros_like(solar.z)

    for hour in range(24):
        date = datetime.datetime(
            YEAR, MONTH, day, hour, 0, 0,
            tzinfo=__import__('pytz').timezone(TIMEZONE)
        )

        from pysolar.solar import get_altitude, get_azimuth
        altitude = get_altitude(LATITUDE, LONGITUDE, date)
        azimuth = get_azimuth(LATITUDE, LONGITUDE, date)

        # Only process if sun is above horizon
        if altitude > 0:
            shadow_mask = solar.compute_shadow_mask(altitude, azimuth)
            incidence = solar.compute_incidence(altitude, azimuth)

            daily_sunlight += shadow_mask
            daily_radiation += incidence * shadow_mask

    accumulated_sunlight_hours += daily_sunlight
    daily_average_potential += daily_radiation

# Average over month
accumulated_sunlight_hours /= NUM_DAYS
daily_average_potential /= NUM_DAYS

print(f"Mean daily sunlight hours: {accumulated_sunlight_hours.mean().get():.2f}")
print(f"Mean daily solar potential: {daily_average_potential.mean().get():.4f}")

# Visualization
print("Creating visualization...")
z_cpu = solar.z.get()
ls = LightSource(azdeg=315, altdeg=45)
dx = dem.x[1].item() - dem.x[0].item()  # Convert to float
hs = ls.hillshade(z_cpu, vert_exag=3, dx=dx, dy=dx)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Hillshade with solar potential overlay
axes[0].imshow(hs, cmap=plt.cm.gray)
im0 = axes[0].imshow(daily_average_potential.get(), alpha=0.5, cmap=plt.cm.plasma)
axes[0].set_title('Daily Average Solar Potential')
plt.colorbar(im0, ax=axes[0], label='Incidence-weighted radiation')

# Accumulated sunlight hours
axes[1].imshow(hs, cmap=plt.cm.gray)
im1 = axes[1].imshow(accumulated_sunlight_hours.get(), alpha=0.5, cmap=plt.cm.plasma)
axes[1].set_title('Accumulated Sunlight Hours (April avg/day)')
plt.colorbar(im1, ax=axes[1], label='Shadow-masked hours/day')

plt.tight_layout()
plt.show()
