import pyproj
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LightSource
import cupy as cp

dem = xr.load_dataset('../data/cop30/cop90_reprojected.nc')

z = cp.array(dem.elevation.values,dtype=cp.float32)
max_zenith = cp.zeros(z.shape,dtype=cp.float32)
max_j = cp.zeros(z.shape,dtype=cp.uint32)
max_i = cp.zeros(z.shape,dtype=cp.uint32)
step_size = cp.float32(1.0)

import datetime
import pytz
from pysolar.solar import get_altitude, get_azimuth

# Define location and time
latitude = 61.0
longitude = -143.0

accumulated_sunlight_hours = cp.zeros_like(z)
solar_potential = cp.zeros_like(z)

dZdy,dZdx = cp.gradient(z,50)
dZdy *= -1

for day in range(1,30):
    for hour in range(0,24):
        date = datetime.datetime(2023, 4, day, hour, 0, 0, tzinfo=pytz.timezone('America/Anchorage'))

        # Calculate angles
        altitude = get_altitude(latitude, longitude, date)
        azimuth = get_azimuth(latitude, longitude, date)
        j_basis =  cp.float32(np.sin(np.deg2rad(azimuth)))
        i_basis = -cp.float32(np.cos(np.deg2rad(azimuth)))

        ny,nx = z.shape
        kernels = cp.RawModule(code=open('../../../glare/cuda/azimuth_trace.cu','r').read())
        block_size = (16,16)
        grid_size = (nx // 16 + 1, ny // 16 + 1)

        kernel = kernels.get_function('azimuth_trace')
        kernel(grid_size,block_size,
                (max_zenith,
                 max_j,
                 max_i,
                 z,
                 j_basis,
                 i_basis,
                 step_size,
                 nx,ny))

        zenith_deg = cp.rad2deg(cp.arctan(max_zenith/50.0))
        z_i = altitude - zenith_deg
        z_sig = 1./(1 + cp.exp(-z_i/0.1))

        sin_phi = cp.sin(cp.deg2rad(azimuth))
        cos_phi = cp.cos(cp.deg2rad(azimuth))

        sin_alpha = cp.sin(cp.deg2rad(altitude))
        cos_alpha = cp.cos(cp.deg2rad(altitude))

        s = cp.array([sin_phi*cos_alpha,cos_phi*cos_alpha,sin_alpha])

        incidence = (-dZdx * sin_phi*cos_alpha - dZdy * cos_phi*cos_alpha + sin_alpha)/(cp.sqrt(dZdx**2 + dZdy**2 + 1)) 

        incidence[incidence<0] = 0.0

        accumulated_sunlight_hours += z_sig
        solar_potential += incidence * z_sig


accumulated_sunlight_hours /= 30
solar_potential /= 30

ls = LightSource(azdeg=315, altdeg=45)
dx = dem.x[1]-dem.x[0]
hs = ls.hillshade(z.get(), vert_exag=3, dx=dx.values, dy=dx.values)
plt.imshow(hs,cmap=plt.cm.gray)

#plt.imshow(z_sig.get(),alpha=0.25,cmap=plt.cm.plasma,vmin=0,vmax=1)
plt.imshow(solar_potential.get(),alpha=0.5,cmap=plt.cm.plasma)
#plt.imshow(accumulated_sunlight_hours.get(),alpha=0.5,cmap=plt.cm.plasma)
plt.show()
