"""Build the gridded monthly surface mass balance field for a domain.

Uses GLARE's improved temperature index model to compute the surface mass
balance based on CARRA2 climate variables and topography-based solar radiation.

Output: ./model_inputs/gridded_smb.nc
"""
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import xarray as xr
import torch

import pyproj

from scipy.ndimage import gaussian_filter

from glare.model import ImprovedTemperatureIndex
from glare.avalanche import AvalancheOperator

domain_path = Path('./model_inputs')
dem_path = domain_path / 'gridded_dem.nc'
climate_path = domain_path / 'gridded_climate.nc'
insolation_path = domain_path / 'gridded_insolation.nc' # Run make_insolation.py before running this.

output_path = domain_path / 'gridded_smb.nc'

dem = xr.load_dataset(dem_path)
clm = xr.load_dataset(climate_path)
ins = xr.load_dataset(insolation_path)

# Get information about the grid
crs = pyproj.CRS(dem.spatial_ref.crs_wkt)

x           = cp.array(dem.x)
y           = cp.array(dem.y)

nx = len(x)
ny = len(y)
nt = 12
dx = (x[1] - x[0]).item()
dt = cp.float32(1./12)

# Build the smb model
smb_model = ImprovedTemperatureIndex(ny=ny,nx=nx,nt=12,
        dx=dx,dt=dt,
        x0=x[0].item(),y0=y[0].item(),
        crs=crs)

# Surface elevation: drives the avalanche redistribution routing.
smb_model.grid.geometry.srf.set(gaussian_filter(dem.elevation.values,1))

# Set the fields required for insolation (which works on Fourier modes for compression)
smb_model.grid.insolation.insol_mean.set(ins.monthly_solar_potential_mean)
smb_model.grid.insolation.insol_cos.set(ins.monthly_solar_potential_cos)
smb_model.grid.insolation.insol_sin.set(ins.monthly_solar_potential_sin)
smb_model.grid.insolation.rf.set(50.0)

# Set 2m temperature from CARRA2 (monthly means)
smb_model.grid.temperature.t2m.set(clm.monthly_t2m.values)
smb_model.grid.temperature.mf.set(2.0)

# Set precipitation from CARRA2 (monthly means)
smb_model.grid.precipitation.precip.set(clm.monthly_precip.values)

# Avalanche redistribution: relocate solid precip off steep, ice-free terrain onto
# downslope deposition zones before melt is applied.  Set to None to disable.
smb_model.avalanche = AvalancheOperator(smb_model.grid,
        s_crit=30.0, w_trans=8.0, p=1.5, K=40)

# Compute the surface mass balance.  model.forward() partitions precip into its
# solid fraction, applies the avalanche operator, then runs the SMB kernel.
smb_model.forward()

# Compute to xarray data array
smb_ds = smb_model.grid.state.smb.to_dataarray()

# Save
smb_ds.to_netcdf(output_path)

# Plotting
fig,axs = plt.subplots(nrows=4,ncols=2,gridspec_kw={'height_ratios':[1,1,1,0.1]})
axs[0,0].imshow(smb_model.grid.state.smb.data[0].get(),cmap=plt.cm.seismic,vmin=-5,vmax=5,alpha=0.5)
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])
axs[0,0].set_title('SMB (m/yr)')
axs[0,0].set_ylabel('January')

axs[0,1].imshow(smb_model.grid.insolation.insol_mean.data[0].get(),cmap=plt.cm.gray)
axs[0,1].imshow(smb_model.grid.state.snow_depth.data[0].get(),cmap=plt.cm.winter,vmin=0,vmax=2,alpha=0.5)
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,1].set_title('Snow depth (m. w.e.)')

axs[1,0].imshow(smb_model.grid.insolation.insol_mean.data[6].get(),cmap=plt.cm.gray)
axs[1,0].imshow(smb_model.grid.state.smb.data[6].get(),cmap=plt.cm.seismic,vmin=-5,vmax=5,alpha=0.5)
axs[1,0].set_xticks([])
axs[1,0].set_yticks([])
axs[1,0].set_ylabel('July')

axs[1,1].imshow(smb_model.grid.insolation.insol_mean.data[6].get(),cmap=plt.cm.gray)
axs[1,1].imshow(smb_model.grid.state.snow_depth.data[6].get(),cmap=plt.cm.winter,vmin=0,vmax=2,alpha=0.5)
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])

axs[2,0].imshow(smb_model.grid.insolation.insol_mean.data.mean(axis=0).get(),cmap=plt.cm.gray)
c1 = axs[2,0].imshow(smb_model.grid.state.smb.data.mean(axis=0).get(),cmap=plt.cm.seismic,vmin=-5,vmax=5,alpha=0.5)
plt.colorbar(c1,cax=axs[3,0],orientation='horizontal')
axs[2,0].set_xticks([])
axs[2,0].set_yticks([])
axs[2,0].set_ylabel('Annual')

axs[2,1].imshow(smb_model.grid.insolation.insol_mean.data.mean(axis=0).get(),cmap=plt.cm.gray)
c2 = axs[2,1].imshow(smb_model.grid.state.snow_depth.data.mean(axis=0).get(),cmap=plt.cm.winter,vmin=0,vmax=2,alpha=0.5)
plt.colorbar(c2,cax=axs[3,1],orientation='horizontal')
axs[2,1].set_xticks([])
axs[2,1].set_yticks([])

fig.subplots_adjust(wspace=0.05,hspace=0.05)






