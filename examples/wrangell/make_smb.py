import cupy as cp
import matplotlib.pyplot as plt
import xarray as xr
import torch

import pyproj

from glare.model import ImprovedTemperatureIndex
from glare.torch import GlareStep

dem = xr.load_dataset('./model_inputs/gridded_dem.nc')
clm = xr.load_dataset('./model_inputs/gridded_climate.nc')
ins = xr.load_dataset('./model_inputs/gridded_insolation.nc')
crs = pyproj.CRS(dem.spatial_ref.crs_wkt)

x           = cp.array(dem.x)
y           = cp.array(dem.y)

nx = len(x)
ny = len(y)
nt = 12
dx = (x[1] - x[0]).item()
dt = cp.float32(1./12)

smb_model = ImprovedTemperatureIndex(ny=ny,nx=nx,nt=12,
        dx=dx,dt=dt,
        x0=x[0].item(),y0=y[0].item(),
        crs=crs)

smb_model.grid.insolation.insol_mean.set(ins.monthly_solar_potential_mean)
smb_model.grid.insolation.insol_cos.set(ins.monthly_solar_potential_cos)
smb_model.grid.insolation.insol_sin.set(ins.monthly_solar_potential_sin)

smb_model.grid.temperature.t2m.set(clm.monthly_t2m.values)
smb_model.grid.precipitation.precip.set(clm.monthly_precip.values)

smb_model.grid.forward_operators.compute_forward()



