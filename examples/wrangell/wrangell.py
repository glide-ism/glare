import cupy as cp
import matplotlib.pyplot as plt
import xarray as xr
import torch

import pyproj

from glare.model import ImprovedTemperatureIndex
from glare.torch import GlareStep

dem = xr.load_dataset('data/gridded_dem.nc')
clm = xr.load_dataset('data/gridded_climate.nc')
ins = xr.load_dataset('data/gridded_insolation.nc')
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

t2m = torch.tensor(clm.monthly_t2m.values,device='cuda',dtype=torch.float32,requires_grad=True)
precip = torch.tensor(clm.monthly_precip.values,device='cuda',dtype=torch.float32,requires_grad=True)
mf = torch.tensor(smb_model.grid.temperature.mf.value,device='cuda',dtype=torch.float32,requires_grad=True)
rf = torch.tensor(smb_model.grid.insolation.rf.value,device='cuda',dtype=torch.float32,requires_grad=True)

f_smb = lambda t2m,precip,mf,rf: GlareStep.apply(smb_model,t2m,precip,mf,rf)

delta_t2m = torch.randn_like(t2m)
delta_precip = torch.randn_like(precip)
delta_mf = torch.randn_like(mf)
delta_rf = torch.randn_like(rf)

smb = GlareStep.apply(smb_model,t2m,precip,mf,rf)
L = smb.sum()
L.backward()

gvp_ad = ((delta_t2m*t2m.grad).sum() + 
        (delta_precip*precip.grad).sum() +
        (delta_mf*mf.grad).sum() +
        (delta_rf*rf.grad).sum())
 

eps = 1e-3

smb_p = GlareStep.apply(smb_model,t2m + eps*delta_t2m,
        precip + eps*delta_precip,
        mf + eps*delta_mf,
        rf + eps*delta_rf)

L_p = smb_p.sum()
        

smb_m = GlareStep.apply(smb_model,t2m - eps*delta_t2m,
        precip - eps*delta_precip,
        mf - eps*delta_mf,
        rf - eps*delta_rf)

L_m = smb_m.sum()

gvp_fd = (L_p - L_m)/(2*eps)

"""
smb_model.grid.temperature.t2m.set(clm.monthly_t2m)
smb_model.grid.precipitation.precip.set(clm.monthly_precip) 

smb_model.forward()
s0 = cp.array(smb_model.grid.state.smb.data)
#L_0 = smb_model.grid.state.smb.data.sum()
smb_model.backward(cp.ones_like(smb_model.grid.state.smb.data))#/(nt*ny*nx))



eps = 0.001

#delta = cp.random.randn(nt,ny,nx,dtype=cp.float32)
#delta = cp.zeros((nt,ny,nx),dtype=cp.float32)
#delta[4,700,900] = 1.0

smb_model.grid.temperature.t2m.data += eps
smb_model.forward()
s1 = cp.array(smb_model.grid.state.smb.data)
#L_1 = smb_model.grid.state.smb.data.sum()

#print(smb_model.grid.state.smb.data[4,700,900])
#print((L_1 - L_0)/eps)
grad = (s1 - s0)/eps


#out_ds = dem.copy()
#out_ds['surface_mass_balance'] = smb_model.grid.state.annual_smb.to_dataarray()
#out_ds.to_netcdf('./results/surface_mass_balance.nc')
"""
