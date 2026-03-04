import xarray as xr
import pyproj
from projection_dictionary import crs

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LightSource

class PancarraTIBase:
    """ 
    Class for importing pan-arctic CARRA 2 data and transforming it to
    a glare/glide model grid
    """
    Nx, Ny = 2869, 2869
    Dx, Dy = 2500.0, 2500.0
    R = 6371229.0
    lon0 = -30.0          # orientationOfTheGridInDegrees
    lat_ts = 90.0         # LaDInDegrees
    lat1 = 42.113802
    lon1 = 288.383154     # GRIB often uses 0..360

    # The parameters of pan-arctic CARRA2 grid
    proj = pyproj.Proj(
            proj="stere",
            lat_0=90,
            lat_ts=lat_ts,
            lon_0=lon0,
            R=R,
            x_0=0,
            y_0=0
        )


    def __init__(self,precip_path,t2m_path,orog_path):
        
        self.precip_path = precip_path
        self.t2m_path = t2m_path
        self.orog_path = orog_path

        self.precip_dataset = xr.open_dataset(precip_path)
        self.temp_dataset = xr.open_dataset(t2m_path)
        self.orog_dataset = xr.open_dataset(orog_path)
        
        # Build PanCarra2 grid
        x1, y1 = self.proj(self.lon1, self.lat1)

        self.x = x1 + self.Dx * np.arange(self.Nx)
        self.y = y1 + self.Dy * np.arange(self.Ny)

    def transform_to(self,X,Y,proj_from_crs):
        # Convert from a different projection to the PanCarra2 Coord system
        transform = pyproj.Transformer.from_proj(proj_from_crs,self.proj,always_xy=True)
        return transform.transform(X,Y)

    def interpolate(self,X,Y,time_index='mean',key='precip',method='linear'):
        if key=='precip':
            dataset=self.precip_dataset['tp']
        elif key=='t2m':
            dataset=self.temp_dataset['t2m']
        else:
            dataset=self.orog_dataset['orog']
        if time_index=='mean':
            field = dataset.mean(axis=0).T.values
        elif time_index==None:
            field = dataset.T.values
        else:
            field = dataset[time_index].T.values

        interpolant = RegularGridInterpolator((self.x,self.y),field,method=method)
        return interpolant((X,Y))

year = 2012
panc = PancarraTIBase(f"../data/pancarra/{year}/precip/precip.nc", 
                       f"../data/pancarra/{year}/t2m/t2m.nc",
                       "../data/pancarra/topo/topo.grib")

dem = xr.load_dataset('../data/cop30/cop90_reprojected.nc')
X,Y = np.meshgrid(dem.x,dem.y)
X_,Y_ = panc.transform_to(X,Y,crs)

#temp_gridded = panc.interpolate(X_,Y_,time_index='mean',key='t2m')
from scipy.special import erf
def monthly_pdd(mu, sigma, days):
    alpha = mu / sigma
    phi = np.exp(-0.5 * alpha**2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1.0 + erf(alpha / np.sqrt(2)))
    return days * (sigma * phi + mu * Phi)

def snow_fraction(mu, sigma, Ts=0.0):
    # f_s = Phi((Ts - mu)/sigma)
    z = (Ts - mu) / sigma
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

lapse = -0.003
z_oro = panc.interpolate(X_,Y_,time_index=None,key='orog',method='cubic')
t2m_fields = [panc.interpolate(X_,Y_,time_index=i,key='t2m',method='cubic') - 273 + lapse*(dem.elevation-z_oro) for i in range(12)]
precip_fields = [panc.interpolate(X_,Y_,time_index=i,key='precip',method='cubic') for i in range(12)]

t2m_fields = np.stack(t2m_fields,axis=0).astype(np.float32)
precip_fields = np.stack(precip_fields,axis=0).astype(np.float32)/917*365/12

t2m_da = xr.DataArray(
        t2m_fields,
        dims=['month','y','x'],
        coords={
            'month':np.arange(0,12),
            'y': dem.y,
            'x': dem.x,
        },
        attrs = {
            "units": "Deg C",
            "long_name": "Monthly average temperatures derived from pan-arctic CARRA2",
        }
    )

precip_da = xr.DataArray(precip_fields,
        dims = ['month','y','x'],
        coords = {"month":np.arange(0,12),
                  "y": dem.y,
                  "x": dem.x},
        attrs = {"units": "m ice equivalent per month",
                 "long_name": "Monthly accumulated precipitation derived from pan-arctic CARRA2"}
        )


out_ds = dem.copy()
out_ds["monthly_t2m"] = t2m_da
out_ds["monthly_precip"] = precip_da

out_ds.to_netcdf("../data/gridded_climate.nc")






"""

pdds = np.array([monthly_pdd(f,5.0,30) for f in t2m_fields])
sfs = np.array([snow_fraction(f,5.0) for f in t2m_fields])

snowfall = np.array(precip_fields)/917*30*sfs

ddf = 0.008
smb = snowfall - ddf*pdds

ls = LightSource(azdeg=315, altdeg=45)
dx = x[1]-x[0]
hs = ls.hillshade(z, vert_exag=10, dx=dx, dy=dx)
plt.pcolormesh(x,y,hs,cmap=plt.cm.grey)
plt.pcolormesh(x,y,smb.sum(axis=0),alpha=0.5,cmap=plt.cm.seismic,vmin=-7,vmax=7)
plt.contour(x,y,smb.sum(axis=0),[0])
plt.axis('equal')
"""



