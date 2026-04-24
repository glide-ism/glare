import pyproj
import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class PanCarraBase:
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
        """
        Require a path to a monthly precipitation dataset, monthly t2m dataset, and orography dataset,
        either netcdf or grib.  Precipitation/t2m needs a tp key with (12,Ny,Nx), orography elev (ny,nx) 

        """
        
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

    def interpolate(self,X,Y,time_index='mean',key='tp',method='linear'):
        
        if key=='precip':
            dataset = self.precip_dataset['tp']
        elif key=='t2m':
            dataset = self.temp_dataset['t2m']
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

    def regrid_carra2_fields(self,dem,crs,method='linear',t2m_lapse_rate=0.003):
        X,Y = np.meshgrid(dem.x,dem.y)
        X_,Y_ = self.transform_to(X,Y,crs)
        z_oro = self.interpolate(X_,Y_,time_index=None,key='orog',method=method)
        
        print("Working on t2m fields")
        t2m_fields = np.stack(
                [self.interpolate(X_,Y_,time_index=i,key='t2m',method=method) - 273 + t2m_lapse_rate*(dem.elevation-z_oro) for i in range(12)],
                axis=0)

        print("Working on precip fields")
        precip_fields = np.stack(
                [self.interpolate(X_,Y_,time_index=i,key='precip',method=method) for i in range(12)],
                axis=0)

        return z_oro.astype('float32'), t2m_fields.astype('float32'), precip_fields.astype('float32')


