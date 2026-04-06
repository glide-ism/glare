import xarray as xr
import pyproj
from projection_dictionary import crs

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LightSource
from glare import PanCarraBase 

year = 2012
panc = PanCarraBase(f"../data/pancarra/{year}/precip/precip.nc", 
                       f"../data/pancarra/{year}/t2m/t2m.nc",
                       "../data/pancarra/topo/topo.grib")

dem = xr.load_dataset('../data/gridded_dem.nc')

_, t2m_fields, precip_fields = panc.regrid_carra2_fields(dem, crs, method='cubic',t2m_lapse_rate=0.003)

t2m_da = xr.DataArray(
        t2m_fields,
        dims=['t','y','x'],
        coords={
            't':np.arange(0,12)/12.,
            'y': dem.y,
            'x': dem.x,
        },
        attrs = {
            "units": "Deg C",
            "long_name": "Monthly average temperatures derived from pan-arctic CARRA2",
        }
    )

precip_da = xr.DataArray(precip_fields/917*365,
        dims = ['t','y','x'],
        coords = {"t":np.arange(0,12)/12,
                  "y": dem.y,
                  "x": dem.x},
        attrs = {"units": "m ice equivalent / yr",
                 "long_name": "Precipitation rate derived from pan-arctic CARRA2 at monthly time steps"}
        )


out_ds = dem.copy()
out_ds["monthly_t2m"] = t2m_da
out_ds["monthly_precip"] = precip_da

out_ds.to_netcdf("../data/gridded_climate.nc")

