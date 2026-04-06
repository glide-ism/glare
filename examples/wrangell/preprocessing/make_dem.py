import xarray as xr
import rioxarray
from projection_dictionary import crs

ds = xr.open_dataset('../data/cop30/output_hh.tif')
ds = ds.rename({'band_data':'elevation'})
ds = ds.isel(band=0)
ds = ds.drop_vars('band')

if ds.rio.crs is None:
    ds.rio.write_crs("EPSG:4326", inplace=True)

ds = ds.rio.reproject(crs,resolution=90)

da = ds["elevation"]  # or whatever your variable is
import numpy as np

def largest_valid_rectangle(valid):
    rows, cols = valid.shape
    heights = np.zeros(cols, dtype=int)
    best = (0, 0, 0, 0, 0)  # area, y_start, x_start, y_end, x_end

    for i in range(rows):
        heights = np.where(valid[i], heights + 1, 0)

        # Largest rectangle in this histogram row
        stack = []
        for j in range(cols + 1):
            h = heights[j] if j < cols else 0
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                x_start = 0 if not stack else stack[-1] + 1
                width = j - x_start
                area = height * width
                if area > best[0]:
                    best = (area, i - height + 1, x_start, i, x_start + width - 1)
            stack.append(j)

    _, y_start, x_start, y_end, x_end = best
    return y_start, y_end, x_start, x_end

valid = da.notnull().values
y_start, y_end, x_start, x_end = largest_valid_rectangle(valid)
da_trimmed = da.isel(y=slice(y_start, y_end + 1), x=slice(x_start, x_end + 1))

ds_trimmed = da_trimmed.to_dataset()

import shapely
outline = np.loadtxt('../data/outline/outline.csv',delimiter=',')
poly = shapely.Polygon(outline[:,:2])
mask = da_trimmed.rio.clip([poly],crs="EPSG:4326",invert=False,drop=False).notnull()
ds_trimmed['domain_mask'] = mask

import geopandas
geodf = geopandas.read_file('../data/outline/rgi_ak/RGI2000-v7.0-C-01_alaska.shp')
glacier_mask = ds_trimmed.elevation.rio.clip(geodf.geometry.values,geodf.crs,drop=False).notnull()
ds_trimmed['rgi_mask'] = glacier_mask

ds_trimmed.to_netcdf('../data/gridded_dem.nc')





