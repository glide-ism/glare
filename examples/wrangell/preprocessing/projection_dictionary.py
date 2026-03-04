from pyproj import CRS

# Example: Create a custom CRS from a PROJ dictionary
custom_proj_dict = {
    'proj': 'aea',
    'lat_0': 50,
    'lat_1': 55,
    'lat_2': 65,
    'lon_0': -143.25,
    'x_0': 0,
    'y_0': 0,
    'ellps': 'WGS84',
    'datum': 'NAD83',
    'units': 'm',
    'no_defs': True
}

crs = CRS.from_dict(custom_proj_dict)
