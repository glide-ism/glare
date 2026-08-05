"""Build gridded monthly snowpack-enthalpy fields for a domain.

Companion to make_smb.py: instead of the PDD/radiation SMB kernel, this drives the
depth-averaged snowpack *enthalpy* model (glare.enthalpy.EnthalpyModel), which
carries an active surface reservoir of mass M and enthalpy E and resolves its
temperature, runoff and ice melt from an implicit surface energy balance.

Shortwave is driven by the seasonal insolation fields (run make_insolation.py
first): the absorbed flux is (1 - albedo) * q_sw_insol * I, where I is the
monthly-mean insolation fraction (terrain-shaded, sun-angle-modulated).

The model needs a basal/substrate temperature T_base beneath the active reservoir.
Deep ice equilibrates to roughly the annual-mean surface temperature, so we set
T_base = min(T_bar, 0), where T_bar is the annual mean of the forcing air
temperature (capped at the melting point).

Output: ./model_inputs/gridded_enthalpy.nc
"""
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import xarray as xr
import pyproj

from scipy.ndimage import gaussian_filter

from glare.enthalpy import EnthalpyModel, SECONDS_PER_YEAR
from glare.avalanche import AvalancheOperator

from matplotlib.colors import LightSource

ls = LightSource(azdeg=315, altdeg=45)

domain_path = Path('./model_inputs')
dem_path = domain_path / 'gridded_dem.nc'
climate_path = domain_path / 'gridded_climate.nc'
insolation_path = domain_path / 'gridded_insolation.nc'  # run make_insolation.py first

output_path = domain_path / 'gridded_enthalpy.nc'

dem = xr.load_dataset(dem_path)
clm = xr.load_dataset(climate_path)
ins = xr.load_dataset(insolation_path)

# Get information about the grid
crs = pyproj.CRS(dem.spatial_ref.crs_wkt)

x = cp.array(dem.x)
y = cp.array(dem.y)

nx = len(x)
ny = len(y)
nt = 12
dx = (x[1] - x[0]).item()
dt = cp.float32(1. / 12)

# Optional sub-monthly stochastic temperature: set N_SUBSTEPS > 1 to advance each
# month in that many daily sub-steps driven by demeaned Gaussian deviations drawn
# from grid.temperature.sigma_t2m (seeded below).  N_SUBSTEPS = 1 is the plain
# monthly model.  Variance tends to raise low-elevation melt (warm days cross 0 C).
N_SUBSTEPS = 30

# Build the model exactly like the ETIM example (make_smb.py): the model builds its
# own grid, which owns the forcing, the parameters (as Constants) and the output
# state, and we populate it below.  The enthalpy core is the sibling of the ETIM core.
enth = EnthalpyModel(ny=ny, nx=nx, nt=nt,
                     dx=dx, dt=dt,
                     x0=x[0].item(), y0=y[0].item(),
                     crs=crs, start_month=9,   # start_month=9 -> October water year
                     n_substeps=N_SUBSTEPS, glacier_surface=True, seed=0)
grid = enth.grid

# Forcing from CARRA2 (monthly means): 2 m temperature and precipitation.
grid.temperature.t2m.set(clm.monthly_t2m.values)
grid.precipitation.precip.set(clm.monthly_precip.values)

# Seasonal shortwave driver: monthly-mean insolation (fraction of direct sun).
grid.radiation.insol_mean.set(ins.monthly_solar_potential_mean.values)

# Shortwave comes entirely from the seasonal insolation field (q_sw_bulk = 0):
# absorbed flux = (1 - albedo) * q_sw_insol * insol_mean, with q_sw_insol the
# clear-sky flux at full direct sun (insol = 1).  Parameters live on the grid.
grid.radiation.q_sw_bulk.set(0.0)
grid.radiation.q_sw_insol.set(600.0 * SECONDS_PER_YEAR)   # J m-2 yr-1 at insol = 1
grid.thermodynamics.H_atm.set(20 * SECONDS_PER_YEAR)

# Basal/substrate temperature: deep ice equilibrates to the annual-mean surface
# temperature, capped at the melting point -> T_base = min(T_bar, 0), where T_bar
# is the annual mean of the forcing air temperature.
t_base = cp.minimum(grid.temperature.t2m.data.mean(axis=0), cp.float32(0.0))
grid.geometry.t_base.set(t_base)

# Avalanche redistribution: relocate precip off steep, ice-free terrain onto
# downslope deposition zones before the recurrence (mirrors make_smb.py).  The
# operator reads the DEM from grid.geometry.srf.  Set enth.avalanche = None to disable.
grid.geometry.srf.set(gaussian_filter(dem.topography.values, 1))
enth.avalanche = AvalancheOperator(grid, s_crit=30.0, w_trans=8.0, p=1.5, K=40)

enth.forward()

# Convert to xarray and save the full monthly state/flux stack.  The surface mass
# balance `smb` [m a-1 w.e.] is the glacier product; the rest are diagnostics.
st = grid.state
ds = xr.Dataset({
    'smb': st.smb.to_dataarray(),
    'M': st.M.to_dataarray(),
    'E': st.E.to_dataarray(),
    'runoff': st.runoff.to_dataarray(),
    'ice_melt': st.ice_melt.to_dataarray(),
    't_surface': st.t_surface.to_dataarray(),
    'albedo': st.albedo.to_dataarray(),
})
ds['t_base'] = grid.geometry.t_base.to_dataarray()
ds.to_netcdf(output_path)

# --------------------------------------------------------------------------- #
# Headline product: the annual surface mass balance, sum_t smb * dt [m w.e./yr].
# Positive (blue) = net accumulation; negative (red) = net ablation.
# --------------------------------------------------------------------------- #
annual_smb = (grid.state.smb.data.get() * float(dt)).sum(axis=0)   # (ny, nx) m w.e./yr
print(f"annual SMB: domain mean = {annual_smb.mean():.3f} m w.e./yr, "
      f"range = [{annual_smb.min():.2f}, {annual_smb.max():.2f}]")

shaded = ls.hillshade(dem.topography.values, vert_exag=1.0)

figa, axa = plt.subplots(figsize=(7, 5))
ca = axa.imshow(annual_smb, cmap=plt.cm.RdBu, vmin=-5, vmax=5)
axa.set_xticks([]); axa.set_yticks([])
axa.set_title('Annual surface mass balance (m w.e./yr)')
figa.colorbar(ca, ax=axa, fraction=0.046, pad=0.04)
figa.savefig(domain_path / 'annual_smb.png', dpi=150, bbox_inches='tight')

# --------------------------------------------------------------------------- #
# Plotting: snow-water-equivalent (M/rho_w), ice melt and surface temperature
# for January, July, and the annual mean.
# --------------------------------------------------------------------------- #
rho_w = float(grid.thermodynamics.rho_w.value)
swe = grid.state.M.data.get() / rho_w                 # (nt, ny, nx) m w.e.
icemelt_rate = grid.state.ice_melt.data.get() / (rho_w * float(dt))   # m w.e. / yr
tsurf = grid.state.t_surface.data.get()

fig, axs = plt.subplots(nrows=4, ncols=3,
                        gridspec_kw={'height_ratios': [1, 1, 1, 0.1]})

rows = [(0, 'January'), (6, 'July'), (None, 'Annual')]
for r, (m, label) in enumerate(rows):
    swe_r = swe[m] if m is not None else swe.mean(axis=0)
    melt_r = icemelt_rate[m] if m is not None else icemelt_rate.mean(axis=0)
    tsurf_r = tsurf[m] if m is not None else tsurf.mean(axis=0)

    c0 = axs[r, 0].imshow(swe_r, cmap=plt.cm.winter, vmin=0, vmax=2)
    c1 = axs[r, 1].imshow(melt_r, cmap=plt.cm.YlOrRd, vmin=0, vmax=5)
    c2 = axs[r, 2].imshow(tsurf_r, cmap=plt.cm.RdBu_r, vmin=-20, vmax=0)
    axs[r, 0].set_ylabel(label)
    for c in range(3):
        axs[r, c].set_xticks([])
        axs[r, c].set_yticks([])

axs[0, 0].set_title('Snow (m w.e.)')
axs[0, 1].set_title('Ice melt (m w.e./yr)')
axs[0, 2].set_title('Surface temp (C)')

plt.colorbar(c0, cax=axs[3, 0], orientation='horizontal')
plt.colorbar(c1, cax=axs[3, 1], orientation='horizontal')
plt.colorbar(c2, cax=axs[3, 2], orientation='horizontal')

fig.subplots_adjust(wspace=0.05, hspace=0.05)
fig.savefig(domain_path / 'gridded_enthalpy.png', dpi=150, bbox_inches='tight')
print(f"wrote {output_path} and gridded_enthalpy.png")
