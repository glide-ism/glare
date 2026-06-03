# GLARE

**GPU-accelerated enhanced temperature index surface mass balance model for mountain glaciers and ice caps.**

A high-performance, GPU-accelerated model for computing surface mass balance (SMB) on mountain glaciers and ice caps. GLARE drives an enhanced temperature index melt model (Hock 1999) with monthly climate forcing and a terrain-aware solar radiation field. It is designed for coupling with GLIDE (GPU-accelerated Lightweight Ice Dynamics Engine) but is maintained as a separate, standalone repository.

> **Solar radiation moved out.** Terrain-aware solar potential is no longer part
> of GLARE — it now lives in the companion library
> [**gtic**](https://github.com/glide-ism/gtic). GLARE consumes a *precomputed
> insolation field* (a static input), so `gtic` is only needed to generate that
> field, not to run the SMB model. See [Insolation input](#insolation-input).

## Features

- **GPU-First Performance**: A custom CUDA kernel for the full SMB integration (`smb.cu`), executed through a CuPy `RawModule`.
- **Fourier-Compressed Insolation Input**: GLARE ingests the diurnal insolation cycle as three coefficients per month (mean / cos / sin) and reconstructs sub-monthly radiation cheaply in-kernel. The field is produced upstream by `gtic`.
- **Enhanced Temperature Index Melt**: Snow/ice melt driven by positive degree-days and incidence-weighted solar radiation, with albedo that transitions between snow and ice as a function of tracked snow depth.
- **Stochastic Sub-Monthly Temperature**: Positive degree-days are integrated analytically from the monthly-mean temperature, a diurnal cycle, and a Gaussian sub-time-step distribution — no daily forcing required.
- **Water-Year Snow Bookkeeping**: Snow accumulation and depth are tracked sequentially across the water year (October → September) and reset each October.
- **xarray-Based Interface**: Spatial fields are exposed as xarray `DataArray`/`Dataset` objects, backed by CuPy arrays on the GPU.
- **Analytical Adjoints**: Hand-coded parameter and field gradients (`smb.cu::compute_smb_grad`) for inverse problems — no autograd required.
- **Optional PyTorch Bridge**: A `torch.autograd.Function` (`glare.torch.GlareStep`) wraps the analytical forward/backward so GLARE can drop into a PyTorch computation graph.

## Installation

### Requirements

- NVIDIA GPU with CUDA Compute Capability 6.0 or higher
- CUDA Toolkit (the default dependency targets CUDA 13.x via `cupy-cuda13x`)
- Python 3.9+
- pip or conda

### From Source

```bash
git clone https://github.com/glide-ism/glare.git
cd glare
pip install .
```

Editable install with development/test/docs tooling:

```bash
pip install -e ".[dev]"
```

### Optional extras

GLARE ships several optional dependency groups:

| Extra | Installs | When you need it |
|-----------|----------|------------------|
| `torch` | PyTorch | Using the `glare.torch.GlareStep` autograd bridge |
| `examples` | gtic, PyTorch, matplotlib, geopandas | Running the scripts under `examples/` (incl. generating the insolation input) |
| `dev` | pytest, pytest-cov, black, ruff | Development and testing |
| `docs` | sphinx, sphinx-rtd-theme | Building the documentation |

```bash
# Core install + PyTorch autograd support
pip install ".[torch]"

# Everything needed to run the bundled examples
pip install ".[examples]"
```

PyTorch is intentionally **optional** — it is a heavy dependency and the core
model is fully usable (including adjoints) without it. Install the `torch`
extra only if you want to embed GLARE in a PyTorch graph.

### GPU Dependencies

GLARE requires CuPy, which must match your installed CUDA toolkit. The default
dependency is `cupy-cuda13x` (CUDA 13.x). If you have a different toolkit,
install the matching wheel instead:

```bash
pip install cupy-cuda12x   # for CUDA 12.x; use cupy-cuda11x for CUDA 11.x
```

For more options, see the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

## Quick Start

### 1. Insolation input

GLARE needs a terrain-aware insolation field, supplied as three monthly Fourier
coefficients. This is a **static input** generated upstream by the
[`gtic`](https://github.com/glide-ism/gtic) library (`pip install ".[examples]"`
pulls it in):

```python
import xarray as xr
from gtic import SolarPotential

# DEM dataset with an 'elevation' (y, x) variable
dem = xr.open_dataset("dem.nc")

solar = SolarPotential(
    dem=dem,
    latitude=63.5,
    longitude=-145.0,
    grid_resolution=90.0,        # metres
    timezone="America/Anchorage",
)

# Diurnal insolation compressed to monthly Fourier modes (each is (12, ny, nx))
mean, cos_mode, sin_mode = solar.potential_fourier(2020)
```

See [Insolation input](#insolation-input) for the dataset contract if you
generate this field another way.

### 2. Surface mass balance

```python
import cupy as cp
from glare.model import ImprovedTemperatureIndex

smb_model = ImprovedTemperatureIndex(
    ny=ny, nx=nx, nt=12,
    dx=dx, dt=cp.float32(1.0 / 12),
    x0=x0, y0=y0, crs=crs,
)

# Insolation Fourier modes (the static input from gtic above)
smb_model.grid.insolation.insol_mean.set(mean)
smb_model.grid.insolation.insol_cos.set(cos_mode)
smb_model.grid.insolation.insol_sin.set(sin_mode)
smb_model.grid.insolation.rf.set(50.0)          # radiation melt factor

# Monthly climate forcing
smb_model.grid.temperature.t2m.set(monthly_t2m)  # (12, ny, nx)
smb_model.grid.temperature.mf.set(2.0)           # degree-day melt factor
smb_model.grid.precipitation.precip.set(monthly_precip)

# Forward model
smb_model.forward()

# Pull results back as xarray
smb = smb_model.grid.state.smb.to_dataarray()    # (t, y, x)
```

### 3. Adjoint / gradients

```python
# Seed the SMB adjoint (e.g. dJ/dSMB from a downstream model) and back-propagate
smb_model.backward(dJdsmb=lambda_field)           # (nt, ny, nx) CuPy array

grad_mf = smb_model.grid.temperature.mf.grad      # scalar parameter gradient
grad_rf = smb_model.grid.insolation.rf.grad
grad_t2m = smb_model.grid.temperature.t2m.grad    # per-pixel field gradient
```

## Physical Model

### Enhanced Temperature Index (after Hock 1999)

The instantaneous melt rate combines a temperature-index term and a
radiation term:

```
M = (MF + a * I_pot) * T+
```

- **MF** (`mf`): degree-day melt factor — melt per unit positive temperature.
- **a** (`rf`): radiation melt factor — melt per unit incidence-weighted solar radiation.
- **I_pot**: potential direct solar radiation, modulated by surface albedo (snow vs. ice).
- **T+**: positive degree-days, integrated analytically over the sub-monthly temperature distribution.

GLARE extends the classic formulation with:

1. **Analytic positive degree-days** from the monthly mean temperature plus a
   diurnal cycle (amplitude `daily_amp_t2m`, phase `phi_0`) and a Gaussian
   sub-time-step spread (`sigma_t2m`), using normal CDF/PDF terms.
2. **Albedo-dependent radiation melt**: melt switches smoothly between snow
   (`albedo_snow`) and ice (`albedo_ice`) via a sigmoid on tracked snow depth
   (`snow_transition_scale`).
3. **Debris cover** (`debris`) that scales snow-free (ice) melt only.
4. **Sequential water-year snow tracking**, reset each October (`start_month`).

### Insolation input

GLARE treats the terrain-aware solar radiation field as a **precomputed, static
input** rather than computing it internally. The field is the diurnal insolation
cycle decomposed into three monthly Fourier coefficients, each shaped
`(12, ny, nx)` on the model grid:

| Field | `TIMGrid` slot | Meaning |
|-------|----------------|---------|
| `monthly_solar_potential_mean` | `insolation.insol_mean` | monthly-mean daily potential |
| `monthly_solar_potential_cos`  | `insolation.insol_cos`  | cosine mode of diurnal variability |
| `monthly_solar_potential_sin`  | `insolation.insol_sin`  | sine mode of diurnal variability |

The companion [`gtic`](https://github.com/glide-ism/gtic) library produces this
field, accounting for horizon angles (terrain ray tracing), slope/aspect,
incidence angle, and soft self-shadowing. Any source that supplies the three
fields above on the model grid will work — GLARE depends on the data contract,
not on `gtic` itself.

## Project Structure

```
glare/
├── glare/
│   ├── __init__.py            # Package exports (PanCarraBase)
│   ├── model.py               # ImprovedTemperatureIndex: forward/backward driver
│   ├── grid.py                # TIMGrid + State/Geometry/Temperature/Precipitation/Insolation
│   ├── field.py               # Field / TimeField / Constant: GPU fields + xarray bridge
│   ├── operators.py           # ForwardOperators / BackwardOperators (CUDA kernel launchers)
│   ├── helpers.py             # PanCarraBase: CARRA2 climate regridding utilities
│   ├── torch.py               # Optional PyTorch autograd bridge (GlareStep)
│   └── cuda/
│       └── smb.cu             # Forward + adjoint SMB kernels
├── examples/
│   └── wrangell/              # Worked example: Wrangell ice cap
│       ├── download_inputs.py # Fetch DEM + climate inputs
│       ├── make_insolation.py # Build gridded insolation (via gtic's SolarPotential)
│       └── make_smb.py        # Build gridded SMB (ImprovedTemperatureIndex)
├── pyproject.toml             # Packaging + dependencies (PEP 621)
├── requirements.txt           # Core runtime pins
├── requirements-dev.txt       # Dev/test/docs/examples pins
├── README.md                  # This file
└── LICENSE                    # BSD-3-Clause
```

## Examples

The `examples/wrangell` directory contains an end-to-end pipeline. It requires
the `examples` extra (`pip install ".[examples]"`), which also installs `gtic`
for the insolation step:

```bash
cd examples/wrangell
python download_inputs.py     # download gridded_dem.nc + gridded_climate.nc
python make_insolation.py     # gtic -> model_inputs/gridded_insolation.nc
python make_smb.py            # glare -> model_inputs/gridded_smb.nc (+ plot)
```

## PyTorch Integration

With the `torch` extra installed, `glare.torch.GlareStep` exposes the model as a
`torch.autograd.Function`, so the analytical adjoint backs the PyTorch
`backward()`:

```python
import torch
from glare.torch import GlareStep

smb = GlareStep.apply(smb_model, t2m, precip, mf, rf, debris)
loss = some_objective(smb)
loss.backward()   # gradients flow into t2m, precip, mf, rf, debris
```

## Data Conventions

- **Coordinate System**: Projected coordinates (e.g. UTM, Alaska Albers) in metres; the CRS is carried on the grid as a `pyproj.CRS`.
- **Array Layout**: `(y, x)` for 2D fields, `(t, y, x)` for monthly time series.
- **On-Disk Format**: NetCDF via xarray.
- **GPU Arrays**: CuPy `float32` for performance.

## Coupling with GLIDE

### Forward Pass
GLARE produces a monthly SMB field (m a⁻¹ ice equivalent) consumed by GLIDE as a forcing field.

### Adjoint / Inverse Problems
GLIDE returns an adjoint variable λ(x, y); GLARE back-propagates it to parameter gradients:

```
dJ/dMF = -Σ_pixel Σ_month λ_pixel * T+_month,pixel
dJ/da  = -Σ_pixel Σ_month λ_pixel * I_pot_month,pixel * T+_month,pixel
```

Per-pixel gradients for `t2m`, `precip`, and `debris` are also produced.

## Performance

Typical performance on an RTX 4070:
- **Monthly SMB evaluation**: milliseconds (single fused CUDA kernel over 12 months)

(Solar potential precomputation timings now live with [`gtic`](https://github.com/glide-ism/gtic).)

## References

- Hock, R. (1999). A distributed temperature-index ice- and snowmelt model including potential direct solar radiation. *Journal of Glaciology*, 45(149), 101–111.

## Contributing

Contributions are welcome. Please keep to the project style:
- Format: `black` (line length 100)
- Linting: `ruff`
- Testing: `pytest`

## License

BSD-3-Clause — see the `LICENSE` file.

## Contact

For questions or issues, open an issue on [GitHub](https://github.com/glide-ism/glare/issues).
