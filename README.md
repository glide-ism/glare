# GLARE

GPU-accelerated enhanced temperature index **L**ightweight ice dynamics engine **A**ctive **RE**gion surface mass balance model.

A high-performance, GPU-accelerated model for computing surface mass balance on mountain glaciers and ice caps. GLARE uses enhanced temperature index methods (Hock 1999) combined with terrain-aware solar radiation calculations for accurate, efficient simulations. Designed for coupling with GLIDE (GPU-accelerated Lightweight Ice Dynamics Engine) but maintained as a separate repository.

## Features

- **GPU-First Performance**: Custom CUDA kernels for terrain ray tracing and radiation integration
- **Terrain-Aware Solar Radiation**: Horizon angle computation, slope/aspect correction, and self-shadowing
- **Enhanced Temperature Index Melt Model**: Parametrized by melt factor (MF) and radiation coefficient (a)
- **xarray-Based Interface**: All spatial fields stored as xarray Datasets, backed by CuPy arrays on GPU
- **Analytical Adjoints**: Hand-coded parameter gradients for inverse problems, no autograd dependency
- **Monthly Resolution**: Monthly climatologies for radiation, temperature, and precipitation

## Installation

### Requirements

- NVIDIA GPU with CUDA Compute Capability 6.0 or higher
- CUDA Toolkit 11.0+
- Python 3.9+
- pip or conda

### From Source

```bash
git clone https://github.com/glide-ism/glare.git
cd glare
pip install .
```

For development mode (editable install with dev/test dependencies):

```bash
pip install -e ".[dev]"
```

### GPU Dependencies

GLARE requires CuPy, which depends on CUDA:

```bash
# Install CuPy (CuPy will auto-detect your CUDA installation)
pip install cupy-cuda11x  # Replace 11x with your CUDA version (e.g., cupy-cuda12x)
```

For more CuPy installation options, see [CuPy documentation](https://docs.cupy.dev/en/stable/install.html).

## Quick Start

```python
import xarray as xr
from glare import SolarPotential

# Load a DEM
dem = xr.open_dataset("dem.nc")

# Create solar potential calculator
solar = SolarPotential(
    dem=dem,
    latitude=63.5,
    longitude=-145.0,
    grid_resolution=50.0,  # 50m grid
    timezone="America/Anchorage"
)

# Compute monthly solar potential (12, ny, nx) array
monthly_potential = solar.compute_monthly_solar_potential(year=2020)
```

## Physical Model

### Enhanced Temperature Index (Hock 1999)

```
M = (MF + a * I_pot) * T+
SMB = accumulation - M
```

Where:
- **MF**: Melt factor (mm/day/°C) — primary degree-day factor
- **a**: Radiation coefficient (mm²/day/J) — sensitivity to solar radiation
- **I_pot**: Potential clear-sky direct solar radiation (MJ/m²/day or W/m²)
- **T+**: Positive degree-days per month
- **SMB**: Surface mass balance (accumulation minus melt)

### Solar Geometry & Terrain Shadowing

The model accounts for:
1. **Horizon angles** from terrain ray tracing (precomputed CUDA kernel)
2. **Slope and aspect** from DEM gradients
3. **Incidence angle** between sun rays and terrain surface
4. **Soft shadow mask** using sigmoid interpolation between sun above/below local horizon

## Project Structure

```
glare/
├── glare/
│   ├── __init__.py               # Package exports
│   ├── solar_potential.py        # SolarPotential class (main interface)
│   └── cuda/
│       └── azimuth_trace.cu      # CUDA kernel for horizon angle computation
├── examples/
│   └── wrangell/                 # Example: Wrangell ice cap
│       └── preprocessing/        # Data preparation scripts
├── pyproject.toml                # Modern Python packaging (PEP 518)
├── README.md                     # This file
├── LICENSE                       # BSD-3-Clause
└── CLAUDE.md                     # Development notes & architecture
```

## Data Conventions

- **Coordinate System**: Projected coordinates (e.g., UTM, Alaska Albers) in meters
- **Array Layout**: (y, x) for 2D, (month, y, x) or (azimuth, y, x) for 3D arrays
- **On-Disk Format**: NetCDF via xarray Datasets
- **GPU Arrays**: Backed by CuPy (float32 for performance)

## Coupling with GLIDE

### Forward Pass
GLARE produces a 2D SMB field (m/yr ice equivalent) consumed by GLIDE as a forcing field.

### Adjoint / Inverse Problems
GLIDE returns an adjoint variable λ(x,y), and GLARE computes parameter gradients:
```
dJ/dMF = -Σ_pixel Σ_month λ_pixel * T+_month,pixel
dJ/da  = -Σ_pixel Σ_month λ_pixel * I_pot_month,pixel * T+_month,pixel
```

## Performance

Typical performance on RTX 4070:
- **Horizon angle computation**: ~28ms per azimuth direction
- **Full 72-azimuth sweep**: ~2 seconds (one-time precomputation)
- **Monthly SMB evaluation**: Milliseconds (12 months of elementwise operations)

## References

- Hock, R. (1999). A distributed temperature-index ice-and snowmelt model including potential direct solar radiation. *Journal of Glaciology*, 45(149), 101–111.
- See `CLAUDE.md` for detailed design notes and computational algorithms.

## Contributing

Contributions are welcome. Please ensure code follows the project style:
- Format: `black` (line length 100)
- Linting: `ruff`
- Testing: `pytest`

## License

BSD-3-Clause — See `LICENSE` file.

## Contact

For questions or issues, open an issue on [GitHub](https://github.com/glide-ism/glare/issues).
