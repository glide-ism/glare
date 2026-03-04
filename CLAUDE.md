# CLAUDE.md

## Project Overview

GPU-accelerated enhanced temperature index surface mass balance (SMB) model for mountain glaciers and ice caps. Designed for coupling with GLIDE (GPU-accelerated Lightweight Ice Dynamics Engine) but maintained as a separate repository. The model computes potential clear-sky direct solar radiation over complex terrain and uses it within a Hock (1999) enhanced temperature index melt framework.

## Core Design Principles

- **Separate from GLIDE.** This repo handles SMB computation, solar geometry, terrain radiation, and climate downscaling. GLIDE handles ice dynamics. The two are coupled at the SMB field interface.
- **GPU-first via CuPy and custom CUDA kernels.** Performance-critical operations (horizon ray tracing, radiation integration) use custom CUDA kernels callable from CuPy. Elementwise monthly accumulations can use plain CuPy array operations.
- **xarray as the primary data interface.** All spatial fields (DEM, horizon angles, monthly radiation/temperature/precipitation climatologies, SMB output) are stored and manipulated as xarray Datasets backed by CuPy arrays where appropriate, with NetCDF as the on-disk format.
- **Hand-coded adjoints, not autograd.** The SMB model is simple enough that analytical gradients are trivial. No dependency on PyTorch for AD. The adjoint interface accepts the dynamics adjoint variable λ(x,y) from GLIDE and returns parameter gradients via explicit inner products.

## Physical Model

### Enhanced Temperature Index (Hock 1999)

```
M = (MF + a * I_pot) * T+
SMB = accumulation - M
```

- `MF`: melt factor (scalar, estimable via inverse problem)
- `a`: radiation coefficient (scalar, estimable via inverse problem)
- `I_pot`: potential clear-sky direct solar radiation (precomputed, monthly, W/m² or MJ/m²/day)
- `T+`: positive degree days per month (downscaled from reanalysis)

Start with a single DDF. Only introduce a snow/ice DDF split if inverse problem residuals show systematic elevation-band structure suggesting it's needed.

### Adjoint / Parameter Gradients

Given adjoint variable λ(x,y) from the ice dynamics model:

```
dJ/dMF = -Σ_pixel Σ_month λ_pixel * T+_month,pixel
dJ/da  = -Σ_pixel Σ_month λ_pixel * I_pot_month,pixel * T+_month,pixel
```

These are simple reductions over existing fields — no tape, no intermediate storage.

## Computational Components

### 1. Horizon Angle Computation (Custom CUDA Kernel)

- **Algorithm:** For each azimuth direction, launch one thread per DEM pixel. Each thread walks along the ray, computes elevation angles via bilinear interpolation of the DEM, and tracks the running maximum. Store `atan` of the max at the end (comparisons use raw rise/run ratios since `atan` is monotonic).
- **Ray stepping:** Use fractional pixel coordinates with a unit direction vector. Sample the DEM via bilinear interpolation of the four surrounding cells at each step. This avoids duplicate/skipped cell issues from nearest-neighbor rounding and produces smoother horizon fields.
- **Data layout:** DEM is (ny, nx) in row-major order. Output horizon angles are (n_azimuth, ny, nx).
- **Performance target:** ~28ms per azimuth on RTX 4070 for a 3000×3000 grid. Full 72-azimuth sweep in ~2 seconds. This is a one-time precomputation stored as NetCDF.
- **Coordinate convention:** DEM and horizontal distances must be in the same units (meters). If working in pixel units within the kernel, divide by cell size before taking `atan` when comparing against solar elevation angles.

### 2. Solar Geometry

Use `pvlib.solarposition` (or hand-rolled spherical astronomy) to compute solar azimuth φ and elevation α at sub-daily resolution (1-hour timesteps are sufficient).

Sun direction vector in (east, north, up) frame:

```
s = (sin(φ) cos(α), cos(φ) cos(α), sin(α))
```

where φ is azimuth clockwise from north and α is elevation above horizon.

### 3. Terrain Slope, Aspect, and Incidence Angle

Surface normal from DEM gradients (east, north, up):

```
n = (-dz/dx, -dz/dy, 1)
```

**Watch the sign convention:** if the DEM y-axis runs south (row index increases downward), negate `dz/dy`.

Cosine of incidence angle:

```
cos(θ_i) = (n · s) / |n|
```

Clamp to zero for self-shading. Multiply by shadow mask from horizon angles.

### 4. Daily/Monthly Radiation Integration

For each timestep within a day/month:
1. Compute solar position (zenith, azimuth, elevation)
2. If sun above geometric horizon: interpolate precomputed horizon angle at solar azimuth, check if solar elevation exceeds local horizon (shadow mask)
3. Compute cos(θ_i) for illuminated cells
4. Optionally apply atmospheric transmittance (Beer-Lambert with elevation-dependent path length)
5. Accumulate: `I_pot += S₀ * cos(θ_i) * shadow * dt`

Store as (12, ny, nx) monthly fields.

### 5. Temperature Downscaling

Downscale reanalysis T2m (e.g., CARRA2 at 2.5km) to the 50m DEM grid using locally-derived lapse rates:
- For each month and each local neighborhood of reanalysis cells, perform linear regression of T2m against reanalysis surface elevation.
- Apply the resulting slope (local lapse rate) and intercept to the fine-resolution DEM.
- This preserves reanalysis-resolved physics (inversions, föhn effects, maritime influence, seasonal stability variations) rather than imposing a fixed lapse rate.
- Use a spatial window of ~5×5 to 7×7 reanalysis cells for robust fits.
- Be aware of extrapolation beyond the reanalysis elevation range in steep terrain.

### 6. SMB Accumulation (CuPy Array Operations or Thin Kernel)

Monthly loop: `SMB = Σ_m [precip_m - (MF + a * I_pot_m) * T+_m]`

This is purely pointwise. If using a custom kernel, keep data as (month, y, x) for coalesced access — threads indexed by (i, j) with an inner loop over 12 months. But plain CuPy operations (12 iterations of array math) are fine and nearly as fast.

## Data Conventions

- **Coordinate system:** Projected coordinates (e.g., UTM or Alaska Albers). All horizontal distances in meters.
- **Array layout:** (y, x) for 2D fields, (month, y, x) or (azimuth, y, x) for 3D. Row-major (C order).
- **On-disk format:** NetCDF via xarray.
- **Precomputed fields cached as NetCDF:** horizon angles, slope, aspect, monthly I_pot climatology, monthly T+ and precip climatology.

## Coupling Interface with GLIDE

- **Forward:** This model produces a 2D SMB field (y, x) in m/yr ice equivalent. GLIDE reads this as a forcing field.
- **Adjoint:** GLIDE produces λ(x, y) — the adjoint of the cost function with respect to SMB. This model consumes λ and returns dJ/dMF and dJ/da via the analytical expressions above.
- **Temporal coupling:** Monthly SMB field, with options to either leave as monthly or - more commonly - convert into an annual rate for inclusion in GLIDE's timestepping, which often uses multiyear time-steps.
