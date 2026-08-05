# Snowpack Enthalpy Model — Design Synopsis

A depth-averaged snowpack thermodynamics core for GLIDE/GLARE. Carries an active
surface reservoir of mass and enthalpy and resolves its temperature, refreezing,
runoff and ice melt from an implicit surface energy balance. It is a **co-equal
sibling** of the PDD/radiation temperature-index core (`cuda/smb.cu`) — a physically
explicit alternative for the snowpack. Both cores subclass an agnostic base `Grid`
(`TIMGrid`, `EnthalpyGrid`), keep their parameters on the grid as `Constant`s and
their output state on `grid.state`, and expose the same surface: `model.forward()`,
`model.adjoint(...)`, `model.grid.state.<field>`.

## State and convention

Per pixel the model carries two prognostic variables:

- `M` — active surface-reservoir mass [kg m⁻²],
- `E` — reservoir enthalpy relative to ice at 0 °C [J m⁻²].

Liquid water drains immediately, so the **retained** state obeys `M ≥ 0` and
`E ≤ 0`. For `M > 0` the surface temperature is `T_s = E / (M c_i)`. A positive
provisional enthalpy is meltwater: it is converted to runoff at 0 °C (and, on a
glacier surface, to ice melt once the snow is exhausted) and never stored.

## Physical model

One forcing step (`step` in `glare/enthalpy.py`):

1. **Advective input.** Precipitation adds mass `dM = max(P,0)·dt` and specific
   enthalpy `h(T_air)`, split by a smooth rain fraction. Solid precip is capped at
   ≤ 0 °C and liquid at ≥ 0 °C so the smoothing interval never implies warm ice or
   supercooled rain.
2. **Albedo** from the *diagnostic solid mass* (liquid drained for the optics)
   via a snow-cover sigmoid → absorbed shortwave
   `q_sw = (1 − α)·(q_sw_bulk + q_sw_insol·I)`. The seasonal term scales
   `q_sw_insol` (the clear-sky flux at full direct sun) by the monthly-mean
   insolation fraction `I` (`grid.insolation.insol_mean`) — the terrain-shaded,
   sun-angle-modulated shortwave. `q_sw_insol = 0` recovers a constant bulk model.
3. **Linear exchange.** `q = q_sw + H_atm(T_air − T_s) + H_base(T_base − T_s)`,
   with a snow-insulation reduction of `H_base`.
4. **Implicit projection.** The cold branch (`T_s ≤ 0`) is solved by **backward
   Euler**. If the implicit solution crosses 0 °C, `T_s` is pinned to 0 °C and the
   surplus energy becomes runoff then ice melt. A bare surface (`M ≈ 0`) is treated
   as quasi-steady rather than storing negative enthalpy in a zero-mass state.

The implicit update on the linear terms is what lets the model take monthly steps
without a stability penalty.

## Output: surface mass balance

The glacier product is the monthly surface mass balance `smb` [m a⁻¹ w.e.]:

`smb = (M_end − M_start − ice_melt) / (ρ_w · dt)`

— the net change of the retained reservoir mass, less glacier-ice melt, per unit
time. Rain that falls and drains within a step contributes ~0 (no spurious
accumulation); snow/ice that melts off drives it negative. Summing `smb·dt` over
the water year telescopes to `(M_final − Σ ice_melt)/ρ_w` = accumulation (snow that
survived to firn) minus ablation (ice melt) — the annual glacier SMB, with the
equilibrium line falling out where it crosses zero. The reservoir state `(M, E)`,
`runoff`, `ice_melt`, `t_surface` and `albedo` are retained as diagnostics.

## Order of operations in GLARE

The recurrence is a **single water-year sweep** (`start_month → +nt`) from a cold,
snow-free surface, month to month with no annual reset — the seasonal pack builds
and ablates over the cycle. This mirrors `compute_smb`'s water-year convention and
its calendar-ordered output layout, so the two cores are drop-in comparable.

Forcing is the `EnthalpyGrid`'s `temperature.t2m` [°C], `precipitation.precip`
[m a⁻¹ w.e.] and `radiation.insol_mean` [fraction of direct sun], plus the static
basal-temperature field `geometry.t_base` [°C]. Parameters live on the grid as
`Constant`s (`thermodynamics.*`, `radiation.*`, `temperature.{sigma_t2m,
T_transition}`); output state lives on `grid.state`. Precipitation is converted to
SI mass internally with `rho_w`; all state/flux outputs are SI.

## Avalanche redistribution (optional)

Like ETIM, the model accepts the shared differentiable `AvalancheOperator` (a fixed
mass-conserving linear map `R` read off `grid.geometry.srf`): attach it as
`enth.avalanche = AvalancheOperator(enth.grid, …)`. `forward` redistributes the raw
precip downslope into `precipitation.precip_eff` (the field the kernel consumes; a
straight copy when no operator is attached), and `adjoint` pulls the precip gradient
back through `R^T`. Unlike ETIM there is **no snow/rain partition** — `R` acts on the
raw *total* precip (rain can run out over cliffs too), and the enthalpy kernel
re-derives rain/snow per sub-step at the destination temperature. The one
approximation: avalanched mass takes on the input enthalpy of its *deposition*
elevation rather than its origin — acceptable, as with the ETIM snow relocation.

## Sub-monthly stochastic temperature (optional)

Temperature-index melt depends on sub-monthly *variance*: near or below 0 °C, warm
excursions cross the melt threshold while cold ones only refreeze, so variance
raises melt. The PDD core captures this **analytically** (`Phi`/`phi`, std
`sigma_t2m`). The enthalpy model can't — it is **stateful** (cold-content,
refreezing, albedo feedback), so no closed-form expectation over the monthly
temperature distribution exists.

Instead each month is advanced in `n_substeps` explicit sub-steps of length
`dt/n_substeps`, carrying `(M, E)`, with the sub-step air temperature
`t2m[m] + temp_dev[m, d]`. Input and output stay monthly: fluxes (`runoff`,
`ice_melt`) are summed over sub-steps, diagnostics (`t_surface`, `albedo`)
averaged, and `smb` recomputed on the month boundary (so the annual identity is
preserved). Precip is distributed across sub-steps and rain/snow-partitioned at
each sub-step's temperature, so a variable month naturally splits into rain and
snow days. `n_substeps = 1` with zero deviations reduces **exactly** to the plain
monthly model.

Design constraints (by request): the **randomness never enters the kernel** — it
receives a precomputed `(nt, n_substeps)` vector of °C deviations; and the
deviations are **spatially uniform** (one domain-wide weather realisation shared by
all pixels). `EnthalpyModel(..., n_substeps=k, seed=…)` draws `sigma_t2m · z`
(per-month demeaned, so the monthly mean is preserved) via a seeded NumPy generator
when `forward()` is called without an explicit vector; `generate_temp_deviations`
exposes the draw. Per-pixel stochasticity would generalise `temp_dev` to
`(nt, n_substeps, ny, nx)` — the kernel index is the only change.

## Implementation

Three pieces kept in lock-step, exactly as the avalanche module keeps its Python
`partition_forward` aligned with `avalanche.cu`:

- **Scalar reference** (`State`/`Fluxes`/`step`/`run_column`) — a readable NumPy
  implementation that is the physics of record and the test oracle; the packaged
  form of `enthalpy_sketch_revised.py`. Its scalars come from the `EnthalpyGrid`
  `Constant`s via `scalar_params(grid)` (there is no `Parameters` dataclass — the
  grid is the single source of truth).
- **Grid + operators** (`EnthalpyGrid`, `Enthalpy{Forward,Backward}Operators`) —
  the grid owns forcing, parameter `Constant`s and output state; the grid-cached
  operators compile and launch `cuda/enthalpy.cu` (the ETIM pattern).
- **`EnthalpyModel`** — a thin driver (sibling of `ImprovedTemperatureIndex`): fills
  `grid.temp_dev`, calls `grid.forward_operators`/`grid.backward_operators`. One
  thread per pixel runs the identical water-year recurrence on the GPU, reproducing
  the reference to floating-point tolerance.

## Differentiability

The model is **reverse-mode differentiable**. The recurrence is written like
`compute_smb`'s, so the hand-derived adjoint follows the same pattern: a forward
replay checkpoints the `(M, E)` water-year recurrence in thread-local memory
(gradient checkpointing), then a reverse-in-time scan carries the state adjoints
`M̄` and `Ē`. The delicate parts are the three regimes (implicit cold branch / melt
/ bare surface) and the numerical-cleanup clamp, each of which gates the carried
adjoint — the same pattern `compute_smb_grad` uses for the snow-depth clamp. Every
sub-step is recomputed forward (capturing its branch selectors) and transposed, so
the adjoint resolves sub-monthly variance exactly as the forward does.

**Targets.** Gradients are produced for the forcing (`t2m`, `precip`, `insol`,
`t_base`) and the seven energy-balance parameters (`H_atm`, `H_base0`, `q_sw_bulk`,
`q_sw_insol`, `albedo_snow`, `albedo_ice`, `M_albedo`) — the invertible set from the
synopsis. The thermodynamic constants (`L_f`, `c_i`, `c_w`, `T_transition`,
`M_insulation`, `rho_w`) are held fixed. The loss is seeded through the mass/energy
outputs (`smb`, `M`, `E`, `runoff`, `ice_melt`); the pure diagnostics (`t_surface`,
`albedo`) feed no state and are not differentiation targets. The monthly `smb`
couples two months (`smb = (M_end − M_prev − ice_melt)/(ρ_w·dt)`), so its adjoint
seed feeds the end-of-month state, the ice-melt flux, and — one month back — the
start-of-month state.

**Layers**, kept in lock-step exactly like the forward: a scalar reference
(`_substep_forward`/`_substep_backward`/`run_column_adjoint`, returning a dict of
gradients) that is the VJP of record and the finite-difference oracle, and the CUDA
kernel (`cuda/enthalpy.cu::compute_enthalpy_grad`, driven by `EnthalpyModel.adjoint`
via `EnthalpyBackwardOperators`). Like ETIM's adjoint, the GPU pass **writes
gradients directly into the grid**: the forcing into `Field.grad` (`t2m`, `precip`,
`insol_mean`, `t_base`) and the seven parameters into their `Constant.grad` (per-pixel
on the GPU, reduced to scalars on the host — the `grad_mf`/`grad_rf` convention). The
reverse pass linearises about the most recent `forward()` and reuses `grid.temp_dev`.

## Test suite (`tests/test_enthalpy.py`)

- **Reference ↔ sketch:** the packaged scalar model reproduces
  `enthalpy_sketch_revised.py` to the digits it prints.
- **Kernel ↔ reference:** `compute_enthalpy` matches `run_column` column-by-column
  (fast-math off, so the only gap is FP32 rounding over the recurrence).
- **Mass conservation:** over the water year, total precipitation mass leaves only
  as pack runoff or is retained in `M` (no burial in v1; ice melt is a separate
  glacier source).
- **SMB identity:** the water-year sum of `smb·dt` equals the net annual balance
  `(M_final − Σ ice_melt)/ρ_w`.
- **Cold accumulation:** a cold non-glacier column melts and runs off nothing and
  retains exactly the accumulated precipitation mass.
- **Sub-monthly loop:** with `n_substeps > 1` and a fixed deviation vector the
  kernel still matches `run_column` column-by-column and conserves mass; and for a
  near-freezing column, demeaned temperature variance strictly increases melt (the
  stateful analog of the PDD variance effect).
- **Adjoint ↔ finite differences:** `_substep_forward` reproduces `step`; the scalar
  VJP `run_column_adjoint` matches central differences of `run_column` (to <0.1%)
  for every forcing and parameter target, with sub-steps.
- **Adjoint kernel ↔ reference:** `compute_enthalpy_grad` reproduces the scalar VJP
  in the grid `.grad` buffers — forcing fields column-by-column and the reduced
  parameter `Constant.grad`s as the summed reference — at `n_substeps` 1 and 6; plus
  an end-to-end check that `EnthalpyModel.adjoint` agrees with finite differences of
  the GPU forward.
- **Symmetry:** the enthalpy driver exposes the same surface as the ETIM driver
  (`grid.state.smb`, `forward()`, `adjoint()` populating grid `.grad`), and both
  grids subclass the agnostic base `Grid`.
- **Avalanche:** with an `AvalancheOperator` attached, `forward` redistributes precip
  mass-conservingly per slab (and genuinely moves it), and the adjoint's raw-precip
  gradient matches finite differences of the GPU forward (gradient flows through
  `R^T`); with no operator attached, `precip_eff` equals `precip` exactly.

## Not in v1

- **Burial** of the active reservoir into firn/ice (`M_active_max`) — not applied in
  v1 (matches the reference `step`).
- **Sub-daily temperature structure** — sub-steps are daily-scale; the explicit
  loop resolves day-to-day weather variance (`sigma_t2m`) but not the *diurnal*
  cycle (`daily_amp_t2m`/`phi_0`) that the PDD core integrates over 24 hourly bins,
  and shortwave uses only `insol_mean` (not the `insol_cos`/`insol_sin` harmonics).
- **Spatially-correlated / per-pixel noise** — the deviation vector is domain-wide
  uniform in v1 (see "Sub-monthly stochastic temperature").
- **Thermodynamic-constant gradients** — the adjoint fixes `L_f`, `c_i`, `c_w`,
  `T_transition`, `M_insulation`, `rho_w`; only the seven energy-balance parameters
  and the forcing carry a gradient (see "Differentiability").
- **Diagnostic-output seeds** — the adjoint seeds only the mass/energy outputs; a
  loss on `t_surface`/`albedo` is not differentiated (they feed no state).
