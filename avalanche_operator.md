# Avalanche Redistribution Operator — Design Synopsis

A mass-conserving snow redistribution operator for GLIDE/GLARE. Relocates solid
precipitation off steep, ice-free accumulation terrain to downslope deposition
zones before melt is applied. Target: CUDA C++ raw kernels with a hand-written
adjoint kernel, integrable into the existing adjoint/AD stack.

## Core design principle

The terrain is **static**. Every non-differentiable decision — flow directions,
neighbor selection, routing topology, normalization — is a pure function of the
(fixed) DEM. None of it depends on the state variables being differentiated.

Consequences that drive the whole design:

- The runtime operator is a **fixed linear map** `q = R · s`, where `s` is the
  input solid-precip field and `q` is the redistributed (effective) field.
- Because it is linear and time-invariant, its **adjoint is exactly `Rᵀ`** — no
  tape, no stored forward state, no checkpointing.
- It runs **once per forcing update**, not per timestep. Precompose it with the
  solid-precip climatology so effective accumulation enters GLARE exactly where
  raw accumulation does today.

Differentiation is only ever with respect to `s` and a handful of scalar
deposition parameters `θ`. We never differentiate through graph construction.

## Physical model

Gruber-style mass-conserving **multiple-flow-direction (MFD)** routing (preferred
over single-direction D8/SnowSlide, which produces one-cell-wide runout stripes
and misrepresents open-slope deposition).

Per cell, the mobile snow arriving from upslope is split:

- a fraction `d(slope; θ)` deposits locally,
- the remainder `1 − d` routes to downslope neighbors with weights
  `w_ij ∝ max(0, Δz_ij)^p`, normalized over the cell's downslope set.

Deposition fraction `d` is a **smooth sigmoid in local slope**:

- ~full deposition below a critical slope (≈25–30°),
- ~total scour above a release slope (≈45–55°),
- smooth transition between (needed for a clean adjoint).

Critically, `d` depends on **slope only, not on snow amount**, so the operator is
**linear in `s`**. This is the property that keeps the adjoint stateless. Do not
make `d` depend on local snow load in v1.

### Inferable parameters `θ`

| param | meaning | notes |
|-------|---------|-------|
| `s_crit` | critical (full-deposition) slope | sigmoid center |
| `w_trans` | transition width | sigmoid sharpness |
| `p` | MFD weighting exponent | flow concentration; ~1–2 |

`θ` enters **only** through `d`, so `∂q/∂θ` is analytic. Calibration target: the
coherent negative-`δP`-vs-slope residual signature in existing inversions (the
avalanche signal currently being absorbed by the precipitation-bias field).

## Algorithm

`A` (the routing matrix, strictly triangular on the flow DAG) is **nilpotent**, so
`R = D (I − A)⁻¹` and the Neumann series `q = Σ_k Aᵏ s` terminates exactly at the
longest flow path. We do **not** materialize `(I − A)⁻¹` or run a level-scheduled
topological sweep in v1.

Instead: **fixed `K` gather-stencil passes.** Each routing step multiplies the
mobile fraction by `(1 − d) < 1` along the path, so untruncated mass decays
geometrically. At 100 m posting, `K ≈ 30–50` covers any plausible runout with
negligible residual. Track total residual mobile snow to verify truncation error.

Properties: branch-free, fixed work per cell, no synchronization subtleties,
static iteration count → **slots directly into CUDA-graph capture**.

The exact level-scheduled sweep is a later optimization, only if `K` passes ever
show up in a profile (they shouldn't — operator runs once per forcing update).

## CUDA implementation

**Structured grid, implicit adjacency.** Do *not* pass CSR neighbor lists. On a
structured grid the adjacency is implicit in indexing; in-kernel recomputation of
flow directions from a read-only 3×3 DEM patch (shared-mem / L2 served) is far
cheaper than the global-memory traffic of CSR indices+weights. These kernels are
bandwidth-bound; arithmetic is effectively free.

**Gather, not scatter.** Scatter-with-atomics is non-deterministic in summation
order → bitwise non-reproducibility, which is poison for FP32 adjoint dot tests
(can't distinguish a bug at 1e-5 from atomic-ordering noise). Use deterministic
gather in both directions.

### Forward / adjoint asymmetry

The MFD weight `A_ij` = fraction of cell `j`'s outflow allocated to neighbor `i`,
normalized over `j`'s downslope set. This makes the two directions asymmetric:

- **Forward scatter** (at cell `i`): Each cell atomicAdds mass transfer to its 
  downstream neighbors
- **Adjoint gather** (at cell `j`): pulls `λ` from `j`'s own downslope neighbors
  weighted by `j`'s own outflow fractions → pure 3×3, the easy direction.


## Order of operations in GLARE

Apply redistribution **before** melt, so relocated snow ablates at its
*destination's* PDD/radiation — this is what delivers the avalanche subsidy to the
glacier tongues. Effective accumulation `q = R · P_solid` replaces raw
accumulation at its current entry point.  This should probably be exposed as a
separate module that acts on the precipitation field.

## Test suite

- **Mass conservation:** column sums of `R` equal `1 − boundary_export`, to
  machine precision.
- **Adjoint dot-product test:** `⟨R s, y⟩ == ⟨s, Rᵀ y⟩` (the standard check you
  already run; deterministic gather keeps it sharp in FP32).
- **Runout footprints:** sanity-check against mapped avalanche paths in a couple
  of well-known basins.
- **Truncation:** residual mobile snow after `K` passes below tolerance.
