"""Tests for the snowpack enthalpy model (scalar reference + CUDA kernel).

Run with:  python -m pytest tests/test_enthalpy.py
Requires a CUDA device (cupy); the whole module is skipped otherwise.

The enthalpy core now lives on its own EnthalpyGrid: forcing and parameters are grid
fields/Constants, output state is grid.state, and the adjoint writes gradients into
the grid's Field.grad / Constant.grad buffers (symmetric with the ETIM core).  The
numpy oracle (step/run_column) reads its scalars from the grid via scalar_params.
"""
import copy
import numpy as np

try:
    import pytest
except ImportError:                       # allow running without pytest installed
    pytest = None

import cupy as cp
assert cp.cuda.runtime.getDeviceCount() >= 1, "no CUDA device"

from glare import (EnthalpyGrid, EnthalpyModel, scalar_params, State, step,
                   run_column, run_column_adjoint, generate_temp_deviations,
                   GRAD_PARAM_NAMES, AvalancheOperator)
from glare.enthalpy import _substep_forward, SECONDS_PER_YEAR


def _close(got, ref, rtol=1e-3, atol=0.0):
    """Max-abs agreement scaled by the reference magnitude."""
    got = np.asarray(got, dtype=float)
    ref = np.asarray(ref, dtype=float)
    tol = rtol * (1.0 + np.max(np.abs(ref))) + atol
    return float(np.max(np.abs(got - ref))) <= tol


# --------------------------------------------------------------------------- #
# Helpers: locate a parameter Constant by name on the grid, override it, and build
# the scalar-parameter view the numpy oracle consumes.  The grid Constants are the
# single source of truth (there is no Parameters dataclass).
# --------------------------------------------------------------------------- #
_PARAM_GROUPS = ("thermodynamics", "temperature", "radiation")


def _param_const(grid, name):
    for grp_name in _PARAM_GROUPS:
        grp = getattr(grid, grp_name)
        if hasattr(grp, name):
            return getattr(grp, name)
    raise KeyError(name)


def _set_params(grid, **overrides):
    for name, value in overrides.items():
        _param_const(grid, name).set(value)


def default_params(**overrides):
    """A scalar-parameter view off a throwaway EnthalpyGrid (for the numpy oracle)."""
    g = EnthalpyGrid(ny=1, nx=1, nt=12, dx=100.0, dt=1.0 / 12)
    _set_params(g, **overrides)
    return scalar_params(g)


def _grid(ny, nx, nt, dx, **kw):
    return EnthalpyGrid(ny=ny, nx=nx, nt=nt, dx=dx, dt=1.0 / nt, **kw)


# --------------------------------------------------------------------------- #
# The packaged scalar reference reproduces the toy enthalpy_sketch_revised.py
# to the digits it prints (guards against drift between sketch and package).
# --------------------------------------------------------------------------- #
def test_reference_matches_sketch():
    # Pin the sketch's parameters explicitly so this stays a sketch-vs-package
    # algorithm check even as the packaged Constant defaults are tuned
    # (enthalpy_sketch_revised.py uses M_albedo = 50).
    p = default_params(M_albedo=50.0)
    state = State()

    n_per_year = 12
    t = np.linspace(0.0, 3.0, 3 * n_per_year + 1)[:-1]
    dt = 1.0 / n_per_year
    P = np.full_like(t, 1000.0)          # kg m-2 yr-1 (SI, as the sketch drives step)
    T = 10.0 * np.cos(2.0 * np.pi * t)
    T_base = -10.0

    total_precip = total_runoff = total_ice_melt = 0.0
    for Pi, Ti in zip(P, T):
        state, flux = step(state, Pi, Ti, T_base, dt, p)
        total_precip += flux.precipitation
        total_runoff += flux.runoff_from_pack
        total_ice_melt += flux.ice_melt

    # Values printed by enthalpy_sketch_revised.py.
    assert abs(state.M - 8.652) <= 1e-2
    assert abs(state.E) <= 1e-3
    assert abs(total_runoff - 2991.348) <= 1e-1
    assert abs(total_ice_melt - 4929.005) <= 1e-1

    residual = total_precip - total_runoff - state.M
    assert abs(residual) <= 1e-6


# --------------------------------------------------------------------------- #
# The CUDA kernel reproduces the scalar reference column-by-column.  fast-math
# is disabled so the only gap is FP32 rounding over the 12-month recurrence.
# --------------------------------------------------------------------------- #
def test_kernel_matches_reference():
    rng = np.random.default_rng(0)
    nt, ny, nx, dx = 12, 6, 5, 100.0
    start_month = 9

    g = _grid(ny, nx, nt, dx, use_fast_math=False)
    precip = rng.uniform(0.0, 2.0, (nt, ny, nx)).astype(np.float32)
    t2m = rng.uniform(-15.0, 8.0, (nt, ny, nx)).astype(np.float32)
    insol = rng.uniform(0.0, 0.6, (nt, ny, nx)).astype(np.float32)
    t_base = rng.uniform(-12.0, -2.0, (ny, nx)).astype(np.float32)
    g.precipitation.precip.set(cp.asarray(precip))
    g.temperature.t2m.set(cp.asarray(t2m))
    g.radiation.insol_mean.set(cp.asarray(insol))
    # Exercise the seasonal shortwave path: nonzero q_sw_insol driven by insol.
    g.radiation.q_sw_insol.set(200.0 * SECONDS_PER_YEAR)
    g.geometry.t_base.set(cp.asarray(t_base))

    enth = EnthalpyModel(g)
    enth.forward()
    p = scalar_params(g)

    fields = {k: cp.asnumpy(getattr(g.state, k).data) for k in
              ("smb", "M", "E", "runoff", "ice_melt", "t_surface", "albedo")}

    for i in range(ny):
        for j in range(nx):
            ref = run_column(precip[:, i, j], t2m[:, i, j], float(t_base[i, j]),
                             dt=1 / nt, p=p, start_month=start_month,
                             insol=insol[:, i, j])
            assert _close(fields["smb"][:, i, j], ref["smb"])
            assert _close(fields["M"][:, i, j], ref["M"])
            assert _close(fields["E"][:, i, j], ref["E"])
            assert _close(fields["runoff"][:, i, j], ref["runoff"])
            assert _close(fields["ice_melt"][:, i, j], ref["ice_melt"])
            assert _close(fields["t_surface"][:, i, j], ref["t_surface"], atol=1e-3)
            assert _close(fields["albedo"][:, i, j], ref["albedo"], atol=1e-4)


# --------------------------------------------------------------------------- #
# Debris attenuation: the static geometry.debris field multiplies every glacier-
# ice melt flux (melting-branch residual and bare-branch melt), leaving the pack
# (M, E, runoff) untouched -- so ice melt scales exactly linearly in debris.
# Checked on the scalar reference and column-by-column against the kernel.
# --------------------------------------------------------------------------- #
def test_debris_attenuates_ice_melt():
    rng = np.random.default_rng(4)
    nt, ny, nx, dx = 12, 6, 5, 100.0
    dt = 1.0 / nt
    p = default_params(q_sw_insol=200.0 * SECONDS_PER_YEAR,
                       H_atm=5.0 * SECONDS_PER_YEAR)
    precip1 = rng.uniform(0.0, 2.0, nt)
    t2m1 = rng.uniform(-15.0, 8.0, nt)
    insol1 = rng.uniform(0.0, 0.6, nt)

    clean = run_column(precip1, t2m1, -5.0, dt, p=p, insol=insol1, debris=1.0)
    half = run_column(precip1, t2m1, -5.0, dt, p=p, insol=insol1, debris=0.5)
    full = run_column(precip1, t2m1, -5.0, dt, p=p, insol=insol1, debris=0.0)
    assert clean["ice_melt"].sum() > 0.0          # regime actually melts ice
    assert np.allclose(half["ice_melt"], 0.5 * clean["ice_melt"], rtol=1e-12)
    assert np.allclose(full["ice_melt"], 0.0)
    for k in ("M", "E", "runoff"):                # pack untouched by debris
        assert np.array_equal(clean[k], full[k])

    # Kernel vs reference with a per-pixel debris field (includes 0 and 1).
    g = _grid(ny, nx, nt, dx, use_fast_math=False)
    precip = rng.uniform(0.0, 2.0, (nt, ny, nx)).astype(np.float32)
    t2m = rng.uniform(-15.0, 8.0, (nt, ny, nx)).astype(np.float32)
    insol = rng.uniform(0.0, 0.6, (nt, ny, nx)).astype(np.float32)
    t_base = rng.uniform(-12.0, -2.0, (ny, nx)).astype(np.float32)
    debris = rng.uniform(0.0, 1.0, (ny, nx)).astype(np.float32)
    debris[0, 0] = 0.0
    debris[0, 1] = 1.0
    g.precipitation.precip.set(cp.asarray(precip))
    g.temperature.t2m.set(cp.asarray(t2m))
    g.radiation.insol_mean.set(cp.asarray(insol))
    g.radiation.q_sw_insol.set(200.0 * SECONDS_PER_YEAR)
    g.geometry.t_base.set(cp.asarray(t_base))
    g.geometry.debris.set(cp.asarray(debris))

    enth = EnthalpyModel(g)
    enth.forward()
    pk = scalar_params(g)
    fields = {k: cp.asnumpy(getattr(g.state, k).data) for k in
              ("smb", "M", "E", "runoff", "ice_melt")}
    for i in range(ny):
        for j in range(nx):
            ref = run_column(precip[:, i, j], t2m[:, i, j], float(t_base[i, j]),
                             dt=dt, p=pk, start_month=9, insol=insol[:, i, j],
                             debris=float(debris[i, j]))
            for k in fields:
                # E needs an absolute floor: FP32 rounding of the O(1e6) energy
                # intermediates dwarfs a near-zero monthly cold content.
                atol = 10.0 if k == "E" else 0.0
                assert _close(fields[k][:, i, j], ref[k], atol=atol), (k, i, j)


# --------------------------------------------------------------------------- #
# Water-mass conservation of the active reservoir: over the water year the total
# precipitation mass leaves only as pack runoff or is retained in M (no burial in
# v1; ice melt is a separate glacier source and is excluded).
# --------------------------------------------------------------------------- #
def test_mass_conservation():
    rng = np.random.default_rng(1)
    nt, ny, nx, dx = 12, 8, 8, 100.0
    start_month = 9

    g = _grid(ny, nx, nt, dx)
    precip = rng.uniform(0.0, 2.0, (nt, ny, nx)).astype(np.float32)
    t2m = rng.uniform(-15.0, 10.0, (nt, ny, nx)).astype(np.float32)
    g.precipitation.precip.set(cp.asarray(precip))
    g.temperature.t2m.set(cp.asarray(t2m))
    g.geometry.t_base.set(-10.0)

    EnthalpyModel(g).forward()
    p = scalar_params(g)

    dt = 1.0 / nt
    last_month = (start_month + nt - 1) % nt
    M = cp.asnumpy(g.state.M.data)
    runoff = cp.asnumpy(g.state.runoff.data)

    in_mass = np.clip(precip, 0.0, None).sum(axis=0) * p.rho_w * dt
    out_mass = runoff.sum(axis=0) + M[last_month]
    resid = np.abs(in_mass - out_mass)
    assert float(np.max(resid)) <= 1e-3 * (1.0 + float(np.max(in_mass)))


# --------------------------------------------------------------------------- #
# SMB identity: the water-year sum of the monthly SMB rate (x dt) equals the net
# annual balance -- surviving snow (M_final) minus total ice ablation, in m w.e.
# --------------------------------------------------------------------------- #
def test_smb_annual_identity():
    rng = np.random.default_rng(5)
    nt, ny, nx, dx = 12, 8, 7, 100.0
    start_month = 9

    g = _grid(ny, nx, nt, dx)
    precip = rng.uniform(0.0, 2.0, (nt, ny, nx)).astype(np.float32)
    t2m = rng.uniform(-15.0, 8.0, (nt, ny, nx)).astype(np.float32)
    insol = rng.uniform(0.0, 0.6, (nt, ny, nx)).astype(np.float32)
    g.precipitation.precip.set(cp.asarray(precip))
    g.temperature.t2m.set(cp.asarray(t2m))
    g.radiation.insol_mean.set(cp.asarray(insol))
    g.radiation.q_sw_insol.set(250.0 * SECONDS_PER_YEAR)
    g.geometry.t_base.set(-8.0)

    EnthalpyModel(g).forward()
    p = scalar_params(g)

    dt = 1.0 / nt
    last_month = (start_month + nt - 1) % nt
    smb = cp.asnumpy(g.state.smb.data)
    M = cp.asnumpy(g.state.M.data)
    ice_melt = cp.asnumpy(g.state.ice_melt.data)

    annual_smb = (smb * dt).sum(axis=0)                       # m w.e.
    identity = (M[last_month] - ice_melt.sum(axis=0)) / p.rho_w
    assert _close(annual_smb, identity, atol=1e-5)


# --------------------------------------------------------------------------- #
# Cold, non-glacier column: everything falls as snow, nothing melts or runs off,
# and the reservoir equals the accumulated precipitation mass.
# --------------------------------------------------------------------------- #
def test_cold_accumulation_no_melt():
    nt, ny, nx, dx = 12, 4, 4, 100.0
    start_month = 9
    g = _grid(ny, nx, nt, dx, glacier_surface=False)
    g.precipitation.precip.set(1.0)      # 1 m a-1 w.e. everywhere
    g.temperature.t2m.set(-20.0)
    g.geometry.t_base.set(-10.0)

    EnthalpyModel(g).forward()
    p = scalar_params(g)

    runoff = cp.asnumpy(g.state.runoff.data)
    ice_melt = cp.asnumpy(g.state.ice_melt.data)
    M = cp.asnumpy(g.state.M.data)

    assert float(np.max(np.abs(runoff))) == 0.0
    assert float(np.max(np.abs(ice_melt))) == 0.0

    dt = 1.0 / nt
    last_month = (start_month + nt - 1) % nt
    expected = 1.0 * p.rho_w * dt * nt        # kg m-2 accumulated over the year
    assert _close(M[last_month], np.full((ny, nx), expected))


# --------------------------------------------------------------------------- #
# Sub-monthly inner loop (n_sub > 1): the kernel still matches the reference
# column-by-column when driven by a fixed daily-deviation vector.
# --------------------------------------------------------------------------- #
def test_kernel_matches_reference_substepped():
    rng = np.random.default_rng(7)
    nt, ny, nx, dx = 12, 6, 5, 100.0
    start_month, n_sub = 9, 6

    g = _grid(ny, nx, nt, dx, n_substeps=n_sub, use_fast_math=False)
    precip = rng.uniform(0.0, 2.0, (nt, ny, nx)).astype(np.float32)
    t2m = rng.uniform(-15.0, 8.0, (nt, ny, nx)).astype(np.float32)
    insol = rng.uniform(0.0, 0.6, (nt, ny, nx)).astype(np.float32)
    t_base = rng.uniform(-12.0, -2.0, (ny, nx)).astype(np.float32)
    g.precipitation.precip.set(cp.asarray(precip))
    g.temperature.t2m.set(cp.asarray(t2m))
    g.radiation.insol_mean.set(cp.asarray(insol))
    g.radiation.q_sw_insol.set(200.0 * SECONDS_PER_YEAR)
    g.geometry.t_base.set(cp.asarray(t_base))

    # Fixed (nt, n_sub) deviation vector shared by every pixel.
    dev = generate_temp_deviations(nt, n_sub, sigma=4.0, rng=rng)

    EnthalpyModel(g).forward(temp_deviations=dev)
    p = scalar_params(g)

    fields = {k: cp.asnumpy(getattr(g.state, k).data) for k in
              ("smb", "M", "E", "runoff", "ice_melt", "t_surface", "albedo")}

    for i in range(ny):
        for j in range(nx):
            ref = run_column(precip[:, i, j], t2m[:, i, j], float(t_base[i, j]),
                             dt=1 / nt, p=p, start_month=start_month,
                             insol=insol[:, i, j], temp_dev=dev)
            assert _close(fields["smb"][:, i, j], ref["smb"])
            assert _close(fields["M"][:, i, j], ref["M"])
            assert _close(fields["E"][:, i, j], ref["E"])
            assert _close(fields["runoff"][:, i, j], ref["runoff"])
            assert _close(fields["ice_melt"][:, i, j], ref["ice_melt"])
            assert _close(fields["t_surface"][:, i, j], ref["t_surface"], atol=1e-3)
            assert _close(fields["albedo"][:, i, j], ref["albedo"], atol=1e-4)


# --------------------------------------------------------------------------- #
# Physics: sub-monthly temperature variance increases melt for a near-freezing
# column (the stateful analog of the PDD Phi/phi variance effect).  Comparing at
# fixed n_sub isolates the variance from the temporal-refinement effect; demeaned
# deviations keep the monthly-mean forcing identical.
# --------------------------------------------------------------------------- #
def test_variance_increases_melt():
    nt, ny, nx, dx = 12, 4, 4, 100.0
    n_sub = 30

    g = _grid(ny, nx, nt, dx, n_substeps=n_sub)
    g.precipitation.precip.set(1.0)      # steady accumulation
    g.temperature.t2m.set(-1.0)          # near-freezing monthly mean
    g.geometry.t_base.set(-1.0)
    enth = EnthalpyModel(g)

    # Baseline: same n_sub, zero deviations (isolates the variance effect).
    enth.forward(temp_deviations=np.zeros((nt, n_sub), dtype=np.float32))
    ablation_flat = float(cp.asnumpy(g.state.runoff.data + g.state.ice_melt.data).sum())

    # Variance: demeaned Gaussian daily deviations, monthly mean preserved.
    dev = generate_temp_deviations(nt, n_sub, sigma=6.0,
                                   rng=np.random.default_rng(0))
    enth.forward(temp_deviations=dev)
    ablation_var = float(cp.asnumpy(g.state.runoff.data + g.state.ice_melt.data).sum())

    assert ablation_var > 1.05 * ablation_flat   # variance unlocks threshold melt


# --------------------------------------------------------------------------- #
# Sub-monthly mass conservation: the per-sub-step precip still sums to the monthly
# input, so total precip mass leaves only as runoff or is retained in M.
# --------------------------------------------------------------------------- #
def test_mass_conservation_substepped():
    rng = np.random.default_rng(8)
    nt, ny, nx, dx = 12, 8, 8, 100.0
    start_month, n_sub = 9, 8

    g = _grid(ny, nx, nt, dx, n_substeps=n_sub)
    precip = rng.uniform(0.0, 2.0, (nt, ny, nx)).astype(np.float32)
    t2m = rng.uniform(-15.0, 10.0, (nt, ny, nx)).astype(np.float32)
    g.precipitation.precip.set(cp.asarray(precip))
    g.temperature.t2m.set(cp.asarray(t2m))
    g.geometry.t_base.set(-10.0)
    dev = generate_temp_deviations(nt, n_sub, sigma=5.0, rng=rng)
    EnthalpyModel(g).forward(temp_deviations=dev)
    p = scalar_params(g)

    dt = 1.0 / nt
    last_month = (start_month + nt - 1) % nt
    M = cp.asnumpy(g.state.M.data)
    runoff = cp.asnumpy(g.state.runoff.data)

    in_mass = np.clip(precip, 0.0, None).sum(axis=0) * p.rho_w * dt
    out_mass = runoff.sum(axis=0) + M[last_month]
    resid = np.abs(in_mass - out_mass)
    assert float(np.max(resid)) <= 1e-3 * (1.0 + float(np.max(in_mass)))


# --------------------------------------------------------------------------- #
# Adjoint: the intermediate-capturing _substep_forward must reproduce step()
# exactly (it is the forward the backward pass linearises about).
# --------------------------------------------------------------------------- #
def test_substep_matches_step():
    rng = np.random.default_rng(2)
    dt = 1.0 / 12
    p = default_params(q_sw_insol=200.0 * SECONDS_PER_YEAR)
    worst = 0.0
    for _ in range(3000):
        M0 = rng.uniform(0.0, 300.0)
        E0 = -rng.uniform(0.0, 1.0e7)
        P = rng.uniform(0.0, 2000.0)
        T = rng.uniform(-20.0, 12.0)
        Tb = rng.uniform(-15.0, 0.0)
        ins = rng.uniform(0.0, 0.6)
        glac = bool(rng.integers(0, 2))
        st, fl = step(State(M0, E0), P, T, Tb, dt, p,
                      glacier_surface=glac, insolation=ins)
        M2, E2, ro, ic, _ = _substep_forward(M0, E0, P, T, Tb, ins, dt, p, glac)
        worst = max(worst, abs(M2 - st.M), abs(E2 - st.E),
                    abs(ro - fl.runoff_from_pack), abs(ic - fl.ice_melt))
    assert worst < 1e-6


# --------------------------------------------------------------------------- #
# Adjoint correctness: the hand-derived scalar VJP (run_column_adjoint) matches
# central finite differences of run_column for every differentiation target --
# forcing (t2m, precip, insol, t_base) and all seven parameters -- with sub-steps.
# This is the ground-truth check; the kernel is then pinned to this reference.
# --------------------------------------------------------------------------- #
def test_run_column_adjoint_finite_difference():
    nt = 12
    dt = 1.0 / nt
    start_month = 9

    def make_case(seed):
        r = np.random.default_rng(seed)
        precip = r.uniform(0.0, 2.0, nt)
        t2m = r.uniform(-15.0, 8.0, nt)
        insol = r.uniform(0.0, 0.6, nt)
        t_base = float(r.uniform(-12.0, -2.0))
        n_sub = int(r.integers(1, 5))
        dev = r.normal(0.0, 4.0, (nt, n_sub))
        dev -= dev.mean(axis=1, keepdims=True)
        p = default_params(q_sw_insol=float(r.uniform(100.0, 400.0)) * SECONDS_PER_YEAR,
                           H_atm=float(r.uniform(0.5, 10.0)) * SECONDS_PER_YEAR)
        seeds = {k: r.normal(0.0, 1.0, nt) for k in
                 ("grad_smb", "grad_M", "grad_E", "grad_runoff", "grad_ice_melt")}
        seeds["grad_M"] *= 1e-2      # keep the loss well-conditioned across outputs
        seeds["grad_E"] *= 1e-8      # (M ~ 1e2 kg m-2, E ~ 1e7 J m-2)
        debris = float(r.uniform(0.1, 1.0))  # drawn last: earlier draws unchanged
        return precip, t2m, insol, t_base, debris, dev, p, seeds

    def loss(precip, t2m, insol, t_base, debris, dev, p, seeds):
        out = run_column(precip, t2m, t_base, dt, p=p, start_month=start_month,
                         insol=insol, temp_dev=dev, debris=debris)
        L = 0.0
        for name, key in (("smb", "grad_smb"), ("M", "grad_M"), ("E", "grad_E"),
                          ("runoff", "grad_runoff"), ("ice_melt", "grad_ice_melt")):
            L += float(np.sum(seeds[key] * out[name]))
        return L

    def rel(a, b):
        return abs(a - b) / (1e-30 + abs(a) + abs(b))

    worst = 0.0
    for case in range(5):
        precip, t2m, insol, t_base, debris, dev, p, seeds = make_case(case)
        grad = run_column_adjoint(precip, t2m, t_base, dt, p=p,
                                  start_month=start_month, insol=insol,
                                  temp_dev=dev, debris=debris, **seeds)

        def L(pre=precip, tt=t2m, ins=insol, tb=t_base, deb=debris, pp=p):
            return loss(pre, tt, ins, tb, deb, dev, pp, seeds)

        for mm in range(nt):
            h = 1e-5
            tp = t2m.copy(); tp[mm] += h
            tm = t2m.copy(); tm[mm] -= h
            worst = max(worst, rel(grad["t2m"][mm], (L(tt=tp) - L(tt=tm)) / (2 * h)))
            pp = precip.copy(); pp[mm] += h
            pm = precip.copy(); pm[mm] -= h
            worst = max(worst, rel(grad["precip"][mm], (L(pre=pp) - L(pre=pm)) / (2 * h)))
            ip = insol.copy(); ip[mm] += h
            im_ = insol.copy(); im_[mm] -= h
            worst = max(worst, rel(grad["insol"][mm], (L(ins=ip) - L(ins=im_)) / (2 * h)))
        h = 1e-4
        worst = max(worst, rel(grad["t_base"],
                               (L(tb=t_base + h) - L(tb=t_base - h)) / (2 * h)))
        h = 1e-5
        worst = max(worst, rel(grad["debris"],
                               (L(deb=debris + h) - L(deb=debris - h)) / (2 * h)))
        for k in GRAD_PARAM_NAMES:
            base = getattr(p, k)
            h = 1e-4 * (abs(base) + 1.0)
            pp = copy.copy(p); setattr(pp, k, base + h)
            pm = copy.copy(p); setattr(pm, k, base - h)
            worst = max(worst, rel(grad[k], (L(pp=pp) - L(pp=pm)) / (2 * h)))

    assert worst < 1e-3      # analytic VJP agrees with FD to well under 0.1%


# --------------------------------------------------------------------------- #
# The CUDA adjoint kernel reproduces the scalar reference column-by-column: the
# forcing-gradient fields per pixel, and the reduced parameter gradients as the
# sum of the reference's per-column parameter gradients.  fast-math off.  The GPU
# gradients land in the grid .grad buffers.
# --------------------------------------------------------------------------- #
def _kernel_vs_reference_adjoint(n_sub, seed):
    rng = np.random.default_rng(seed)
    nt, ny, nx, dx = 12, 6, 5, 100.0
    start_month = 9
    dt = 1.0 / nt

    g = _grid(ny, nx, nt, dx, n_substeps=n_sub, use_fast_math=False)
    precip = rng.uniform(0.0, 2.0, (nt, ny, nx)).astype(np.float32)
    t2m = rng.uniform(-15.0, 8.0, (nt, ny, nx)).astype(np.float32)
    insol = rng.uniform(0.0, 0.6, (nt, ny, nx)).astype(np.float32)
    t_base = rng.uniform(-12.0, -2.0, (ny, nx)).astype(np.float32)
    debris = rng.uniform(0.0, 1.0, (ny, nx)).astype(np.float32)
    g.precipitation.precip.set(cp.asarray(precip))
    g.temperature.t2m.set(cp.asarray(t2m))
    g.radiation.insol_mean.set(cp.asarray(insol))
    g.radiation.q_sw_insol.set(200.0 * SECONDS_PER_YEAR)
    g.geometry.t_base.set(cp.asarray(t_base))
    g.geometry.debris.set(cp.asarray(debris))

    dev = generate_temp_deviations(nt, n_sub, sigma=4.0, rng=rng) if n_sub > 1 else None

    enth = EnthalpyModel(g)
    enth.forward(temp_deviations=dev)
    p = scalar_params(g)

    seeds = {name: rng.normal(0.0, 1.0, (nt, ny, nx)).astype(np.float32)
             for name in ("smb", "M", "E", "runoff", "ice_melt")}
    seeds["M"] *= 1e-2
    seeds["E"] *= 1e-8
    enth.adjoint(**{f"grad_{k}": cp.asarray(v) for k, v in seeds.items()})

    g_t2m = cp.asnumpy(g.temperature.t2m.grad)
    g_precip = cp.asnumpy(g.precipitation.precip.grad)
    g_insol = cp.asnumpy(g.radiation.insol_mean.grad)
    g_t_base = cp.asnumpy(g.geometry.t_base.grad)
    g_debris = cp.asnumpy(g.geometry.debris.grad)

    ref_t2m = np.zeros((nt, ny, nx))
    ref_precip = np.zeros((nt, ny, nx))
    ref_insol = np.zeros((nt, ny, nx))
    ref_t_base = np.zeros((ny, nx))
    ref_debris = np.zeros((ny, nx))
    ref_params = {k: 0.0 for k in GRAD_PARAM_NAMES}
    dev_col = np.zeros((nt, 1)) if dev is None else dev
    for i in range(ny):
        for j in range(nx):
            gr = run_column_adjoint(
                precip[:, i, j], t2m[:, i, j], float(t_base[i, j]), dt, p=p,
                start_month=start_month, insol=insol[:, i, j], temp_dev=dev_col,
                debris=float(debris[i, j]),
                grad_smb=seeds["smb"][:, i, j], grad_M=seeds["M"][:, i, j],
                grad_E=seeds["E"][:, i, j], grad_runoff=seeds["runoff"][:, i, j],
                grad_ice_melt=seeds["ice_melt"][:, i, j])
            ref_t2m[:, i, j] = gr["t2m"]
            ref_precip[:, i, j] = gr["precip"]
            ref_insol[:, i, j] = gr["insol"]
            ref_t_base[i, j] = gr["t_base"]
            ref_debris[i, j] = gr["debris"]
            for k in GRAD_PARAM_NAMES:
                ref_params[k] += gr[k]

    assert _close(g_t2m, ref_t2m)
    assert _close(g_precip, ref_precip)
    assert _close(g_insol, ref_insol)
    assert _close(g_t_base, ref_t_base)
    assert _close(g_debris, ref_debris)
    for k in GRAD_PARAM_NAMES:
        got, ref = _param_const(g, k).grad, ref_params[k]
        assert abs(got - ref) <= 1e-3 * (1.0 + abs(ref)), k


def test_kernel_adjoint_matches_reference():
    _kernel_vs_reference_adjoint(n_sub=1, seed=11)


def test_kernel_adjoint_matches_reference_substepped():
    _kernel_vs_reference_adjoint(n_sub=6, seed=12)


# --------------------------------------------------------------------------- #
# End-to-end gradient check: the GPU adjoint agrees with central differences of
# the GPU forward for a loss on smb -- guards the seed/reduction wiring in
# EnthalpyModel.adjoint (not just the internal kernel algebra).  Gradients are read
# off the grid .grad buffers the adjoint populated.
# --------------------------------------------------------------------------- #
def test_adjoint_gradient_check_end_to_end():
    rng = np.random.default_rng(21)
    nt, ny, nx, dx = 12, 4, 3, 100.0

    g = _grid(ny, nx, nt, dx, use_fast_math=False)
    precip = rng.uniform(0.2, 2.0, (nt, ny, nx)).astype(np.float32)
    t2m = rng.uniform(-8.0, 4.0, (nt, ny, nx)).astype(np.float32)
    insol = rng.uniform(0.1, 0.6, (nt, ny, nx)).astype(np.float32)
    t_base = rng.uniform(-10.0, -2.0, (ny, nx)).astype(np.float32)
    g.precipitation.precip.set(cp.asarray(precip))
    g.temperature.t2m.set(cp.asarray(t2m))
    g.radiation.insol_mean.set(cp.asarray(insol))
    g.radiation.q_sw_insol.set(200.0 * SECONDS_PER_YEAR)
    g.geometry.t_base.set(cp.asarray(t_base))
    enth = EnthalpyModel(g)

    # Loss = sum of smb weighted by a fixed random field; seed is d loss / d smb = w.
    w = rng.normal(0.0, 1.0, (nt, ny, nx)).astype(np.float32)

    def loss():
        enth.forward()
        return float(cp.sum(cp.asarray(w) * g.state.smb.data))

    enth.forward()
    enth.adjoint(grad_smb=cp.asarray(w))
    g_t2m = cp.asnumpy(g.temperature.t2m.grad)
    g_H_atm = g.thermodynamics.H_atm.grad

    # Central difference w.r.t. a handful of t2m cells.
    base_t2m = g.temperature.t2m.data.copy()
    for (mm, i, j) in [(0, 0, 0), (3, 1, 2), (6, 2, 1), (9, 3, 0)]:
        h = 1e-2
        pert = base_t2m.copy(); pert[mm, i, j] += h
        g.temperature.t2m.set(pert); Lp = loss()
        pert = base_t2m.copy(); pert[mm, i, j] -= h
        g.temperature.t2m.set(pert); Lm = loss()
        num = (Lp - Lm) / (2 * h)
        assert abs(g_t2m[mm, i, j] - num) <= 1e-2 * (1.0 + abs(num))
    g.temperature.t2m.set(base_t2m)

    # Parameter gradient (scalar, reduced over the grid), perturbing the Constant.
    base_H = float(g.thermodynamics.H_atm.value)
    h = 1e-2 * base_H
    g.thermodynamics.H_atm.set(base_H + h); Lp = loss()
    g.thermodynamics.H_atm.set(base_H - h); Lm = loss()
    g.thermodynamics.H_atm.set(base_H)
    num = (Lp - Lm) / (2 * h)
    assert abs(g_H_atm - num) <= 2e-2 * (1.0 + abs(num))


# --------------------------------------------------------------------------- #
# Symmetry: the enthalpy driver exposes the same surface as the ETIM driver --
# grid.state.smb output, forward(), and adjoint() populating grid .grad buffers.
# --------------------------------------------------------------------------- #
def test_enthalpy_matches_etim_surface():
    from glare import TIMGrid, ImprovedTemperatureIndex, Grid

    nt, ny, nx, dx = 12, 5, 4, 100.0
    eg = _grid(ny, nx, nt, dx)
    tg = TIMGrid(ny=ny, nx=nx, nt=nt, dx=dx, dt=1.0 / nt)

    # Both grids subclass the agnostic base and expose grid.state.smb.
    assert isinstance(eg, Grid) and isinstance(tg, Grid)
    assert hasattr(eg.state, "smb") and hasattr(tg.state, "smb")

    em = EnthalpyModel(eg)
    tm = ImprovedTemperatureIndex(grid=tg)
    for model in (em, tm):
        assert hasattr(model, "forward") and hasattr(model, "adjoint")

    # Enthalpy adjoint writes gradients into the grid buffers (returns None).
    eg.temperature.t2m.set(-3.0)
    eg.precipitation.precip.set(1.0)
    eg.geometry.t_base.set(-5.0)
    em.forward()
    out = em.adjoint(grad_smb=cp.ones((nt, ny, nx), dtype=cp.float32))
    assert out is None
    assert eg.temperature.t2m.grad.shape == (nt, ny, nx)
    # Parameter gradient lands on the Constant as a scalar (cp.float32, per the
    # Constant.grad setter -- the same convention as ETIM's mf/rf grads).
    assert eg.thermodynamics.H_atm.has_grad()
    assert np.isfinite(float(eg.thermodynamics.H_atm.grad))


# --------------------------------------------------------------------------- #
# Avalanche redistribution: attaching an AvalancheOperator relocates the raw precip
# downslope before the recurrence (mass-conserving per slab), and the adjoint pulls
# the precip gradient back through R^T -- verified against finite differences of the
# GPU forward.  No snow/rain partition: R acts on the raw total precip.
# --------------------------------------------------------------------------- #
def _ramp_dem(ny, nx, drop_per_row=60.0, bench_row=6):
    z = np.zeros((ny, nx), dtype=np.float32)
    for i in range(ny):
        z[i, :] = (bench_row - i) * drop_per_row if i < bench_row else 0.0
    return cp.asarray(z)


def test_avalanche_redistribution_and_adjoint():
    rng = np.random.default_rng(3)
    nt, ny, nx, dx = 12, 16, 14, 100.0

    enth = EnthalpyModel(ny=ny, nx=nx, nt=nt, dx=dx, dt=1 / nt, use_fast_math=False,
                         seed=0)
    g = enth.grid
    precip = rng.uniform(0.2, 2.0, (nt, ny, nx)).astype(np.float32)
    g.precipitation.precip.set(cp.asarray(precip))
    # Cold, accumulation-dominated column: precip flows linearly into snow mass and
    # thus smb, so the end-to-end finite-difference adjoint check is smooth (away from
    # the melt/regime kinks -- the melt-path adjoint is covered separately, without
    # avalanche, by test_adjoint_gradient_check_end_to_end).
    g.temperature.t2m.set(-18.0)
    g.geometry.t_base.set(-15.0)
    g.geometry.srf.set(_ramp_dem(ny, nx))
    enth.avalanche = AvalancheOperator(g, s_crit=30.0, w_trans=8.0, p=1.5, K=40)

    enth.forward()
    raw = cp.asnumpy(g.precipitation.precip.data)
    eff = cp.asnumpy(g.precipitation.precip_eff.data)

    # Per-slab mass is conserved by R, but the field is genuinely redistributed.
    slab_raw = raw.sum(axis=(1, 2))
    slab_eff = eff.sum(axis=(1, 2))
    assert float(np.max(np.abs(slab_eff - slab_raw) / slab_raw)) <= 1e-4
    assert float(np.max(np.abs(eff - raw))) > 1e-2

    # Adjoint pulls the precip gradient back through R^T: check a scalar smb-loss
    # gradient w.r.t. a few raw-precip cells against central differences.
    w = rng.normal(0.0, 1.0, (nt, ny, nx)).astype(np.float32)
    enth.forward()
    enth.adjoint(grad_smb=cp.asarray(w))
    gp = cp.asnumpy(g.precipitation.precip.grad)

    def loss(P):
        g.precipitation.precip.set(cp.asarray(P))
        enth.forward()
        return float(cp.sum(cp.asarray(w) * g.state.smb.data))

    for (t, i, j) in [(0, 2, 3), (5, 7, 5), (9, 10, 8), (3, 0, 0), (11, 15, 13)]:
        h = 1e-2
        Pp = precip.copy(); Pp[t, i, j] += h
        Pm = precip.copy(); Pm[t, i, j] -= h
        num = (loss(Pp) - loss(Pm)) / (2 * h)
        assert abs(gp[t, i, j] - num) <= 1e-2 * (1.0 + abs(num)), (t, i, j)


# --------------------------------------------------------------------------- #
# Avalanche off is transparent: with no operator attached the effective precip is a
# straight copy of the raw precip, so the result is byte-identical to not having the
# indirection at all.
# --------------------------------------------------------------------------- #
def test_avalanche_off_is_identity():
    rng = np.random.default_rng(4)
    nt, ny, nx, dx = 12, 5, 4, 100.0
    enth = EnthalpyModel(ny=ny, nx=nx, nt=nt, dx=dx, dt=1 / nt, use_fast_math=False)
    g = enth.grid
    precip = rng.uniform(0.0, 2.0, (nt, ny, nx)).astype(np.float32)
    g.precipitation.precip.set(cp.asarray(precip))
    g.temperature.t2m.set(cp.asarray(rng.uniform(-10.0, 5.0, (nt, ny, nx)).astype(np.float32)))
    g.geometry.t_base.set(-8.0)
    enth.forward()
    assert float(cp.max(cp.abs(g.precipitation.precip_eff.data
                               - g.precipitation.precip.data))) == 0.0


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
