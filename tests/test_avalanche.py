"""Tests for the avalanche redistribution operator and the snow/rain partition.

Run with:  python -m pytest tests/test_avalanche.py
Requires a CUDA device (cupy); the whole module is skipped otherwise.
"""
import numpy as np

try:
    import pytest
except ImportError:                       # allow running without pytest installed
    pytest = None

import cupy as cp
assert cp.cuda.runtime.getDeviceCount() >= 1, "no CUDA device"

from glare.model import ImprovedTemperatureIndex
from glare.avalanche import (
    AvalancheOperator,
    partition_forward,
    partition_adjoint,
)


def _ramp_bench_dem(ny, nx, dx, drop_per_row=60.0, bench_row=6):
    """Steep ramp (rows 0..bench_row, descending southward) above a flat bench."""
    z = np.zeros((ny, nx), dtype=np.float32)
    for i in range(ny):
        if i < bench_row:
            z[i, :] = (bench_row - i) * drop_per_row
        else:
            z[i, :] = 0.0
    return cp.asarray(z)


# --------------------------------------------------------------------------- #
# Partition adjoint: finite-difference check of partition_forward/adjoint.
# --------------------------------------------------------------------------- #
def test_partition_adjoint_matches_fd():
    rng = np.random.default_rng(0)
    nt, ny, nx = 3, 6, 5
    sigma = 5.0
    T = cp.asarray(rng.uniform(-8, 8, (nt, ny, nx)).astype(np.float32))
    P = cp.asarray(rng.uniform(0.0, 3.0, (nt, ny, nx)).astype(np.float32))
    y = cp.asarray(rng.uniform(-1, 1, (nt, ny, nx)).astype(np.float32))

    raw = cp.zeros_like(P)
    partition_forward(T, P, sigma, raw)

    gP = cp.zeros_like(P)
    gT = cp.zeros_like(P)
    partition_adjoint(T, P, sigma, y, gP, gT)

    # J = <raw, y>; check dJ/dP and dJ/dT at a few entries by central differences.
    def J(Tf, Pf):
        r = cp.zeros_like(Pf)
        partition_forward(Tf, Pf, sigma, r)
        return float((r * y).sum())

    eps = 1e-2
    for (t, i, j) in [(0, 1, 2), (1, 3, 0), (2, 5, 4)]:
        Pp = P.copy(); Pp[t, i, j] += eps
        Pm = P.copy(); Pm[t, i, j] -= eps
        fd_P = (J(T, Pp) - J(T, Pm)) / (2 * eps)
        assert abs(fd_P - float(gP[t, i, j])) <= 1e-2 * (1 + abs(fd_P))

        Tp = T.copy(); Tp[t, i, j] += eps
        Tm = T.copy(); Tm[t, i, j] -= eps
        fd_T = (J(Tp, P) - J(Tm, P)) / (2 * eps)
        assert abs(fd_T - float(gT[t, i, j])) <= 1e-2 * (1 + abs(fd_T))


# --------------------------------------------------------------------------- #
# Avalanche operator: adjoint dot-product test <R s, y> == <s, R^T y>.
# rtol is loose because the forward scatter uses atomicAdd (non-deterministic
# summation order in FP32) -- the accepted cost of the scatter design.
# --------------------------------------------------------------------------- #
def test_avalanche_adjoint_dot_product():
    rng = np.random.default_rng(1)
    nt, ny, nx, dx = 4, 16, 14, 100.0
    m = ImprovedTemperatureIndex(ny=ny, nx=nx, nt=nt, dx=dx, dt=1 / 12)
    m.grid.geometry.srf.set(_ramp_bench_dem(ny, nx, dx))
    ava = AvalancheOperator(m.grid, s_crit=30.0, w_trans=8.0, p=1.5, K=40)

    s = cp.asarray(rng.uniform(0, 2, (nt, ny, nx)).astype(np.float32))
    y = cp.asarray(rng.uniform(-1, 1, (nt, ny, nx)).astype(np.float32))

    Rs = cp.zeros_like(s)
    ava.forward(s, Rs)
    RTy = cp.zeros_like(s)
    ava.adjoint(y, RTy)

    lhs = float((Rs * y).sum())
    rhs = float((s * RTy).sum())
    assert abs(lhs - rhs) <= 1e-4 * (1 + abs(lhs))


# --------------------------------------------------------------------------- #
# Mass conservation: with off-grid neighbours excluded and the residual mobile
# snow deposited after K passes, R conserves mass per slab to ~FP precision
# (boundary export = 0 in v1).
# --------------------------------------------------------------------------- #
def test_mass_conservation():
    rng = np.random.default_rng(2)
    nt, ny, nx, dx = 2, 16, 16, 100.0
    m = ImprovedTemperatureIndex(ny=ny, nx=nx, nt=nt, dx=dx, dt=1 / 12)
    m.grid.geometry.srf.set(_ramp_bench_dem(ny, nx, dx))
    ava = AvalancheOperator(m.grid, K=60)

    s = cp.asarray(rng.uniform(0, 2, (nt, ny, nx)).astype(np.float32))
    Rs = cp.zeros_like(s)
    ava.forward(s, Rs)

    for t in range(nt):
        in_mass = float(s[t].sum())
        out_mass = float(Rs[t].sum())
        assert abs(out_mass - in_mass) <= 1e-4 * in_mass


# --------------------------------------------------------------------------- #
# Truncation: residual mobile snow after K passes decays as K grows.
# --------------------------------------------------------------------------- #
def test_truncation_decreases_with_K():
    rng = np.random.default_rng(3)
    nt, ny, nx, dx = 1, 24, 8, 100.0
    s = cp.asarray(rng.uniform(0, 2, (nt, ny, nx)).astype(np.float32))

    def residual_for(K):
        m = ImprovedTemperatureIndex(ny=ny, nx=nx, nt=nt, dx=dx, dt=1 / 12)
        m.grid.geometry.srf.set(_ramp_bench_dem(ny, nx, dx, bench_row=ny - 2))
        ava = AvalancheOperator(m.grid, s_crit=20.0, w_trans=6.0, p=1.5, K=K)
        return ava.residual(s)

    r_small = residual_for(5)
    r_large = residual_for(80)
    assert r_large < r_small
    assert r_large <= 1e-3 * float(s.sum())


# --------------------------------------------------------------------------- #
# Runout: a point source high on the steep ramp deposits downslope (toward the
# bench), not upslope.
# --------------------------------------------------------------------------- #
def test_runout_is_downslope():
    nt, ny, nx, dx = 1, 16, 9, 100.0
    m = ImprovedTemperatureIndex(ny=ny, nx=nx, nt=nt, dx=dx, dt=1 / 12)
    # Steep ramp descending toward increasing i, flat bench at the bottom.
    m.grid.geometry.srf.set(_ramp_bench_dem(ny, nx, dx, drop_per_row=80.0,
                                            bench_row=8))
    ava = AvalancheOperator(m.grid, s_crit=25.0, w_trans=6.0, p=1.5, K=60)

    src_row = 1
    s = cp.zeros((nt, ny, nx), dtype=cp.float32)
    s[0, src_row, nx // 2] = 1.0
    Rs = cp.zeros_like(s)
    ava.forward(s, Rs)

    dep = cp.asnumpy(Rs[0])
    upslope_mass = dep[:src_row].sum()       # rows above the source
    bench_mass = dep[8:].sum()               # the flat bench below
    assert upslope_mass <= 1e-6
    assert bench_mass > 0.5                   # most of the unit source runs out


# --------------------------------------------------------------------------- #
# Disabled operator (R = I): the partition+kernel pipeline runs and snowfall
# equals the raw solid precip.
# --------------------------------------------------------------------------- #
def test_identity_when_disabled():
    rng = np.random.default_rng(4)
    nt, ny, nx, dx = 12, 10, 10, 100.0
    m = ImprovedTemperatureIndex(ny=ny, nx=nx, nt=nt, dx=dx, dt=1 / 12)
    m.grid.temperature.t2m.set(cp.asarray(rng.uniform(-10, 5, (nt, ny, nx)).astype(np.float32)))
    m.grid.precipitation.precip.set(cp.asarray(rng.uniform(0, 3, (nt, ny, nx)).astype(np.float32)))
    m.grid.insolation.insol_mean.set(cp.asarray(rng.uniform(0, 0.5, (nt, ny, nx)).astype(np.float32)))

    m.forward()  # avalanche is None
    raw = cp.zeros_like(m.grid.precipitation.precip.data)
    partition_forward(m.grid.temperature.t2m.data,
                      m.grid.precipitation.precip.data,
                      m.grid.temperature.sigma_t2m.value, raw)
    assert float(cp.abs(m.grid.precipitation.snowfall.data - raw).max()) == 0.0


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
