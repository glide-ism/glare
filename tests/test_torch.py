"""Tests for the PyTorch autograd wrappers (glare/torch.py).

Run with:  python -m pytest tests/test_torch.py
Requires a CUDA device (cupy) and torch; the whole module is skipped otherwise.
"""
import numpy as np

try:
    import pytest
except ImportError:                       # allow running without pytest installed
    pytest = None

import cupy as cp
assert cp.cuda.runtime.getDeviceCount() >= 1, "no CUDA device"

try:
    import torch
    _HAVE_TORCH = torch.cuda.is_available()
except ImportError:                       # torch is an optional dependency
    torch = None
    _HAVE_TORCH = False

from glare import EnthalpyModel, ImprovedTemperatureIndex
from glare.enthalpy import SECONDS_PER_YEAR
from glare.avalanche import AvalancheOperator

if _HAVE_TORCH:
    from glare.torch import EnthalpyStep, GlareStep

_skip = None
if pytest is not None:
    _skip = pytest.mark.skipif(not _HAVE_TORCH,
                               reason="torch (with CUDA) not available")


def _maybe_skip(fn):
    return _skip(fn) if _skip is not None else fn


DEV = "cuda"


def _t(a, requires_grad=True):
    return torch.tensor(np.asarray(a, np.float32), device=DEV,
                        requires_grad=requires_grad)


# --------------------------------------------------------------------------- #
# EnthalpyStep: forward returns smb; backward populates a gradient on each of the
# thirteen differentiable inputs (incl. the optional debris field), and every
# gradient matches a direct model.forward()/model.adjoint() (the autograd wiring
# adds nothing but bookkeeping).
# --------------------------------------------------------------------------- #
def _enthalpy_step_case(with_avalanche):
    rng = np.random.default_rng(0)
    nt, ny, nx, dx = 12, 8, 6, 100.0
    enth = EnthalpyModel(ny=ny, nx=nx, nt=nt, dx=dx, dt=1 / nt, use_fast_math=False,
                         seed=0)
    if with_avalanche:
        z = np.zeros((ny, nx), np.float32)
        for i in range(ny):
            z[i, :] = (6 - i) * 60.0 if i < 6 else 0.0
        enth.grid.geometry.srf.set(cp.asarray(z))
        enth.avalanche = AvalancheOperator(enth.grid, s_crit=30.0, w_trans=8.0,
                                           p=1.5, K=40)

    t2m = _t(rng.uniform(-10, 5, (nt, ny, nx)))
    precip = _t(rng.uniform(0.0, 2.0, (nt, ny, nx)))
    insol = _t(rng.uniform(0.0, 0.6, (nt, ny, nx)))
    t_base = _t(rng.uniform(-12.0, -2.0, (ny, nx)))
    H_atm = _t(1.0 * SECONDS_PER_YEAR)
    H_base0 = _t(0.6 * SECONDS_PER_YEAR)
    q_sw_bulk = _t(100.0 * SECONDS_PER_YEAR)
    q_sw_insol = _t(200.0 * SECONDS_PER_YEAR)
    q_lw0 = _t(-30.0 * SECONDS_PER_YEAR)
    a_snow = _t(0.9)
    a_ice = _t(0.4)
    M_alb = _t(20.0)
    debris = _t(rng.uniform(0.2, 1.0, (ny, nx)))
    inputs = [t2m, precip, insol, t_base, H_atm, H_base0, q_sw_bulk, q_sw_insol,
              q_lw0, a_snow, a_ice, M_alb, debris]

    smb = EnthalpyStep.apply(enth, *inputs)
    assert smb.shape == (nt, ny, nx) and smb.device.type == "cuda"
    # smb matches a plain forward on the same model.
    enth.forward()
    assert float(np.max(np.abs(smb.detach().cpu().numpy()
                               - cp.asnumpy(enth.grid.state.smb.data)))) <= 1e-4

    w = torch.tensor(rng.normal(0, 1, (nt, ny, nx)).astype(np.float32), device=DEV)
    (smb * w).sum().backward()
    assert all(x.grad is not None for x in inputs)

    # Reference: direct forward+adjoint with the same forcing/seed.
    enth.forward()
    enth.adjoint(grad_smb=cp.asarray(w.detach().cpu().numpy()))
    g = enth.grid

    def close_field(t, cparr):
        return float(np.max(np.abs(t.grad.detach().cpu().numpy() - cp.asnumpy(cparr))))

    tol = 1e-3
    assert close_field(t2m, g.temperature.t2m.grad) <= tol
    assert close_field(precip, g.precipitation.precip.grad) <= tol
    assert close_field(insol, g.radiation.insol_mean.grad) <= tol
    assert close_field(t_base, g.geometry.t_base.grad) <= tol
    assert close_field(debris, g.geometry.debris.grad) <= tol
    for tens, const in ((H_atm, g.thermodynamics.H_atm),
                        (H_base0, g.thermodynamics.H_base0),
                        (q_sw_bulk, g.radiation.q_sw_bulk),
                        (q_sw_insol, g.radiation.q_sw_insol),
                        (q_lw0, g.radiation.q_lw0),
                        (a_snow, g.radiation.albedo_snow),
                        (a_ice, g.radiation.albedo_ice),
                        (M_alb, g.radiation.M_albedo)):
        assert abs(float(tens.grad) - float(const.grad)) <= 1e-2 * (1.0 + abs(float(const.grad)))


@_maybe_skip
def test_enthalpy_step_matches_adjoint():
    _enthalpy_step_case(with_avalanche=False)


@_maybe_skip
def test_enthalpy_step_with_avalanche():
    # Precip gradient must flow back through R^T inside the autograd wrapper too.
    _enthalpy_step_case(with_avalanche=True)


@_maybe_skip
def test_shared_model_multi_step_adjoint():
    # Several autograd nodes may share one model inside a single graph (e.g. a
    # per-year anomaly average within one checkpointed smb call). Each node's
    # backward re-sets its own saved inputs, so the adjoint must re-derive the
    # effective inputs (ETIM's partitioned/avalanched snowfall, enthalpy's
    # precip_eff) from them rather than replay against whatever the LAST
    # forward left on the grid. Check: gradients of a weighted two-call loss
    # equal the weighted sum of independently-computed single-call gradients.
    rng = np.random.default_rng(2)
    nt, ny, nx = 12, 8, 6

    def _etim_case(shift):
        m = ImprovedTemperatureIndex(ny=ny, nx=nx, nt=nt, dx=100.0, dt=1 / nt)
        z = np.zeros((ny, nx), np.float32)
        for i in range(ny):
            z[i, :] = max(6 - i, 0) * 60.0
        m.grid.geometry.srf.set(cp.asarray(z))
        m.avalanche = AvalancheOperator(m.grid, s_crit=30.0, w_trans=8.0,
                                        p=1.5, K=40)
        return m

    t2m_np = rng.uniform(-10, 5, (nt, ny, nx)).astype(np.float32)
    precip_np = rng.uniform(0.0, 2.0, (nt, ny, nx)).astype(np.float32)
    # Debris < 1 and nonzero insolation keep the snow-depth-dependent paths
    # (the sig-blended debris factor and the albedo-modulated rf term) live —
    # with clean ice and zero insolation the stale-snowfall error is invisible.
    debris_np = np.full((ny, nx), 0.5, np.float32)
    insol_np = rng.uniform(0.1, 0.6, (nt, ny, nx)).astype(np.float32)
    w_np = rng.normal(0, 1, (nt, ny, nx)).astype(np.float32)
    shifts, weights = (0.0, 3.0), (0.3, 0.7)

    def grads(shift_list, weight_list):
        m = _etim_case(0.0)
        m.grid.insolation.insol_mean.set(cp.asarray(insol_np))
        m.grid.insolation.insol_cos.set(cp.asarray(0.3 * insol_np))
        m.grid.insolation.insol_sin.set(cp.asarray(0.2 * insol_np))
        t2m = _t(t2m_np)
        precip = _t(precip_np)
        mf, rf = _t(1.825), _t(18.25)
        debris = _t(debris_np, False)
        w = torch.tensor(w_np, device=DEV)
        L = 0.0
        for s, ww in zip(shift_list, weight_list):
            L = L + ww * (w * GlareStep.apply(m, t2m + s, precip, mf, rf, debris)).sum()
        L.backward()
        return {k: v.grad.detach().clone() for k, v in
                (("t2m", t2m), ("precip", precip), ("mf", mf), ("rf", rf))}

    combined = grads(shifts, weights)
    ga = grads(shifts[:1], (1.0,))
    gb = grads(shifts[1:], (1.0,))
    for k in combined:
        expect = weights[0] * ga[k] + weights[1] * gb[k]
        denom = float(expect.abs().max()) + 1e-12
        assert float((combined[k] - expect).abs().max()) / denom < 1e-3, k


@_maybe_skip
def test_avalanche_step():
    # Standalone R wrapper: matches the raw operator, commutes with scalar
    # multiplication (the property hoisting relies on), and its backward is
    # R^T (adjoint dot-product identity through autograd).
    from glare.torch import AvalancheStep

    rng = np.random.default_rng(3)
    nt, ny, nx = 12, 24, 20
    enth = EnthalpyModel(ny=ny, nx=nx, nt=nt, dx=100.0, dt=1 / nt,
                         use_fast_math=False, seed=0)
    z = rng.uniform(0, 1, (ny, nx)).astype(np.float32) * 50.0
    z[:, :10] += np.arange(10, 0, -1, dtype=np.float32) * 80.0   # a steep wall
    enth.grid.geometry.srf.set(cp.asarray(z))
    op = AvalancheOperator(enth.grid, s_crit=30.0, w_trans=8.0, p=1.5, K=15)

    P_np = rng.uniform(0.0, 2.0, (nt, ny, nx)).astype(np.float32)
    P = _t(P_np)
    eff = AvalancheStep.apply(op, P)

    # Matches a direct operator application.
    raw = cp.asarray(P_np)
    eff_ref = cp.empty_like(raw)
    op.forward(raw, eff_ref)
    assert float(np.max(np.abs(eff.detach().cpu().numpy()
                               - cp.asnumpy(eff_ref)))) <= 1e-4
    # Mass conservation (R is a redistribution).
    assert abs(float(eff.sum()) - float(P.sum())) <= 1e-2 * float(P.sum())

    # Scalar commute: R(c * P) == c * R(P) up to atomicAdd jitter.
    with torch.no_grad():
        eff_scaled = AvalancheStep.apply(op, 3.0 * P)
        denom = float(eff.abs().max())
        assert float((eff_scaled - 3.0 * eff).abs().max()) / denom <= 1e-3

    # Backward is R^T: <R P, w> == <P, R^T w>.
    w = torch.tensor(rng.normal(0, 1, (nt, ny, nx)).astype(np.float32),
                     device=DEV)
    (eff * w).sum().backward()
    lhs = float((eff.detach() * w).sum())
    rhs = float((P.detach() * P.grad).sum())   # grad holds R^T w
    assert abs(lhs - rhs) <= 1e-3 * (abs(lhs) + 1.0)


@_maybe_skip
def test_enthalpy_step_optional_args():
    # Debris and temp_deviations are trailing optional args: the 12-input
    # call still works (debris -> clean ice), and omitting debris after a debris
    # call must reset the grid field rather than reuse the stale one.
    rng = np.random.default_rng(1)
    nt, ny, nx = 12, 6, 5
    enth = EnthalpyModel(ny=ny, nx=nx, nt=nt, dx=100.0, dt=1 / nt,
                         use_fast_math=False, seed=0)
    args = [_t(rng.uniform(-10, 5, (nt, ny, nx)), False),
            _t(rng.uniform(0.0, 2.0, (nt, ny, nx)), False),
            _t(rng.uniform(0.0, 0.6, (nt, ny, nx)), False),
            _t(rng.uniform(-12.0, -2.0, (ny, nx)), False),
            _t(1.0 * SECONDS_PER_YEAR, False), _t(0.6 * SECONDS_PER_YEAR, False),
            _t(100.0 * SECONDS_PER_YEAR, False), _t(200.0 * SECONDS_PER_YEAR, False),
            _t(0.0, False),
            _t(0.9, False), _t(0.4, False), _t(20.0, False)]
    with torch.no_grad():
        clean = EnthalpyStep.apply(enth, *args)                       # legacy form
        dusty = EnthalpyStep.apply(enth, *args, _t(np.full((ny, nx), 0.3), False))
        clean_again = EnthalpyStep.apply(enth, *args)                 # resets debris
    assert not torch.equal(clean, dusty)
    assert torch.equal(clean, clean_again)


# --------------------------------------------------------------------------- #
# GlareStep (the ETIM sibling) still runs end-to-end and returns its five grads.
# --------------------------------------------------------------------------- #
@_maybe_skip
def test_glare_step_smoke():
    rng = np.random.default_rng(1)
    nt, ny, nx, dx = 12, 6, 5, 100.0
    m = ImprovedTemperatureIndex(ny=ny, nx=nx, nt=nt, dx=dx, dt=1 / nt)
    t2m = _t(rng.uniform(-8, 4, (nt, ny, nx)))
    precip = _t(rng.uniform(0, 2, (nt, ny, nx)))
    mf = _t(2.0)
    rf = _t(50.0)
    debris = _t(np.ones((ny, nx), np.float32))

    smb = GlareStep.apply(m, t2m, precip, mf, rf, debris)
    assert smb.shape == (nt, ny, nx)
    smb.sum().backward()
    assert all(x.grad is not None for x in (t2m, precip, mf, rf, debris))


if __name__ == "__main__":
    import traceback
    if not _HAVE_TORCH:
        print("SKIP  torch (with CUDA) not available")
        raise SystemExit(0)
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t.__wrapped__() if hasattr(t, "__wrapped__") else t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
