"""Depth-averaged snowpack enthalpy model.

A physically explicit thermodynamic core for the seasonal snowpack, offered as a
*companion* to the PDD/radiation SMB kernel in ``cuda/smb.cu``.  Where that kernel
tracks a scalar snow depth and melts it with positive-degree-days plus a radiation
term, this model carries an active surface reservoir of mass ``M`` and enthalpy
``E`` and resolves its temperature, refreezing, runoff and ice melt from an
implicit surface energy balance.

State convention
----------------
``M`` : active surface-reservoir mass [kg m-2]
``E`` : reservoir enthalpy relative to ice at 0 degC [J m-2]

After each step, liquid water drains immediately, so the *retained* state obeys
``M >= 0`` and ``E <= 0``.  For ``M > 0``, ``T_s = E / (M c_i)``.  A positive
provisional enthalpy corresponds to meltwater and is converted to runoff at 0 degC
(and, on a glacier surface, to ice melt once the snow is exhausted).

This module has two layers:

* a **scalar reference** (:class:`State`, :class:`Fluxes`, :func:`step`,
  :func:`run_column`) -- a readable, NumPy implementation that is the physics of
  record and the test oracle.  It is the packaged form of the toy
  ``enthalpy_sketch_revised.py``.  Its scalar parameters come from an
  :class:`~glare.grid.EnthalpyGrid` via :func:`scalar_params`.
* an **operator** (:class:`EnthalpyModel`) -- a thin driver that runs the same
  recurrence on the GPU via the grid-cached ``Enthalpy{Forward,Backward}Operators``
  (``cuda/enthalpy.cu``) over an :class:`~glare.grid.EnthalpyGrid`, reproducing the
  reference to floating-point tolerance.  It is the enthalpy sibling of
  :class:`~glare.model.ImprovedTemperatureIndex`: same shapes
  (``model.grid.state.<field>``, ``model.forward()``, ``model.adjoint(...)``).

The reference and the kernel are kept in lock-step exactly as the avalanche module
keeps its Python :func:`~glare.avalanche.partition_forward` in step with
``cuda/avalanche.cu``.

The model is reverse-mode differentiable: :func:`run_column_adjoint` (scalar) and
``cuda/enthalpy.cu``'s ``compute_enthalpy_grad`` (driven by
:meth:`EnthalpyModel.adjoint`) are the hand-derived VJP of the water-year recurrence
-- a checkpointed replay then a reverse-in-time scan carrying the state adjoints, as
in ``compute_smb_grad``.  Like ETIM, the adjoint writes gradients directly into the
grid's ``Field.grad`` / ``Constant.grad`` buffers: the forcing (``t2m``, ``precip``,
``insol_mean``, ``t_base``) and the seven energy-balance parameters in
:data:`~glare.operators.GRAD_PARAM_NAMES`.  Burial of the active reservoir into ice
(``M_active_max``) is not applied in v1, matching the reference ``step``.
"""

from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import cupy as cp

from glare.grid import EnthalpyGrid
from glare.operators import GRAD_PARAM_NAMES

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


# --------------------------------------------------------------------------- #
# Diagnostic containers and the grid -> scalar-parameter view.
# --------------------------------------------------------------------------- #
@dataclass
class State:
    M: float = 0.0  # kg m-2
    E: float = 0.0  # J m-2


@dataclass
class Fluxes:
    precipitation: float = 0.0
    runoff_from_pack: float = 0.0
    ice_melt: float = 0.0
    smb: float = np.nan          # surface mass balance rate [m a-1 w.e.]
    q_nonadvective: float = 0.0
    albedo: float = np.nan
    T_surface: float = np.nan


def scalar_params(grid) -> SimpleNamespace:
    """Read the enthalpy scalar parameters off an ``EnthalpyGrid`` as Python floats.

    The grid ``Constant``s are the single source of truth for the parameters (there is
    no ``Parameters`` dataclass); this returns a lightweight read-only view the NumPy
    scalar reference (:func:`step`/:func:`run_column`) consumes via ``p.<attr>``,
    including a precomputed ``inv_M_insulation`` (0 when the insulation scale is
    infinite/invalid, i.e. a constant basal conductance).  It touches only scalars, so
    the reference stays pure-NumPy once the view is built.
    """
    th, rad, tm = grid.thermodynamics, grid.radiation, grid.temperature
    m_ins = float(th.M_insulation.value)
    inv = 1.0 / m_ins if (m_ins and np.isfinite(m_ins) and m_ins > 0.0) else 0.0
    return SimpleNamespace(
        L_f=float(th.L_f.value), c_i=float(th.c_i.value), c_w=float(th.c_w.value),
        H_atm=float(th.H_atm.value), H_base0=float(th.H_base0.value),
        q_sw_bulk=float(rad.q_sw_bulk.value), q_sw_insol=float(rad.q_sw_insol.value),
        albedo_snow=float(rad.albedo_snow.value), albedo_ice=float(rad.albedo_ice.value),
        M_albedo=float(rad.M_albedo.value),
        T_transition=float(tm.T_transition.value),
        M_insulation=m_ins, inv_M_insulation=inv,
        M_eps=float(th.M_eps.value), rho_w=float(th.rho_w.value))


# --------------------------------------------------------------------------- #
# Scalar reference: the physics of record (packaged enthalpy_sketch_revised.py).
# --------------------------------------------------------------------------- #
def rain_fraction(T_air: float, width: float) -> float:
    return 1.0 / (1.0 + np.exp(-T_air / width))


def precipitation_enthalpy(T_air: float, f_rain: float, p) -> float:
    """Specific enthalpy [J kg-1], referenced to ice at 0 degC.

    Solid precipitation is not allowed to be warmer than 0 degC and liquid
    precipitation is not allowed to be colder than 0 degC, so the smoothing
    interval never implies warm ice or supercooled rain.
    """
    T_solid = min(T_air, 0.0)
    T_liquid = max(T_air, 0.0)
    return ((1.0 - f_rain) * p.c_i * T_solid
            + f_rain * (p.L_f + p.c_w * T_liquid))


def solid_mass_from_state(M: float, E: float, p) -> float:
    """Diagnostic solid mass before drainage, used only for albedo."""
    liquid_mass = np.clip(max(E, 0.0) / p.L_f, 0.0, M)
    return max(M - liquid_mass, 0.0)


def snow_cover_fraction(M_solid: float, M_scale: float) -> float:
    """Exactly zero for bare ice and asymptotes smoothly to one."""
    return 1.0 - np.exp(-max(M_solid, 0.0) / M_scale)


def basal_conductance(M_solid: float, p) -> float:
    """Simple snow-insulation parameterization."""
    if np.isinf(p.M_insulation):
        return p.H_base0
    return p.H_base0 / (1.0 + M_solid / p.M_insulation)


def step(
    state: State,
    P: float,          # kg m-2 yr-1
    T_air: float,      # degC
    T_base: float,     # degC
    dt: float,         # yr
    p,                 # scalar-parameter view (see scalar_params)
    glacier_surface: bool = True,
    insolation: float = 0.0,   # monthly-mean insolation [fraction of direct sun]
    debris: float = 1.0,       # debris melt attenuation for snow-free ice [0, 1]
) -> tuple[State, Fluxes]:
    """Advance one forcing interval.

    The atmospheric and basal terms are
        ``q = q_sw + H_atm (T_air - T_s) + H_base (T_base - T_s)``,
    with ``T_s <= 0`` degC.  The cold branch is solved by backward Euler.  If the
    implicit cold solution crosses 0 degC, the surface is held at 0 degC and
    positive energy is converted to runoff and, after snow exhaustion, ice melt.

    ``debris`` attenuates melt where snow is not present: every glacier-ice melt
    flux (the pack-exhausted residual of the melting branch and the bare
    quasi-steady branch) is multiplied by it.  1 is clean ice, 0 fully
    suppresses bare-ice melt; the pack (snow) evolution is untouched.  Ice melt
    never feeds back into the retained ``(M, E)`` state, so the attenuation is
    purely a flux (and hence smb) modification.
    """
    M0, E0 = float(state.M), float(state.E)
    if M0 < -p.M_eps or E0 > p.M_eps:
        raise ValueError("Retained state must satisfy M >= 0 and E <= 0")

    flux = Fluxes()

    # Advective mass and enthalpy input from precipitation.
    dM = max(P, 0.0) * dt
    f_rain = rain_fraction(T_air, p.T_transition)
    h_precip = precipitation_enthalpy(T_air, f_rain, p)
    M = M0 + dM
    E = E0 + dM * h_precip
    flux.precipitation = dM

    # Use diagnostic solid mass, not total rain+snow mass, for albedo.
    M_solid = solid_mass_from_state(M, E, p)
    f_snow = snow_cover_fraction(M_solid, p.M_albedo)
    alpha = p.albedo_ice + f_snow * (p.albedo_snow - p.albedo_ice)
    q_sw = (1.0 - alpha) * (p.q_sw_bulk + p.q_sw_insol * insolation)

    H_base = basal_conductance(M_solid, p)
    H_total = p.H_atm + H_base
    Q0 = q_sw + p.H_atm * T_air + H_base * T_base
    flux.albedo = alpha

    if M > p.M_eps:
        # Backward-Euler cold branch: T_s = E_new / (M c_i).
        E_cold = (E + dt * Q0) / (1.0 + dt * H_total / (M * p.c_i))

        if E_cold <= 0.0:
            E = E_cold
            T_surface = E / (M * p.c_i)
            q_nonadvective = q_sw + p.H_atm * (T_air - T_surface) \
                              + H_base * (T_base - T_surface)
        else:
            # Melting branch: T_s = 0 degC.
            T_surface = 0.0
            q_nonadvective = Q0
            E_available = E + dt * q_nonadvective

            runoff = min(E_available / p.L_f, M)
            excess_energy = E_available - runoff * p.L_f
            ice_melt = debris * max(excess_energy, 0.0) / p.L_f \
                if glacier_surface else 0.0

            M -= runoff
            E = 0.0
            flux.runoff_from_pack = runoff
            flux.ice_melt = ice_melt
    else:
        # No snow reservoir exists.  Do not store negative enthalpy in a
        # zero-mass state.  Treat the exposed surface as quasi-steady.
        if H_total > 0.0:
            T_equilibrium = Q0 / H_total
        else:
            T_equilibrium = 0.0

        if T_equilibrium < 0.0:
            T_surface = T_equilibrium
            q_nonadvective = 0.0  # exact steady balance at T_equilibrium
        else:
            T_surface = 0.0
            q_nonadvective = Q0
            flux.ice_melt = debris * max(dt * q_nonadvective, 0.0) / p.L_f \
                            if glacier_surface else 0.0
        M, E = 0.0, 0.0

    flux.q_nonadvective = q_nonadvective
    flux.T_surface = T_surface

    # Numerical cleanup and invariants.
    if M < p.M_eps:
        M, E = 0.0, 0.0
    if M < -p.M_eps or E > p.M_eps:
        raise RuntimeError("Thermodynamic projection failed")

    # Surface mass balance rate [m a-1 w.e.]: net change of the retained reservoir
    # mass less glacier-ice melt (see the kernel for the annual-sum identity).
    flux.smb = (M - M0 - flux.ice_melt) / (p.rho_w * dt)

    return State(M=M, E=E), flux


def generate_temp_deviations(nt, n_sub, sigma, rng, demean=True):
    """Sub-step air-temperature deviations [degC], shape ``(nt, n_sub)``.

    Draws ``sigma * z`` with ``z`` standard normal from ``rng`` (a
    ``numpy.random.Generator``).  With ``demean=True`` each month's ``n_sub``
    deviations are shifted to sum to zero, so the monthly-mean forcing is preserved
    exactly and any change relative to the deterministic run is the *pure variance*
    effect.  The randomness lives here in Python; the kernel only ever sees the
    resulting degC vector.
    """
    z = rng.standard_normal((nt, n_sub))
    dev = sigma * z
    if demean:
        dev -= dev.mean(axis=1, keepdims=True)
    return dev.astype(np.float32)


def run_column(
    precip,            # (nt,) total precipitation [m a-1 w.e.]
    t2m,               # (nt,) air temperature [degC]
    t_base: float,     # basal/substrate temperature [degC]
    dt: float,         # yr
    p,                 # scalar-parameter view from scalar_params(grid)
    start_month: int = 9,
    glacier_surface: bool = True,
    insol=None,        # (nt,) monthly-mean insolation [fraction]; None -> zeros
    temp_dev=None,     # (nt, n_sub) sub-step deviations [degC]; None -> 1 step, no noise
    debris: float = 1.0,  # debris melt attenuation for snow-free ice [0, 1]
) -> dict:
    """Scan one water year of a single column with the scalar :func:`step`.

    Mirrors the kernel exactly: starts from a cold, snow-free surface
    (``M = E = 0``), scans ``start_month -> +nt``, and returns the per-month
    diagnostics in **calendar-month order** (the same layout ``EnthalpyModel``
    writes).  ``precip`` is converted from m a-1 w.e. to the SI mass rate the
    reference expects with ``p.rho_w``.  ``p`` is a scalar-parameter view built with
    :func:`scalar_params` from the ``EnthalpyGrid`` whose Constants hold the values.

    When ``temp_dev`` is supplied, each month is advanced in ``n_sub =
    temp_dev.shape[1]`` sub-steps of length ``dt/n_sub``, with the sub-step air
    temperature ``t2m[m] + temp_dev[m, d]``.  Fluxes (``runoff``, ``ice_melt``) are
    summed over sub-steps, diagnostics (``t_surface``, ``albedo``) averaged, and
    ``smb`` recomputed on the month boundary; ``temp_dev=None`` is the single-step
    monthly model.
    """
    precip = np.asarray(precip, dtype=float)
    t2m = np.asarray(t2m, dtype=float)
    nt = precip.shape[0]
    insol = np.zeros(nt) if insol is None else np.asarray(insol, dtype=float)
    if temp_dev is None:
        temp_dev = np.zeros((nt, 1))
    else:
        temp_dev = np.asarray(temp_dev, dtype=float)
    n_sub = temp_dev.shape[1]
    dt_sub = dt / n_sub

    out = {k: np.zeros(nt) for k in
           ("smb", "M", "E", "runoff", "ice_melt", "t_surface", "albedo")}
    state = State()
    for s in range(nt):
        m = (start_month + s) % nt
        P_si = precip[m] * p.rho_w        # m a-1 w.e. -> kg m-2 yr-1
        M_start = state.M                 # month boundary, for the monthly smb

        runoff_m = ice_melt_m = 0.0
        tsurf_sum = alpha_sum = 0.0
        for d in range(n_sub):
            T_air = t2m[m] + temp_dev[m, d]
            state, flux = step(state, P_si, T_air, t_base, dt_sub, p,
                               glacier_surface=glacier_surface, insolation=insol[m],
                               debris=debris)
            runoff_m += flux.runoff_from_pack
            ice_melt_m += flux.ice_melt
            tsurf_sum += flux.T_surface
            alpha_sum += flux.albedo

        # Monthly smb from the month-boundary mass (the per-sub-step flux.smb, a
        # per-dt_sub rate, is discarded).  For n_sub == 1 this equals flux.smb.
        out["smb"][m] = (state.M - M_start - ice_melt_m) / (p.rho_w * dt)
        out["M"][m] = state.M
        out["E"][m] = state.E
        out["runoff"][m] = runoff_m
        out["ice_melt"][m] = ice_melt_m
        out["t_surface"][m] = tsurf_sum / n_sub
        out["albedo"][m] = alpha_sum / n_sub
    return out


# --------------------------------------------------------------------------- #
# Reverse-mode adjoint (scalar reference).
#
# The forward recurrence is a chain of sub-steps carrying the state (M, E).  The
# adjoint is the hand-derived vector-Jacobian product of that chain: a reverse-in-
# time scan carrying the state adjoints (bar_M, bar_E), exactly as `compute_smb_grad`
# carries the snow-depth adjoint.  `_substep_forward` recomputes one sub-step and
# returns every intermediate the backward pass needs; `_substep_backward` is its
# transpose.  These are the oracle for `cuda/enthalpy.cu`'s `compute_enthalpy_grad`,
# kept in lock-step exactly as `run_column` mirrors the forward kernel.
#
# Differentiation targets: the forcing (t2m, precip, insol, t_base), the static
# debris attenuation field, and the seven energy-balance parameters in
# GRAD_PARAM_NAMES.  The thermodynamic constants (L_f, c_i, c_w, T_transition,
# M_insulation, rho_w) are held fixed.  The loss is seeded through the
# mass/energy outputs (smb, M, E, runoff, ice_melt); the pure diagnostics
# (t_surface, albedo) are not seeded.
# --------------------------------------------------------------------------- #
def _substep_forward(M_in, E_in, P, T_air, T_base, insol, dt, p, glacier,
                     debris=1.0):
    """One forcing sub-step, returning ``(M, E, runoff, ice_melt, im)``.

    Numerically identical to :func:`step` (asserted by ``test_substep_matches_step``)
    but also returns ``im``, a dict of every intermediate and branch selector the
    adjoint needs, so the backward pass never has to reconstruct a branch.  ``P`` is
    the SI precip rate [kg m-2 yr-1] (as :func:`step` takes it).
    """
    Pc = max(P, 0.0)
    dM = Pc * dt
    fr = 1.0 / (1.0 + np.exp(-T_air / p.T_transition))
    Ts = min(T_air, 0.0)
    Tl = max(T_air, 0.0)
    h = (1.0 - fr) * p.c_i * Ts + fr * (p.L_f + p.c_w * Tl)
    M1 = M_in + dM
    E1 = E_in + dM * h

    # Diagnostic solid mass for albedo/insulation (liquid drained for the optics).
    liq_un = max(E1, 0.0) / p.L_f                 # >= 0
    liq_by_M = liq_un > M1                         # which argument of the min wins
    liq = M1 if liq_by_M else liq_un
    Msol_raw = M1 - liq
    Msol_pos = Msol_raw > 0.0
    Msol = Msol_raw if Msol_pos else 0.0
    fexp = np.exp(-Msol / p.M_albedo)
    fsnow = 1.0 - fexp
    alpha = p.albedo_ice + fsnow * (p.albedo_snow - p.albedo_ice)
    S = p.q_sw_bulk + p.q_sw_insol * insol
    q_sw = (1.0 - alpha) * S

    invMins = p.inv_M_insulation
    denomH = 1.0 + Msol * invMins
    Hb = p.H_base0 / denomH
    Htot = p.H_atm + Hb
    Q0 = q_sw + p.H_atm * T_air + Hb * T_base

    runoff = 0.0
    ice_melt = 0.0
    denom = numer = Ecold = Eavail = 0.0
    runoff_capped = False
    bare_melt = False

    if M1 > p.M_eps:
        denom = 1.0 + dt * Htot / (M1 * p.c_i)
        numer = E1 + dt * Q0
        Ecold = numer / denom
        if Ecold <= 0.0:
            regime = 0                              # cold
            M2 = M1
            E2 = Ecold
        else:
            regime = 1                              # melt at 0 C
            Eavail = E1 + dt * Q0
            runoff_capped = (Eavail / p.L_f) > M1
            runoff = M1 if runoff_capped else Eavail / p.L_f
            excess = Eavail - runoff * p.L_f
            ice_melt = debris * max(excess, 0.0) / p.L_f if glacier else 0.0
            M2 = M1 - runoff
            E2 = 0.0
    else:
        regime = 2                                  # bare surface, quasi-steady
        Teq = Q0 / Htot if Htot > 0.0 else 0.0
        if Teq >= 0.0:
            bare_melt = True
            ice_melt = debris * max(dt * Q0, 0.0) / p.L_f if glacier else 0.0
        M2 = 0.0
        E2 = 0.0

    # Cleanup: a dust-mass reservoir collapses to a clean bare state.  The gate is
    # the transpose of that clamp -- state-output adjoints only flow when it is open.
    gate = 1.0 if M2 >= p.M_eps else 0.0
    if gate == 0.0:
        M2 = 0.0
        E2 = 0.0

    im = dict(M1=M1, E1=E1, h=h, dM=dM, P=P, T_air=T_air, Ts=Ts, Tl=Tl, fr=fr,
              insol=insol, T_base=T_base, liq=liq, liq_by_M=liq_by_M,
              Msol=Msol, Msol_pos=Msol_pos, fexp=fexp, fsnow=fsnow, alpha=alpha,
              S=S, denomH=denomH, Hb=Hb, Htot=Htot, Q0=Q0, denom=denom,
              numer=numer, Ecold=Ecold, Eavail=Eavail, regime=regime,
              runoff_capped=runoff_capped, bare_melt=bare_melt, gate=gate,
              debris=debris)
    return M2, E2, runoff, ice_melt, im


def _substep_backward(im, bar_M2, bar_E2, g_runoff, g_icemelt, dt, p, glacier):
    """Transpose of :func:`_substep_forward`.

    Consumes the adjoints of this sub-step's outputs -- ``bar_M2``/``bar_E2`` of the
    carried state and ``g_runoff``/``g_icemelt`` of the two flux outputs -- and
    returns ``(bar_M_in, bar_E_in, g)``: the adjoints of the input state and a dict
    of gradient contributions for the forcing and the seven parameters.
    """
    # Cleanup clamp (transpose): kill the state-output adjoints when the reservoir
    # collapsed to a clean bare state (gate closed).
    aM2 = bar_M2 * im["gate"]
    aE2 = bar_E2 * im["gate"]

    L_f, c_i = p.L_f, p.c_i
    bar_M1 = 0.0
    bar_E1 = 0.0
    bar_Q0 = 0.0
    bar_Htot = 0.0

    D = im["debris"]
    g_debris = 0.0
    regime = im["regime"]
    if regime == 0:                                 # cold branch
        denom, numer, Ecold = im["denom"], im["numer"], im["Ecold"]
        M1, Htot = im["M1"], im["Htot"]
        bar_numer = aE2 / denom
        bar_denom = -aE2 * Ecold / denom
        bar_E1 += bar_numer
        bar_Q0 += dt * bar_numer
        bar_Htot += bar_denom * (dt / (M1 * c_i))
        bar_M1 += bar_denom * (-dt * Htot / (M1 * M1 * c_i))
        bar_M1 += aM2                                # M2 = M1
    elif regime == 1:                               # melt-at-0 branch
        if im["runoff_capped"]:                     # runoff = M1, ice_melt = D excess/L_f
            bar_M1 += g_runoff                        # d runoff / d M1
            bar_excess = g_icemelt * (D / L_f if glacier else 0.0)
            if glacier:                               # excess > 0 in the capped path
                g_debris += g_icemelt * (im["Eavail"] - im["M1"] * L_f) / L_f
            bar_M1 += bar_excess * (-L_f)             # excess = Eavail - M1 L_f
            bar_Eavail = bar_excess                   # (M2 = 0 -> aM2 gated out)
        else:                                        # runoff = Eavail/L_f, ice_melt = 0
            bar_Eavail = g_runoff * (1.0 / L_f) + aM2 * (-1.0 / L_f)
            bar_M1 += aM2                             # M2 = M1 - runoff, runoff indep of M1
        bar_E1 += bar_Eavail                          # Eavail = E1 + dt Q0
        bar_Q0 += dt * bar_Eavail
    else:                                            # bare branch
        if im["bare_melt"] and glacier and (dt * im["Q0"] > 0.0):
            bar_Q0 += g_icemelt * D * (dt / L_f)      # ice_melt = D dt Q0 / L_f
            g_debris += g_icemelt * dt * im["Q0"] / L_f
        # M2 = E2 = 0: no state-adjoint flow through the regime here.

    # ---- common pre-regime chain (shared by every branch) -------------------- #
    alpha, S, insol = im["alpha"], im["S"], im["insol"]
    Hb, denomH, Msol = im["Hb"], im["denomH"], im["Msol"]
    invMins = p.inv_M_insulation

    g = {k: 0.0 for k in GRAD_PARAM_NAMES}
    g["t2m"] = 0.0
    g["precip"] = 0.0
    g["insol"] = 0.0
    g["t_base"] = 0.0
    g["debris"] = g_debris

    # Q0 = q_sw + H_atm T_air + Hb T_base
    bar_qsw = bar_Q0
    g["H_atm"] += bar_Q0 * im["T_air"]
    bar_T_air = bar_Q0 * p.H_atm
    bar_Hb = bar_Q0 * im["T_base"]
    g["t_base"] += bar_Q0 * Hb
    # Htot = H_atm + Hb
    g["H_atm"] += bar_Htot
    bar_Hb += bar_Htot
    # Hb = H_base0 / denomH,  denomH = 1 + Msol invMins
    g["H_base0"] += bar_Hb * (1.0 / denomH)
    bar_Msol = bar_Hb * (-Hb * invMins / denomH)
    # q_sw = (1 - alpha) (q_sw_bulk + q_sw_insol insol)
    g["q_sw_bulk"] += bar_qsw * (1.0 - alpha)
    g["q_sw_insol"] += bar_qsw * (1.0 - alpha) * insol
    g["insol"] += bar_qsw * (1.0 - alpha) * p.q_sw_insol
    bar_alpha = -bar_qsw * S
    # alpha = albedo_ice + fsnow (albedo_snow - albedo_ice)
    g["albedo_ice"] += bar_alpha * (1.0 - im["fsnow"])
    g["albedo_snow"] += bar_alpha * im["fsnow"]
    bar_fsnow = bar_alpha * (p.albedo_snow - p.albedo_ice)
    # fsnow = 1 - exp(-Msol / M_albedo)
    bar_Msol += bar_fsnow * (im["fexp"] / p.M_albedo)
    g["M_albedo"] += bar_fsnow * (-im["fexp"] * Msol / (p.M_albedo * p.M_albedo))
    # Msol = max(M1 - liq, 0)
    if im["Msol_pos"]:
        bar_M1 += bar_Msol
        bar_liq = -bar_Msol
    else:
        bar_liq = 0.0
    # liq = min(max(E1,0)/L_f, M1)
    if im["liq_by_M"]:
        bar_M1 += bar_liq
    elif im["E1"] > 0.0:
        bar_E1 += bar_liq * (1.0 / L_f)
    # M1 = M_in + dM ;  E1 = E_in + dM h
    bar_M_in = bar_M1
    bar_E_in = bar_E1
    bar_dM = bar_M1 + bar_E1 * im["h"]
    bar_h = bar_E1 * im["dM"]
    # dM = max(P,0) dt  (P is SI; the m w.e. -> kg conversion is applied by the caller)
    g["precip"] = bar_dM * dt if im["P"] > 0.0 else 0.0
    # h = (1 - fr) c_i min(T,0) + fr (L_f + c_w max(T,0)),  fr = sigmoid(T / width)
    fr = im["fr"]
    dfr = fr * (1.0 - fr) / p.T_transition
    dh_dT = ((1.0 - fr) * c_i * (1.0 if im["T_air"] < 0.0 else 0.0)
             + fr * p.c_w * (1.0 if im["T_air"] > 0.0 else 0.0)
             + dfr * ((p.L_f + p.c_w * im["Tl"]) - c_i * im["Ts"]))
    bar_T_air += bar_h * dh_dT
    g["t2m"] = bar_T_air                             # T_air = t2m + dev (dev exogenous)
    return bar_M_in, bar_E_in, g


def run_column_adjoint(
    precip, t2m, t_base, dt, p, start_month=9, glacier_surface=True,
    insol=None, temp_dev=None, debris=1.0,
    grad_smb=None, grad_M=None, grad_E=None, grad_runoff=None, grad_ice_melt=None,
) -> dict:
    """Reverse-mode gradient of one water-year column (scalar reference).

    Takes the same forcing as :func:`run_column` (``p`` from :func:`scalar_params`)
    plus the adjoint seeds -- the gradient of a scalar loss with respect to each
    seeded output field (``(nt,)`` each, calendar-month order; ``None`` -> zeros).
    Returns a plain **dict** of gradients: ``t2m``/``precip``/``insol`` as ``(nt,)``
    arrays, ``t_base`` and ``debris`` scalars (static per-column fields), and the
    seven parameter gradients (summed over the year) keyed by name -- the CPU
    counterpart of the grid ``.grad`` buffers the GPU ``EnthalpyModel.adjoint``
    populates.

    Mirrors ``compute_enthalpy_grad``: a forward replay checkpoints the sub-step
    entry states, then a reverse-in-time scan carries ``(bar_M, bar_E)``.  The
    monthly ``smb`` couples two months (``smb = (M_end - M_prev - ice_melt)/(rho_w
    dt)``), so its seed feeds the end-of-month state, the ice-melt flux, and -- one
    month back -- the start-of-month state.
    """
    precip = np.asarray(precip, dtype=float)
    t2m = np.asarray(t2m, dtype=float)
    nt = precip.shape[0]
    insol = np.zeros(nt) if insol is None else np.asarray(insol, dtype=float)
    if temp_dev is None:
        temp_dev = np.zeros((nt, 1))
    else:
        temp_dev = np.asarray(temp_dev, dtype=float)
    n_sub = temp_dev.shape[1]
    dt_sub = dt / n_sub

    def _seed(a):
        return np.zeros(nt) if a is None else np.asarray(a, dtype=float)
    g_smb = _seed(grad_smb)
    g_M = _seed(grad_M)
    g_E = _seed(grad_E)
    g_runoff = _seed(grad_runoff)
    g_ice = _seed(grad_ice_melt)

    P_si = precip * p.rho_w
    inv = 1.0 / (p.rho_w * dt)

    # Forward replay: checkpoint the (M, E) entering every sub-step of every month.
    sub_entries = [[None] * n_sub for _ in range(nt)]
    M = E = 0.0
    for s in range(nt):
        m = (start_month + s) % nt
        for d in range(n_sub):
            sub_entries[s][d] = (M, E)
            T_air = t2m[m] + temp_dev[m, d]
            M, E, _, _, _ = _substep_forward(M, E, P_si[m], T_air, t_base,
                                             insol[m], dt_sub, p, glacier_surface,
                                             debris=debris)

    grad = {"t2m": np.zeros(nt), "precip": np.zeros(nt), "insol": np.zeros(nt),
            "t_base": 0.0, "debris": 0.0, **{k: 0.0 for k in GRAD_PARAM_NAMES}}

    # Reverse-in-time scan carrying the state adjoints.
    bar_M = bar_E = 0.0
    for s in range(nt - 1, -1, -1):
        m = (start_month + s) % nt
        # Seed the end-of-month state and the two flux outputs for this month.
        bar_M += g_M[m] + g_smb[m] * inv          # d smb / d M_end = +inv
        bar_E += g_E[m]
        g_ro = g_runoff[m]                          # applied to each sub-step's runoff
        g_ic = g_ice[m] - g_smb[m] * inv           # d smb / d ice_melt = -inv
        for d in range(n_sub - 1, -1, -1):
            M_in, E_in = sub_entries[s][d]
            T_air = t2m[m] + temp_dev[m, d]
            _, _, _, _, im = _substep_forward(M_in, E_in, P_si[m], T_air, t_base,
                                              insol[m], dt_sub, p, glacier_surface,
                                              debris=debris)
            bar_M, bar_E, gd = _substep_backward(im, bar_M, bar_E, g_ro, g_ic,
                                                 dt_sub, p, glacier_surface)
            for k in GRAD_PARAM_NAMES:
                grad[k] += gd[k]
            grad["t2m"][m] += gd["t2m"]
            grad["precip"][m] += gd["precip"] * p.rho_w   # d/d precip = d/dP * rho_w
            grad["insol"][m] += gd["insol"]
            grad["t_base"] += gd["t_base"]
            grad["debris"] += gd["debris"]
        # M_prev = start-of-month state; d smb / d M_prev = -inv (couples to month s-1).
        bar_M += -g_smb[m] * inv
    return grad


# --------------------------------------------------------------------------- #
# GPU driver: the enthalpy sibling of ImprovedTemperatureIndex.  All state,
# parameters and CUDA operators live on the EnthalpyGrid; this is a thin driver.
# --------------------------------------------------------------------------- #
class EnthalpyModel:
    """Snowpack enthalpy core over an :class:`~glare.grid.EnthalpyGrid`.

    A thin driver, symmetric with :class:`~glare.model.ImprovedTemperatureIndex`.
    The grid owns everything: the forcing (``temperature.t2m`` [degC],
    ``precipitation.precip`` [m a-1 w.e.], ``radiation.insol_mean`` [fraction of
    direct sun], the static ``geometry.t_base`` [degC], and the static
    ``geometry.debris`` melt-attenuation field for snow-free ice, 1 = clean),
    the parameters (as grid ``Constant``s), and the per-month, calendar-ordered
    output state
    ``grid.state.{smb, M, E, runoff, ice_melt, t_surface, albedo}``.  ``smb`` is the
    glacier product; summing ``smb*dt`` over the water year gives the annual surface
    mass balance.  The recurrence is a single water-year sweep from a cold, snow-free
    surface (matching :func:`run_column` and ``cuda/smb.cu``'s water-year convention).

    Constructed like :class:`~glare.model.ImprovedTemperatureIndex`: pass a prebuilt
    ``grid``, or the grid dimensions and the model builds its own
    :class:`~glare.grid.EnthalpyGrid` (then populate ``model.grid`` externally).

    Parameters
    ----------
    grid
        An :class:`~glare.grid.EnthalpyGrid`; supplies the forcing, parameters,
        output state, ``n_substeps``/``glacier_surface`` config and the grid-cached
        ``Enthalpy{Forward,Backward}Operators``.  If ``None``, a grid is built from
        ``ny``/``nx``/``nt``/``dx``/``dt`` (+ the optional grid kwargs below).
    ny, nx, nt, dx, dt, x0, y0, crs, start_month, n_substeps, glacier_surface, use_fast_math
        Grid-construction arguments used when ``grid is None`` (see
        :class:`~glare.grid.EnthalpyGrid`).
    avalanche
        Optional :class:`~glare.avalanche.AvalancheOperator` (or ``None``).  When set,
        the raw total precip is redistributed downslope (reading ``grid.geometry.srf``)
        before the recurrence, and the adjoint pulls the precip gradient back through
        ``R^T`` -- mirroring ETIM.  No snow/rain partition is applied (the enthalpy
        kernel re-derives rain/snow per sub-step at the destination temperature).
        Usually set post-construction: ``enth.avalanche = AvalancheOperator(enth.grid, ...)``.
    seed
        Seed for the internal ``numpy`` generator used to draw the spatially-uniform
        sub-step temperature deviations when none are supplied to :meth:`forward`.

    Notes
    -----
    The temperature deviations are **spatially uniform** -- one ``(nt, n_substeps)``
    sequence shared by every pixel (a domain-wide weather realisation) held in
    ``grid.temp_dev``.  Per-pixel stochasticity would generalise it to
    ``(nt, n_substeps, ny, nx)``.
    """

    def __init__(self, grid=None, ny=None, nx=None, nt=None, dx=None, dt=None,
                 x0=cp.float32(0.0), y0=cp.float32(0.0), crs=None, start_month=9,
                 n_substeps=1, glacier_surface=True, use_fast_math=True,
                 materialize_state=True, avalanche=None, seed=None):
        if grid is not None:
            self.grid = grid
        elif ny and nx and nt and dx and dt:
            # materialize_state=False collapses the six diagnostic state
            # cubes (M, E, runoff, ice_melt, t_surface, albedo) onto one
            # shared scratch buffer — `smb` is unaffected. Recover them at
            # any time with grid.materialize_state_fields() + one forward().
            self.grid = EnthalpyGrid(ny, nx, nt, dx, dt, x0=x0, y0=y0, crs=crs,
                                     start_month=start_month, n_substeps=n_substeps,
                                     glacier_surface=glacier_surface,
                                     use_fast_math=use_fast_math,
                                     materialize_state=materialize_state)
        else:
            raise ValueError("provide either grid= or ny/nx/nt/dx/dt")

        # Optional avalanche redistribution operator (an AvalancheOperator or None).
        # When set, the raw precip is redistributed downslope (reading the grid DEM
        # grid.geometry.srf) before the recurrence; when None the effective precip is
        # a straight copy of the raw precip.  Mirrors ImprovedTemperatureIndex.
        self.avalanche = avalanche

        # The adjoint kernel re-derives each month's sub-step states in a fixed-size
        # thread-local array (GLARE_ENTH_NSUB_MAX in cuda/enthalpy.cu).
        if not 1 <= self.grid.n_substeps <= 64:
            raise ValueError("n_substeps must be in [1, 64] (adjoint replay bound)")
        self._rng = np.random.default_rng(seed)

    def forward(self, temp_deviations=None, rng=None):
        """Run the water-year enthalpy recurrence, filling ``grid.state``.

        Fills ``grid.temp_dev`` with the sub-step air-temperature deviations and
        ``grid.precipitation.precip_eff`` with the (optionally avalanche-redistributed)
        precip the kernel consumes, then launches the grid-cached forward operator.

        Parameters
        ----------
        temp_deviations
            Optional ``(nt, n_substeps)`` array of sub-step air-temperature
            deviations [degC].  If omitted and ``n_substeps > 1``, deviations are
            drawn from ``grid.temperature.sigma_t2m`` via :func:`generate_temp_deviations`
            (per-month demeaned) using ``rng`` (or the model's seeded generator).
            If ``n_substeps == 1`` and omitted, no perturbation is applied and the
            result is the plain monthly model.
        rng
            Optional ``numpy.random.Generator`` for this call; defaults to the
            model's seeded generator.
        """
        nt, n_sub = self.grid.nt, self.grid.n_substeps
        if temp_deviations is not None:
            dev = np.asarray(temp_deviations, dtype=np.float32)
            if dev.shape != (nt, n_sub):
                raise ValueError(
                    f"temp_deviations must have shape {(nt, n_sub)}, got {dev.shape}")
        elif n_sub == 1:
            dev = np.zeros((nt, 1), dtype=np.float32)
        else:
            sigma = float(cp.asnumpy(self.grid.temperature.sigma_t2m.value))
            dev = generate_temp_deviations(nt, n_sub, sigma, rng or self._rng)
        self.grid.temp_dev[...] = cp.asarray(dev, dtype=cp.float32)

        self._prepare_effective_precip()
        self.grid.forward_operators.compute_forward()

    def _prepare_effective_precip(self):
        """Redistribute the raw precip downslope (q = R * precip) before the
        recurrence.  When no avalanche operator is attached the effective
        precip IS the raw precip: the two fields are aliased onto one buffer
        (data and grad), saving two (nt, ny, nx) cubes and a memcpy per call
        instead of maintaining a byte-for-byte copy.  The aliasing follows
        `self.avalanche` (usually set post-construction) and is undone if an
        operator appears later."""
        pr = self.grid.precipitation.precip
        pe = self.grid.precipitation.precip_eff
        if self.avalanche is not None:
            if pe.data is pr.data:          # un-alias: operator attached later
                pe.data = cp.empty_like(pr.data)
            if pe._grad is not None and pe._grad is pr._grad:
                pe._grad = None             # R^T needs distinct grad buffers
            self.avalanche.forward(pr.data, pe.data)
        else:
            if pe.data is not pr.data:
                pe.data = pr.data
            # grad sharing is arranged in adjoint(), which owns the pull-back.

    def adjoint(self, grad_smb=None, grad_M=None, grad_E=None,
                grad_runoff=None, grad_ice_melt=None, wanted=None):
        """Reverse-mode gradient of the water-year recurrence (``compute_enthalpy_grad``).

        Linearises about the **most recent** :meth:`forward`'s ``grid.temp_dev``
        deviations, but re-derives the effective precip from the *currently-set*
        raw inputs (callers may have re-set forcing since the last forward --
        e.g. several :class:`~glare.torch.EnthalpyStep` nodes sharing one model
        inside a checkpoint segment).  Call ``forward()`` first.  Given the adjoint seeds (the gradient of a scalar loss with respect to
        each mass/energy output field, each ``(nt, ny, nx)``; ``None`` -> zeros),
        **writes the gradients directly into the grid** (like ETIM's adjoint): the
        forcing gradients into ``grid.temperature.t2m.grad``,
        ``grid.precipitation.precip.grad``, ``grid.radiation.insol_mean.grad``,
        ``grid.geometry.t_base.grad`` and ``grid.geometry.debris.grad``, and the
        seven energy-balance parameter gradients into the corresponding
        ``Constant.grad`` scalars (summed over the grid -- the loss is scalar).
        Returns ``None``.

        The pure diagnostics ``t_surface`` and ``albedo`` are not differentiated
        targets; seeding a loss on them is not supported (they do not feed the state).

        ``wanted`` (optional) names the forcing gradients the caller will
        actually consume, from {"t2m", "precip", "insol_mean", "t_base",
        "debris"}; ``None`` (the default) means all of them. The kernel
        computes every gradient regardless, but an unwanted **cube-sized**
        gradient is routed to one shared scratch buffer instead of lazily
        allocating a dedicated persistent (nt, ny, nx) ``.grad`` — mirrors
        the ``take()`` gating in :class:`~glare.torch.EnthalpyStep.backward`,
        which passes its ``needs_input_grad`` set here.
        """
        # The adjoint kernel replays the forward internally from precip_eff, so
        # rebuild it from the raw precip that is set *now* (see docstring).
        self._prepare_effective_precip()
        pr = self.grid.precipitation.precip
        pe = self.grid.precipitation.precip_eff
        want_precip = wanted is None or "precip" in wanted
        if self.avalanche is None and want_precip:
            # No redistribution: the effective-precip gradient IS the raw
            # gradient. Share one buffer so the kernel writes precip.grad
            # directly and the pull-back copy below becomes a no-op.
            if pe._grad is not pr.grad:
                pe._grad = pr.grad
        bo = self.grid.backward_operators
        # Seed buffers are lazy: outputs the caller did not seed share one
        # read-only zero cube instead of each holding a (nt, ny, nx) buffer —
        # an inversion only ever seeds smb, which saves four cubes of VRAM.
        seeds = {name: bo.set_seed(name, value) for name, value in (
            ("smb", grad_smb), ("M", grad_M), ("E", grad_E),
            ("runoff", grad_runoff), ("ice_melt", grad_ice_melt))}
        bo.compute_gradient(seeds=seeds, wanted=wanted)

        # The kernel wrote the gradient w.r.t. the *effective* precip; pull it
        # back to the raw precip through the avalanche transpose (R^T). With
        # no avalanche the buffers are shared above — nothing to copy; and an
        # unwanted precip gradient is left in the scratch sink untouched.
        if want_precip and self.avalanche is not None:
            self.avalanche.adjoint(pe.grad, pr.grad)
