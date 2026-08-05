"""PyTorch autograd wrappers around the GLARE cores.

Each ``torch.autograd.Function`` sets the model's differentiable inputs from torch
tensors, runs ``model.forward()``, returns ``smb`` as a tensor, and in ``backward``
runs ``model.adjoint(...)`` and reads the gradients back off the grid.  ``GlareStep``
wraps the temperature-index core; ``EnthalpyStep`` wraps the snowpack-enthalpy core.
Inputs/outputs are CUDA tensors (the forward returns a cupy-backed ``smb``).
"""
import torch
import cupy as cp

class GlareStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx,model,t2m,precip,mf,rf,debris):
        ctx.model = model
        model.grid.temperature.t2m.set(cp.asarray(t2m.data))
        model.grid.precipitation.precip.set(cp.asarray(precip.data))
        model.grid.temperature.mf.set(cp.float32(mf.item()))
        model.grid.insolation.rf.set(cp.float32(rf.item()))
        model.grid.geometry.debris.set(cp.asarray(debris.data))

        model.forward()
        ctx.save_for_backward(t2m,precip,mf,rf,debris)
        ctx.model = model
        return torch.tensor(model.grid.state.smb.data)

    @staticmethod
    def backward(ctx,grad_smb):
        t2m,precip,mf,rf,debris = ctx.saved_tensors
        model = ctx.model

        model.grid.temperature.t2m.set(cp.asarray(t2m.data))
        model.grid.precipitation.precip.set(cp.asarray(precip.data))
        model.grid.temperature.mf.set(cp.float32(mf.item()))
        model.grid.insolation.rf.set(cp.float32(rf.item()))
        model.grid.geometry.debris.set(cp.asarray(debris.data))

        model.adjoint(cp.asarray(grad_smb))

        need = ctx.needs_input_grad

        # Copy a gradient off the grid only when torch will consume it (see
        # EnthalpyStep.backward — same transient-VRAM consideration).
        def take(idx, value):
            return torch.tensor(value) if need[idx] else None

        g_t2m = take(1, model.grid.temperature.t2m.grad)
        g_precip = take(2, model.grid.precipitation.precip.grad)
        g_mf = take(3, model.grid.temperature.mf.grad)
        g_rf = take(4, model.grid.insolation.rf.grad)
        g_debris = take(5, model.grid.geometry.debris.grad)

        return None, g_t2m, g_precip, g_mf, g_rf, g_debris


class AvalancheStep(torch.autograd.Function):
    """Apply a fixed :class:`~glare.avalanche.AvalancheOperator` ``R`` to a
    ``(nt, ny, nx)`` precipitation tensor as a standalone autograd op
    (forward ``R``, backward ``R^T``).

    This exists so an application can *hoist* the redistribution out of the
    per-step SMB call: because ``R`` is a fixed linear map, ``R(c * P) = c *
    R(P)`` for any scalar ``c``, so when the per-step precip only varies by a
    scalar multiplier the operator can be applied once to ``P`` instead of
    inside every (checkpointed, re-executed) SMB evaluation. The caller is
    responsible for the validity of that assumption — see the loud discussion
    on ``GlacierConfig.avalanche_hoisted`` in the application layer; if the
    per-step precip ever varies by more than a scalar multiple, hoisting is
    silently wrong and the operator must go back inside the SMB call
    (``model.avalanche``).
    """

    @staticmethod
    def forward(ctx, op, precip):
        ctx.op = op
        raw = cp.ascontiguousarray(cp.asarray(precip.detach()))
        eff = cp.empty_like(raw)
        op.forward(raw, eff)
        return torch.tensor(eff)

    @staticmethod
    def backward(ctx, grad_eff):
        g_eff = cp.ascontiguousarray(cp.asarray(grad_eff))
        g_raw = cp.empty_like(g_eff)
        ctx.op.adjoint(g_eff, g_raw)
        return None, torch.tensor(g_raw)


class EnthalpyStep(torch.autograd.Function):
    """Autograd wrapper for the snowpack-enthalpy core (mirrors :class:`GlareStep`).

    Differentiable inputs (the full set the enthalpy adjoint produces gradients for):
    the forcing fields ``t2m``, ``precip``, ``insol_mean`` (all ``(nt, ny, nx)``) and
    ``t_base`` (``(ny, nx)``), the seven energy-balance parameters
    (``H_atm``, ``H_base0``, ``q_sw_bulk``, ``q_sw_insol``, ``albedo_snow``,
    ``albedo_ice``, ``M_albedo``) as scalar tensors, and the optional ``debris``
    melt-attenuation field (``(ny, nx)``, 1 = clean ice; ``None`` sets the grid to
    clean ice, mirroring how :class:`GlareStep` carries ETIM's debris field).
    Returns ``smb`` (``(nt, ny, nx)``).

    Other configuration (``n_substeps``, ``glacier_surface``, the DEM ``geometry.srf``
    and an optional ``model.avalanche``) is set on the model/grid beforehand, exactly as
    the ETIM insolation geometry is preset for :class:`GlareStep`.  ``backward`` re-sets
    the saved inputs and calls ``model.adjoint`` -- it does not re-run ``forward`` (that
    would redraw the stochastic ``temp_dev``), so it linearises about the same forward.

    The optional trailing ``temp_deviations`` (a ``(nt, n_substeps)`` array, not a
    differentiable input) is forwarded to ``model.forward``.  Pass a fixed array when the
    step runs under ``torch.utils.checkpoint`` -- the checkpointed re-execution during
    backprop must reproduce the same weather realisation the outer graph saw, which an
    unseeded redraw would not.
    """

    @staticmethod
    def forward(ctx, model, t2m, precip, insol_mean, t_base,
                H_atm, H_base0, q_sw_bulk, q_sw_insol,
                albedo_snow, albedo_ice, M_albedo,
                debris=None, temp_deviations=None):
        EnthalpyStep._set_inputs(model, t2m, precip, insol_mean, t_base,
                                 H_atm, H_base0, q_sw_bulk, q_sw_insol,
                                 albedo_snow, albedo_ice, M_albedo, debris)
        model.forward(temp_deviations=temp_deviations)
        ctx.model = model
        ctx.debris_was_tensor = torch.is_tensor(debris)
        saved = (t2m, precip, insol_mean, t_base,
                 H_atm, H_base0, q_sw_bulk, q_sw_insol,
                 albedo_snow, albedo_ice, M_albedo)
        if ctx.debris_was_tensor:
            saved = saved + (debris,)
        ctx.save_for_backward(*saved)
        return torch.tensor(model.grid.state.smb.data)

    @staticmethod
    def backward(ctx, grad_smb):
        (t2m, precip, insol_mean, t_base, H_atm, H_base0, q_sw_bulk,
         q_sw_insol, albedo_snow, albedo_ice, M_albedo,
         *rest) = ctx.saved_tensors
        debris = rest[0] if ctx.debris_was_tensor else None
        model = ctx.model

        EnthalpyStep._set_inputs(model, t2m, precip, insol_mean, t_base,
                                 H_atm, H_base0, q_sw_bulk, q_sw_insol,
                                 albedo_snow, albedo_ice, M_albedo, debris)

        need = ctx.needs_input_grad
        # Tell the adjoint which forcing gradients torch will consume, so an
        # unwanted (nt, ny, nx) gradient goes to a shared sink instead of a
        # persistent per-field buffer (the grid-side twin of take() below).
        wanted = {name for idx, name in
                  ((1, "t2m"), (2, "precip"), (3, "insol_mean"), (4, "t_base"))
                  if need[idx]}
        if ctx.debris_was_tensor and need[12]:
            wanted.add("debris")
        model.adjoint(grad_smb=cp.asarray(grad_smb), wanted=wanted)

        g = model.grid

        # The fused adjoint kernel computes every gradient regardless, but only
        # copy one off the grid when torch will actually consume it — under the
        # inversion's checkpointed call only precip and two scalars require
        # grad, and skipping the (nt, ny, nx) t2m/insol copies trims the
        # backward's transient peak substantially on large domains.
        def take(idx, value):
            return torch.tensor(value) if need[idx] else None

        g_t2m = take(1, g.temperature.t2m.grad)
        g_precip = take(2, g.precipitation.precip.grad)
        g_insol = take(3, g.radiation.insol_mean.grad)
        g_t_base = take(4, g.geometry.t_base.grad)
        g_H_atm = take(5, g.thermodynamics.H_atm.grad)
        g_H_base0 = take(6, g.thermodynamics.H_base0.grad)
        g_q_sw_bulk = take(7, g.radiation.q_sw_bulk.grad)
        g_q_sw_insol = take(8, g.radiation.q_sw_insol.grad)
        g_albedo_snow = take(9, g.radiation.albedo_snow.grad)
        g_albedo_ice = take(10, g.radiation.albedo_ice.grad)
        g_M_albedo = take(11, g.radiation.M_albedo.grad)
        g_debris = (take(12, g.geometry.debris.grad)
                    if ctx.debris_was_tensor else None)

        # Trailing None is the (non-differentiable) temp_deviations slot; torch trims
        # trailing None gradients when the caller omitted the optional arguments.
        return (None, g_t2m, g_precip, g_insol, g_t_base,
                g_H_atm, g_H_base0, g_q_sw_bulk, g_q_sw_insol,
                g_albedo_snow, g_albedo_ice, g_M_albedo, g_debris, None)

    @staticmethod
    def _set_inputs(model, t2m, precip, insol_mean, t_base,
                    H_atm, H_base0, q_sw_bulk, q_sw_insol,
                    albedo_snow, albedo_ice, M_albedo, debris=None):
        g = model.grid
        g.temperature.t2m.set(cp.asarray(t2m.data))
        g.precipitation.precip.set(cp.asarray(precip.data))
        g.radiation.insol_mean.set(cp.asarray(insol_mean.data))
        g.geometry.t_base.set(cp.asarray(t_base.data))
        if debris is None:
            g.geometry.debris.set(1.0)              # clean ice, no attenuation
        else:
            g.geometry.debris.set(cp.asarray(debris.data))
        g.thermodynamics.H_atm.set(cp.float32(H_atm.item()))
        g.thermodynamics.H_base0.set(cp.float32(H_base0.item()))
        g.radiation.q_sw_bulk.set(cp.float32(q_sw_bulk.item()))
        g.radiation.q_sw_insol.set(cp.float32(q_sw_insol.item()))
        g.radiation.albedo_snow.set(cp.float32(albedo_snow.item()))
        g.radiation.albedo_ice.set(cp.float32(albedo_ice.item()))
        g.radiation.M_albedo.set(cp.float32(M_albedo.item()))




