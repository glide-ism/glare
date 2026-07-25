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

        g_t2m = torch.tensor(model.grid.temperature.t2m.grad)
        g_precip = torch.tensor(model.grid.precipitation.precip.grad)
        g_mf = torch.tensor(model.grid.temperature.mf.grad)
        g_rf = torch.tensor(model.grid.insolation.rf.grad)
        g_debris = torch.tensor(model.grid.geometry.debris.grad)

        return None, g_t2m, g_precip, g_mf, g_rf, g_debris


class EnthalpyStep(torch.autograd.Function):
    """Autograd wrapper for the snowpack-enthalpy core (mirrors :class:`GlareStep`).

    Differentiable inputs (the full set the enthalpy adjoint produces gradients for):
    the forcing fields ``t2m``, ``precip``, ``insol_mean`` (all ``(nt, ny, nx)``) and
    ``t_base`` (``(ny, nx)``), plus the seven energy-balance parameters
    (``H_atm``, ``H_base0``, ``q_sw_bulk``, ``q_sw_insol``, ``albedo_snow``,
    ``albedo_ice``, ``M_albedo``) as scalar tensors.  Returns ``smb`` (``(nt, ny, nx)``).

    Other configuration (``n_substeps``, ``glacier_surface``, the DEM ``geometry.srf``
    and an optional ``model.avalanche``) is set on the model/grid beforehand, exactly as
    the ETIM insolation geometry is preset for :class:`GlareStep`.  ``backward`` re-sets
    the saved inputs and calls ``model.adjoint`` -- it does not re-run ``forward`` (that
    would redraw the stochastic ``temp_dev``), so it linearises about the same forward.
    """

    @staticmethod
    def forward(ctx, model, t2m, precip, insol_mean, t_base,
                H_atm, H_base0, q_sw_bulk, q_sw_insol,
                albedo_snow, albedo_ice, M_albedo):
        EnthalpyStep._set_inputs(model, t2m, precip, insol_mean, t_base,
                                 H_atm, H_base0, q_sw_bulk, q_sw_insol,
                                 albedo_snow, albedo_ice, M_albedo)
        model.forward()
        ctx.model = model
        ctx.save_for_backward(t2m, precip, insol_mean, t_base,
                              H_atm, H_base0, q_sw_bulk, q_sw_insol,
                              albedo_snow, albedo_ice, M_albedo)
        return torch.tensor(model.grid.state.smb.data)

    @staticmethod
    def backward(ctx, grad_smb):
        (t2m, precip, insol_mean, t_base, H_atm, H_base0, q_sw_bulk,
         q_sw_insol, albedo_snow, albedo_ice, M_albedo) = ctx.saved_tensors
        model = ctx.model

        EnthalpyStep._set_inputs(model, t2m, precip, insol_mean, t_base,
                                 H_atm, H_base0, q_sw_bulk, q_sw_insol,
                                 albedo_snow, albedo_ice, M_albedo)

        model.adjoint(grad_smb=cp.asarray(grad_smb))

        g = model.grid
        g_t2m = torch.tensor(g.temperature.t2m.grad)
        g_precip = torch.tensor(g.precipitation.precip.grad)
        g_insol = torch.tensor(g.radiation.insol_mean.grad)
        g_t_base = torch.tensor(g.geometry.t_base.grad)
        g_H_atm = torch.tensor(g.thermodynamics.H_atm.grad)
        g_H_base0 = torch.tensor(g.thermodynamics.H_base0.grad)
        g_q_sw_bulk = torch.tensor(g.radiation.q_sw_bulk.grad)
        g_q_sw_insol = torch.tensor(g.radiation.q_sw_insol.grad)
        g_albedo_snow = torch.tensor(g.radiation.albedo_snow.grad)
        g_albedo_ice = torch.tensor(g.radiation.albedo_ice.grad)
        g_M_albedo = torch.tensor(g.radiation.M_albedo.grad)

        return (None, g_t2m, g_precip, g_insol, g_t_base,
                g_H_atm, g_H_base0, g_q_sw_bulk, g_q_sw_insol,
                g_albedo_snow, g_albedo_ice, g_M_albedo)

    @staticmethod
    def _set_inputs(model, t2m, precip, insol_mean, t_base,
                    H_atm, H_base0, q_sw_bulk, q_sw_insol,
                    albedo_snow, albedo_ice, M_albedo):
        g = model.grid
        g.temperature.t2m.set(cp.asarray(t2m.data))
        g.precipitation.precip.set(cp.asarray(precip.data))
        g.radiation.insol_mean.set(cp.asarray(insol_mean.data))
        g.geometry.t_base.set(cp.asarray(t_base.data))
        g.thermodynamics.H_atm.set(cp.float32(H_atm.item()))
        g.thermodynamics.H_base0.set(cp.float32(H_base0.item()))
        g.radiation.q_sw_bulk.set(cp.float32(q_sw_bulk.item()))
        g.radiation.q_sw_insol.set(cp.float32(q_sw_insol.item()))
        g.radiation.albedo_snow.set(cp.float32(albedo_snow.item()))
        g.radiation.albedo_ice.set(cp.float32(albedo_ice.item()))
        g.radiation.M_albedo.set(cp.float32(M_albedo.item()))




