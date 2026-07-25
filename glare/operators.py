from pathlib import Path
import math
import cupy as cp


class ForwardOperators:
    def __init__(self,grid,
            use_fast_math=True):
        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['smb.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)        
    def compute_forward(self):
        block = (16, 16)
        grid = ((self.grid.nx + block[0] - 1) // block[0],
                (self.grid.ny + block[1] - 1) // block[1])

        smb_kernel = self.kernels.get_function('compute_smb')

        smb_kernel(grid, block, (
            self.grid.state.smb.data,
            self.grid.state.snow_depth.data,
            self.grid.insolation.insol_mean.data,
            self.grid.insolation.insol_cos.data,
            self.grid.insolation.insol_sin.data,
            self.grid.temperature.t2m.data,
            self.grid.precipitation.snowfall.data,
            self.grid.geometry.debris.data,
            self.grid.temperature.mf.value,
            self.grid.insolation.rf.value,
            self.grid.temperature.daily_amp_t2m.value,
            self.grid.temperature.sigma_t2m.value,
            self.grid.temperature.phi_0.value,
            self.grid.insolation.albedo_snow.value,
            self.grid.insolation.albedo_ice.value,
            self.grid.insolation.snow_transition_scale.value,
            cp.float32(self.grid.dt),
            self.grid.start_month,
            self.grid.ny, self.grid.nx, self.grid.nt
        ))


class BackwardOperators:
    def __init__(self,grid,
            use_fast_math=True):
        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['smb.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        self.grad_mf_pixel = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.grad_rf_pixel = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.grad_smb = cp.zeros((grid.nt,grid.ny,grid.nx),dtype=cp.float32)

    def compute_gradient(self):
        block = (16, 16)
        grid = ((self.grid.nx + block[0] - 1) // block[0],
                (self.grid.ny + block[1] - 1) // block[1])

        smb_kernel = self.kernels.get_function('compute_smb_grad')

        smb_kernel(grid, block, (
            self.grid.temperature.t2m.grad,
            self.grid.precipitation.snowfall.grad,
            self.grad_mf_pixel,
            self.grad_rf_pixel,
            self.grid.geometry.debris.grad,
            self.grad_smb,
            self.grid.insolation.insol_mean.data,
            self.grid.insolation.insol_cos.data,
            self.grid.insolation.insol_sin.data,
            self.grid.temperature.t2m.data,
            self.grid.precipitation.snowfall.data,
            self.grid.geometry.debris.data,
            self.grid.temperature.mf.value,
            self.grid.insolation.rf.value,
            self.grid.temperature.daily_amp_t2m.value,
            self.grid.temperature.sigma_t2m.value,
            self.grid.temperature.phi_0.value,
            self.grid.insolation.albedo_snow.value,
            self.grid.insolation.albedo_ice.value,
            self.grid.insolation.snow_transition_scale.value,
            cp.float32(self.grid.dt),
            self.grid.start_month,
            self.grid.ny, self.grid.nx, self.grid.nt
        ))

        self.grid.temperature.mf.grad = self.grad_mf_pixel.sum().item()
        self.grid.insolation.rf.grad = self.grad_rf_pixel.sum().item()


# =========================================================================== #
# Snowpack-enthalpy operators (cuda/enthalpy.cu), the grid-cached counterparts of
# the ETIM operators above.  They read every scalar from the EnthalpyGrid Constants,
# forcing from the grid fields, and the sub-step deviations / config from the grid,
# so the launch is arg-free just like compute_smb.
# =========================================================================== #

# The energy-balance parameters the adjoint differentiates (their Constant.grad is
# populated by EnthalpyBackwardOperators).  The thermodynamic constants are fixed.
GRAD_PARAM_NAMES = ("H_atm", "H_base0", "q_sw_bulk", "q_sw_insol",
                    "albedo_snow", "albedo_ice", "M_albedo")


def _inv_M_insulation(m_insulation):
    """Reciprocal insulation scale; 0 when infinite/invalid (constant H_base)."""
    m = float(m_insulation)
    if m and math.isfinite(m) and m > 0.0:
        return 1.0 / m
    return 0.0


class EnthalpyForwardOperators:
    def __init__(self, grid):
        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"
        cuda_files = ['enthalpy.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)

        options = ("--use_fast_math",) if grid.use_fast_math else ()
        self.kernels = cp.RawModule(code=cuda_source, options=options)

    def compute_forward(self):
        g = self.grid
        block = (16, 16)
        grid = ((g.nx + block[0] - 1) // block[0],
                (g.ny + block[1] - 1) // block[1])

        kernel = self.kernels.get_function('compute_enthalpy')

        st, th, rad, tm = g.state, g.thermodynamics, g.radiation, g.temperature
        f = cp.float32
        kernel(grid, block, (
            st.smb.data, st.M.data, st.E.data, st.runoff.data, st.ice_melt.data,
            st.t_surface.data, st.albedo.data,
            g.precipitation.precip_eff.data,   # redistributed precip (post-avalanche)
            tm.t2m.data,
            rad.insol_mean.data,
            g.geometry.t_base.data,
            g.temp_dev,
            th.L_f.value, th.c_i.value, th.c_w.value,
            th.H_atm.value, th.H_base0.value,
            rad.q_sw_bulk.value, rad.q_sw_insol.value,
            rad.albedo_snow.value, rad.albedo_ice.value, rad.M_albedo.value,
            tm.T_transition.value, f(_inv_M_insulation(th.M_insulation.value)),
            th.M_eps.value,
            th.rho_w.value, f(g.dt), cp.int32(g.glacier_surface),
            cp.int32(g.start_month),
            g.ny, g.nx, g.nt, cp.int32(g.n_substeps),
        ))


class EnthalpyBackwardOperators:
    def __init__(self, grid):
        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"
        cuda_files = ['enthalpy.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)

        options = ("--use_fast_math",) if grid.use_fast_math else ()
        self.kernels = cp.RawModule(code=cuda_source, options=options)

        # Adjoint seeds (d loss / d output), set by EnthalpyModel.adjoint().
        shape = (grid.nt, grid.ny, grid.nx)
        self.grad_smb = cp.zeros(shape, dtype=cp.float32)
        self.grad_M = cp.zeros(shape, dtype=cp.float32)
        self.grad_E = cp.zeros(shape, dtype=cp.float32)
        self.grad_runoff = cp.zeros(shape, dtype=cp.float32)
        self.grad_ice_melt = cp.zeros(shape, dtype=cp.float32)

        # Per-pixel parameter-gradient scratch, reduced to the Constant.grad scalars.
        self._pgrad = {k: cp.zeros((grid.ny, grid.nx), dtype=cp.float32)
                       for k in GRAD_PARAM_NAMES}

    def compute_gradient(self):
        g = self.grid
        block = (16, 16)
        grid = ((g.nx + block[0] - 1) // block[0],
                (g.ny + block[1] - 1) // block[1])

        kernel = self.kernels.get_function('compute_enthalpy_grad')

        st, th, rad, tm = g.state, g.thermodynamics, g.radiation, g.temperature

        # Forcing-gradient outputs land directly in the field .grad buffers (the
        # kernel overwrites every element).  Touch .grad to lazily allocate them.
        # The precip gradient is w.r.t. the *effective* precip the kernel read;
        # EnthalpyModel.adjoint pulls it back to precip.grad through R^T.
        grad_t2m = tm.t2m.grad
        grad_precip = g.precipitation.precip_eff.grad
        grad_insol = rad.insol_mean.grad
        grad_t_base = g.geometry.t_base.grad

        pg = self._pgrad
        f = cp.float32
        kernel(grid, block, (
            grad_t2m, grad_precip, grad_insol, grad_t_base,
            pg["H_atm"], pg["H_base0"], pg["q_sw_bulk"], pg["q_sw_insol"],
            pg["albedo_snow"], pg["albedo_ice"], pg["M_albedo"],
            self.grad_smb, self.grad_M, self.grad_E,
            self.grad_runoff, self.grad_ice_melt,
            g.precipitation.precip_eff.data,   # recompute forward from the effective precip
            tm.t2m.data,
            rad.insol_mean.data,
            g.geometry.t_base.data,
            g.temp_dev,
            th.L_f.value, th.c_i.value, th.c_w.value,
            th.H_atm.value, th.H_base0.value,
            rad.q_sw_bulk.value, rad.q_sw_insol.value,
            rad.albedo_snow.value, rad.albedo_ice.value, rad.M_albedo.value,
            tm.T_transition.value, f(_inv_M_insulation(th.M_insulation.value)),
            th.M_eps.value,
            th.rho_w.value, f(g.dt), cp.int32(g.glacier_surface),
            cp.int32(g.start_month),
            g.ny, g.nx, g.nt, cp.int32(g.n_substeps),
        ))

        # Reduce the per-pixel parameter gradients onto the Constant.grad scalars.
        targets = {
            "H_atm": th.H_atm, "H_base0": th.H_base0,
            "q_sw_bulk": rad.q_sw_bulk, "q_sw_insol": rad.q_sw_insol,
            "albedo_snow": rad.albedo_snow, "albedo_ice": rad.albedo_ice,
            "M_albedo": rad.M_albedo,
        }
        for name, const in targets.items():
            const.grad = float(pg[name].sum())



