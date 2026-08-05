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
            g.geometry.debris.data,
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
        # Allocated lazily: an inversion only ever seeds smb, and on large
        # domains each (nt, ny, nx) cube is substantial. Unseeded outputs share
        # the single read-only `zero_seed` cube at launch time.
        self._seed_shape = (grid.nt, grid.ny, grid.nx)
        self._seeds = {}
        self._zero_seed = None

        # Per-pixel parameter-gradient scratch, reduced to the Constant.grad scalars.
        self._pgrad = {k: cp.zeros((grid.ny, grid.nx), dtype=cp.float32)
                       for k in GRAD_PARAM_NAMES}

        # Write-only sink for cube-sized forcing gradients the caller does
        # not consume (see compute_gradient's `wanted`). Reuses the grid's
        # dematerialized-state scratch when one exists (forward and backward
        # launches never overlap), else allocated lazily.
        self._sink = None

    @property
    def seed_shape(self):
        return self._seed_shape

    def _seed_buffer(self, name):
        if name not in self._seeds:
            self._seeds[name] = cp.zeros(self._seed_shape, dtype=cp.float32)
        return self._seeds[name]

    # Per-output seed buffers, allocated on first touch (reading one of these
    # properties allocates it — use set_seed/zero_seed to avoid that).
    @property
    def grad_smb(self): return self._seed_buffer("smb")
    @property
    def grad_M(self): return self._seed_buffer("M")
    @property
    def grad_E(self): return self._seed_buffer("E")
    @property
    def grad_runoff(self): return self._seed_buffer("runoff")
    @property
    def grad_ice_melt(self): return self._seed_buffer("ice_melt")

    @property
    def zero_seed(self):
        """Shared all-zeros cube standing in for every unseeded output. Read
        only by contract: the kernel takes the seeds as const pointers and
        never writes them, so aliasing one cube across several seed arguments
        is safe (the __restrict__ qualifiers only matter for writes)."""
        if self._zero_seed is None:
            self._zero_seed = cp.zeros(self._seed_shape, dtype=cp.float32)
        return self._zero_seed

    def set_seed(self, name, value):
        """Resolve one adjoint seed to its launch buffer: the shared zero cube
        when `value` is None (no allocation), else the per-output buffer
        (allocated on first use) filled with `value`."""
        if value is None:
            return self.zero_seed
        value = cp.asarray(value, dtype=cp.float32)
        if value.shape != self._seed_shape:
            raise ValueError(
                f"seed must have shape {self._seed_shape}, got {value.shape}")
        buf = self._seed_buffer(name)
        buf[...] = value
        return buf

    def _sink_cube(self):
        scratch = getattr(self.grid, "_state_scratch", None)
        if scratch is not None:
            return scratch
        if self._sink is None:
            self._sink = cp.zeros(self._seed_shape, dtype=cp.float32)
        return self._sink

    def compute_gradient(self, seeds=None, wanted=None):
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
        #
        # `wanted` (None = all) names the forcing gradients the caller will
        # consume ("t2m", "precip", "insol_mean", "t_base", "debris"). The
        # kernel computes every gradient regardless, but an unwanted
        # cube-sized output is routed to one shared write-only sink instead
        # of lazily allocating a persistent (nt, ny, nx) .grad. Several sunk
        # outputs may alias the same sink: the cube-gradient kernel
        # parameters are deliberately not __restrict__ (see cuda/enthalpy.cu)
        # and are never read back. Plane-sized gradients (t_base, debris) are
        # cheap and always keep their own buffers; a field whose .grad
        # already exists keeps receiving it.
        def _cube_grad(field, name):
            if wanted is None or name in wanted or field.has_grad():
                return field.grad
            return self._sink_cube()

        grad_t2m = _cube_grad(tm.t2m, "t2m")
        grad_precip = _cube_grad(g.precipitation.precip_eff, "precip")
        grad_insol = _cube_grad(rad.insol_mean, "insol_mean")
        grad_t_base = g.geometry.t_base.grad
        grad_debris = g.geometry.debris.grad

        # `seeds` maps output name -> launch buffer (see EnthalpyModel.adjoint).
        # Without it, fall back to whatever per-output buffers exist, standing
        # in the shared zero cube for the rest — legacy callers that filled
        # e.g. `bo.grad_smb[...]` directly keep working without allocating the
        # other four cubes.
        if seeds is None:
            seeds = {name: self._seeds.get(name, self.zero_seed)
                     for name in ("smb", "M", "E", "runoff", "ice_melt")}

        pg = self._pgrad
        f = cp.float32
        kernel(grid, block, (
            grad_t2m, grad_precip, grad_insol, grad_t_base, grad_debris,
            pg["H_atm"], pg["H_base0"], pg["q_sw_bulk"], pg["q_sw_insol"],
            pg["albedo_snow"], pg["albedo_ice"], pg["M_albedo"],
            seeds["smb"], seeds["M"], seeds["E"],
            seeds["runoff"], seeds["ice_melt"],
            g.precipitation.precip_eff.data,   # recompute forward from the effective precip
            tm.t2m.data,
            rad.insol_mean.data,
            g.geometry.t_base.data,
            g.geometry.debris.data,
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



