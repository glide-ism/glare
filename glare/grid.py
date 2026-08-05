from dataclasses import dataclass, field
import cupy as cp

from glare.field import Field, TimeField, Constant, GridEntity
from glare.operators import (ForwardOperators, BackwardOperators,
                             EnthalpyForwardOperators, EnthalpyBackwardOperators)

# Seconds per (365-day) year: converts annual-rate energy/flux constants to SI.
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


# =========================================================================== #
# Field groups for the temperature-index (ETIM) core.
# =========================================================================== #
@dataclass
class State:
    smb: TimeField | None = None
    snow_depth: TimeField | None = None

@dataclass
class Geometry:
    srf: Field | None = None
    debris: Field | None = None

@dataclass
class Precipitation:
    precip: TimeField | None = None
    # Effective solid precipitation consumed by the SMB kernel: precip after the
    # snow/rain partition and (optionally) avalanche redistribution.  Its .grad
    # buffer holds grad_snowfall during the adjoint pass.
    snowfall: TimeField | None = None

@dataclass
class Temperature:
    t2m: TimeField | None = None
    mf: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(1.825),
            name='mf',
            units='m a^{-1} C^{-1}',
            attrs={'long_name':'positive temperature melt factor'})
        )

    daily_amp_t2m: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(5.0),
            name='daily_amp_t2m',
            units='C',
            attrs={'long_name':'diurnal temperature fluctuation'})
        )
    sigma_t2m: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(5.0),
            name='sigma_t2m',
            units='C',
            attrs={'long_name':'standard deviation of sub-time step temperature'})
        )
    phi_0: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(2*cp.pi*15.0/24.0),
            name='phi_0',
            units='',
            attrs={'long_name':'phase of daily temperature maximum (15:00 LST)'})
        )

@dataclass
class Insolation:
    insol_mean: TimeField | None = None
    insol_cos: TimeField | None = None
    insol_sin: TimeField | None = None
    rf: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(18.25),
            name='rf',
            units='m a^{-1} per maximum direct solar',
            attrs={'long_name':'melt rate due to direct full sunlight (modulated by albedo)'})
        )
    albedo_snow: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.8),
            name='albedo_snow',
            units='',
            attrs={'long_name':'broadband albedo of snow-covered surface'})
        )
    albedo_ice: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.4),
            name='albedo_ice',
            units='',
            attrs={'long_name':'broadband albedo of ice/bare surface'})
        )
    snow_transition_scale: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.1),
            name='snow_transition_scale',
            units='m',
            attrs={'long_name':'snow-depth (water equiv.) scale of the snow/ice albedo sigmoid'})
        )


# =========================================================================== #
# Field groups for the snowpack-enthalpy core.  Its parameters live here as grid
# Constants (the single source of truth), exactly as the ETIM parameters do above;
# the seven invertible ones (see glare.operators.GRAD_PARAM_NAMES) get their .grad
# populated by the adjoint.
# =========================================================================== #
@dataclass
class EnthalpyState:
    smb: TimeField | None = None
    M: TimeField | None = None
    E: TimeField | None = None
    runoff: TimeField | None = None
    ice_melt: TimeField | None = None
    t_surface: TimeField | None = None
    albedo: TimeField | None = None

@dataclass
class EnthalpyGeometry:
    # Static per-cell basal/substrate temperature forcing; its .grad is an adjoint
    # target (mirrors how ETIM's debris field carries a gradient).
    t_base: Field | None = None
    # Debris melt attenuation for snow-free ice (1 = clean, 0 = fully insulated):
    # multiplies every glacier-ice melt flux in the kernel.  Static per-cell,
    # adjoint target like t_base; initialized to 1 so unset grids are clean ice.
    debris: Field | None = None
    # Surface elevation: the routing DEM read by the AvalancheOperator (only used
    # when an avalanche operator is attached to the model).
    srf: Field | None = None

@dataclass
class EnthalpyPrecipitation:
    precip: TimeField | None = None
    # Effective precipitation consumed by the enthalpy kernel: the raw precip after
    # (optional) avalanche redistribution.  Its .grad buffer holds the effective-precip
    # adjoint, pulled back to precip.grad through R^T in EnthalpyModel.adjoint.
    precip_eff: TimeField | None = None

@dataclass
class EnthalpyTemperature:
    t2m: TimeField | None = None
    sigma_t2m: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(5.0),
            name='sigma_t2m',
            units='C',
            attrs={'long_name':'standard deviation of sub-time step temperature'})
        )
    T_transition: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(1.0),
            name='T_transition',
            units='C',
            attrs={'long_name':'rain/snow transition temperature scale'})
        )

@dataclass
class Thermodynamics:
    L_f: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(3.35e5), name='L_f', units='J kg^{-1}',
            attrs={'long_name':'latent heat of fusion'}))
    c_i: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(2.09e3), name='c_i', units='J kg^{-1} K^{-1}',
            attrs={'long_name':'specific heat of ice'}))
    c_w: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(4.184e3), name='c_w', units='J kg^{-1} K^{-1}',
            attrs={'long_name':'specific heat of water'}))
    H_atm: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(1.0*SECONDS_PER_YEAR), name='H_atm',
            units='J m^{-2} yr^{-1} K^{-1}',
            attrs={'long_name':'atmospheric heat-exchange coefficient'}))
    H_base0: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.6*SECONDS_PER_YEAR), name='H_base0',
            units='J m^{-2} yr^{-1} K^{-1}',
            attrs={'long_name':'bare-surface basal heat-exchange coefficient'}))
    M_insulation: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(200.0), name='M_insulation', units='kg m^{-2}',
            attrs={'long_name':'snow-insulation mass scale (inf -> constant H_base)'}))
    M_eps: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(1.0e-10), name='M_eps', units='kg m^{-2}',
            attrs={'long_name':'reservoir cleanup threshold'}))
    rho_w: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(1000.0), name='rho_w', units='kg m^{-3}',
            attrs={'long_name':'water density (m w.e. <-> kg m^{-2})'}))

@dataclass
class Radiation:
    insol_mean: TimeField | None = None
    q_sw_bulk: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(100.0*SECONDS_PER_YEAR), name='q_sw_bulk',
            units='J m^{-2} yr^{-1}',
            attrs={'long_name':'constant (pre-albedo) shortwave flux'}))
    q_sw_insol: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.0), name='q_sw_insol', units='J m^{-2} yr^{-1}',
            attrs={'long_name':'seasonal shortwave flux at full direct sun (x insolation)'}))
    albedo_snow: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.9), name='albedo_snow', units='',
            attrs={'long_name':'broadband albedo of snow-covered surface'}))
    albedo_ice: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.4), name='albedo_ice', units='',
            attrs={'long_name':'broadband albedo of ice/bare surface'}))
    M_albedo: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(20.0), name='M_albedo', units='kg m^{-2}',
            attrs={'long_name':'optical snow-cover mass scale'}))


# =========================================================================== #
# Agnostic base grid: coordinates, a generic output-state slot, lazily-cached
# forward/backward operators, and field-construction helpers.  Model-specific grids
# (TIMGrid, EnthalpyGrid) subclass this and add their forcing, parameter Constants,
# output state, and CUDA-operator factories.
# =========================================================================== #
class Grid:

    def __init__(self, ny: int, nx: int, nt: int,
                 dx: cp.float32, dt: cp.float32,
                 x0: cp.float32 = cp.float32(0.0),
                 y0: cp.float32 = cp.float32(0.0),
                 crs=None,
                 start_month: int = 9):
        self.ny = ny
        self.nx = nx
        self.nt = nt
        self.dx = cp.float32(dx)
        self.dt = cp.float32(dt)

        self.x0 = cp.float32(x0)
        self.y0 = cp.float32(y0)
        self.crs = crs
        self.start_month = int(start_month)   # 9 = October (Jan-indexed)

        self.x_cell = cp.arange(x0, x0 + dx*nx, dx)
        self.y_cell = cp.arange(y0, y0 - dx*ny, -dx)

        self.state = None                     # generic output slot (subclass fills)
        self._forward_operators = None
        self._backward_operators = None

    # --- lazily-cached CUDA operators (subclass provides the factories) --- #
    @property
    def forward_operators(self):
        if self._forward_operators is None:
            self._forward_operators = self._make_forward_operators()
        return self._forward_operators

    @property
    def backward_operators(self):
        if self._backward_operators is None:
            self._backward_operators = self._make_backward_operators()
        return self._backward_operators

    def _make_forward_operators(self):
        raise NotImplementedError

    def _make_backward_operators(self):
        raise NotImplementedError

    # --- field-construction helpers (shared by every subclass) --- #
    def _timefield(self, name, units, long_name, fill=0.0):
        return TimeField(
            data=cp.full((self.nt, self.ny, self.nx), fill, dtype=cp.float32),
            grid_entity=GridEntity.CELL, dx=self.dx, dt=self.dt, grid=self,
            name=name, units=units, attrs={'long_name': long_name})

    def _field(self, name, units, long_name, fill=0.0):
        return Field(
            data=cp.full((self.ny, self.nx), fill, dtype=cp.float32),
            grid_entity=GridEntity.CELL, dx=self.dx, grid=self,
            name=name, units=units, attrs={'long_name': long_name})


# =========================================================================== #
# Temperature-index (ETIM) grid.
# =========================================================================== #
class TIMGrid(Grid):

    def __init__(self, ny, nx, nt, dx, dt,
                 x0=cp.float32(0.0), y0=cp.float32(0.0), crs=None, start_month=9,
                 state=None, geometry=None, precipitation=None,
                 temperature=None, insolation=None):
        super().__init__(ny, nx, nt, dx, dt,
                         x0=x0, y0=y0, crs=crs, start_month=start_month)

        self.state = state if state is not None else self._allocate_state()
        self.geometry = geometry if geometry is not None else self._allocate_geometry()
        self.precipitation = precipitation if precipitation is not None else self._allocate_precipitation()
        self.temperature = temperature if temperature is not None else self._allocate_temperature()
        self.insolation = insolation if insolation is not None else self._allocate_insolation()

    def _make_forward_operators(self):
        return ForwardOperators(self)

    def _make_backward_operators(self):
        return BackwardOperators(self)

    def _allocate_state(self):
        return State(
            smb=self._timefield('smb', 'm a^{-1}', 'monthly surface mass balance'),
            snow_depth=self._timefield('snow_depth', 'm',
                                       'end-of-month snow depth (water equivalent)'))

    def _allocate_geometry(self):
        return Geometry(
            srf=self._field('srf', 'm', 'surface elevation'),
            debris=self._field('debris', '',
                               'debris-cover melt factor for snow-free ice '
                               '(1 = bare ice, <1 = insulated)', fill=1.0))

    def _allocate_precipitation(self):
        return Precipitation(
            precip=self._timefield('precip', 'm a^{-1}', 'monthly total precipitation'),
            snowfall=self._timefield('snowfall', 'm a^{-1}',
                                     'effective solid precipitation (partitioned, '
                                     'post-avalanche)'))

    def _allocate_temperature(self):
        return Temperature(
            t2m=self._timefield('t2m', 'C', 'monthly mean 2 meter temperature'))

    def _allocate_insolation(self):
        return Insolation(
            insol_mean=self._timefield('insol_mean', 'fraction of direct sunlight',
                                       'monthly mean insolation'),
            insol_cos=self._timefield('insol_cos', 'fraction of direct sunlight',
                                      'Cosine component of fourier decomposition of diurnal temp cycle'),
            insol_sin=self._timefield('insol_sin', 'fraction of direct sunlight',
                                      'Sine component of fourier decomposition of diurnal temp cycle'))


# =========================================================================== #
# Snowpack-enthalpy grid.  Sibling of TIMGrid: same base machinery, its own
# forcing, parameter Constants, and output state.  Sub-monthly stochastic forcing
# config (n_substeps, glacier_surface, use_fast_math) lives here so the grid-cached
# operators can launch arg-free; the model fills the temp_dev buffer.
# =========================================================================== #
class EnthalpyGrid(Grid):

    # State fields that are pure outputs of the forward kernel and are never
    # read back by the model (only `smb` leaves via the torch layer; the
    # adjoint replays from the inputs). With materialize_state=False these
    # share a single scratch cube instead of one (nt, ny, nx) buffer each.
    DIAGNOSTIC_STATE = ("M", "E", "runoff", "ice_melt", "t_surface", "albedo")

    def __init__(self, ny, nx, nt, dx, dt,
                 x0=cp.float32(0.0), y0=cp.float32(0.0), crs=None, start_month=9,
                 n_substeps=1, glacier_surface=True, use_fast_math=True,
                 materialize_state=True,
                 state=None, geometry=None, precipitation=None,
                 temperature=None, thermodynamics=None, radiation=None):
        super().__init__(ny, nx, nt, dx, dt,
                         x0=x0, y0=y0, crs=crs, start_month=start_month)

        self.n_substeps = int(n_substeps)
        self.glacier_surface = bool(glacier_surface)
        self.use_fast_math = bool(use_fast_math)
        # materialize_state=False aliases the six diagnostic state fields onto
        # one shared scratch cube (their contents are then interleaved garbage
        # — check `field.is_materialized` before reading). Toggle at runtime
        # with materialize_state_fields()/dematerialize_state_fields(); the
        # next forward() fills freshly materialized buffers.
        self.materialize_state = bool(materialize_state)

        self.state = state if state is not None else self._allocate_state()
        self.geometry = geometry if geometry is not None else self._allocate_geometry()
        self.precipitation = precipitation if precipitation is not None else self._allocate_precipitation()
        self.temperature = temperature if temperature is not None else self._allocate_temperature()
        self.thermodynamics = thermodynamics if thermodynamics is not None else Thermodynamics()
        self.radiation = radiation if radiation is not None else self._allocate_radiation()

        # Spatially-uniform sub-step temperature deviations [C], (nt, n_substeps).
        # Filled by EnthalpyModel.forward() before each launch.
        self.temp_dev = cp.zeros((self.nt, self.n_substeps), dtype=cp.float32)

    def _make_forward_operators(self):
        return EnthalpyForwardOperators(self)

    def _make_backward_operators(self):
        return EnthalpyBackwardOperators(self)

    _STATE_META = {
        "M": ('kg m^{-2}', 'active surface-reservoir mass'),
        "E": ('J m^{-2}', 'active surface-reservoir enthalpy (ref. ice at 0 C)'),
        "runoff": ('kg m^{-2}', 'meltwater/rain runoff from the pack this month'),
        "ice_melt": ('kg m^{-2}', 'glacier-ice melt this month'),
        "t_surface": ('C', 'reservoir/surface temperature'),
        "albedo": ('', 'broadband surface albedo'),
    }

    def _allocate_state(self):
        smb = self._timefield(
            'smb', 'm a^{-1}', 'monthly surface mass balance (water equivalent)')
        smb.is_materialized = True
        if self.materialize_state:
            diag = {name: self._timefield(name, units, long_name)
                    for name, (units, long_name) in self._STATE_META.items()}
            for field in diag.values():
                field.is_materialized = True
            self._state_scratch = None
        else:
            # One shared scratch cube stands in for all six diagnostic
            # outputs (kernel write-only targets; their contents interleave
            # to garbage). materialize_state_fields() gives them real
            # buffers on demand.
            scratch = cp.zeros((self.nt, self.ny, self.nx), dtype=cp.float32)
            diag = {}
            for name, (units, long_name) in self._STATE_META.items():
                field = TimeField(
                    data=scratch, grid_entity=GridEntity.CELL, dx=self.dx,
                    dt=self.dt, grid=self, name=name, units=units,
                    attrs={'long_name': long_name})
                field.is_materialized = False
                diag[name] = field
            self._state_scratch = scratch
        return EnthalpyState(smb=smb, **diag)

    def materialize_state_fields(self):
        """Give each diagnostic state field (M, E, runoff, ice_melt,
        t_surface, albedo) its own (nt, ny, nx) buffer again. The buffers are
        zero until the NEXT forward() fills them — run one forward before
        reading. No-op when already materialized."""
        if self.materialize_state:
            return
        for name in self.DIAGNOSTIC_STATE:
            field = getattr(self.state, name)
            field.data = cp.zeros((self.nt, self.ny, self.nx),
                                  dtype=cp.float32)
            field.is_materialized = True
        self._state_scratch = None
        self.materialize_state = True

    def dematerialize_state_fields(self):
        """Point the six diagnostic state fields at ONE shared scratch cube,
        freeing five (nt, ny, nx) buffers. Their contents become interleaved
        kernel-output garbage (`is_materialized` is False); `smb` keeps its
        own buffer and is unaffected. Safe because the kernels only ever
        WRITE these outputs — nothing reads them back. No-op when already
        dematerialized."""
        if not self.materialize_state:
            return
        scratch = cp.zeros((self.nt, self.ny, self.nx), dtype=cp.float32)
        for name in self.DIAGNOSTIC_STATE:
            field = getattr(self.state, name)
            field.data = scratch
            field.is_materialized = False
        self._state_scratch = scratch
        self.materialize_state = False

    def _allocate_geometry(self):
        geometry = EnthalpyGeometry(
            t_base=self._field('t_base', 'C', 'basal/substrate temperature'),
            debris=self._field('debris', '',
                               'debris melt attenuation for snow-free ice '
                               '(1 = clean)'),
            srf=self._field('srf', 'm', 'surface elevation'))
        geometry.debris.data[...] = 1.0   # default: clean ice, no attenuation
        return geometry

    def _allocate_precipitation(self):
        return EnthalpyPrecipitation(
            precip=self._timefield('precip', 'm a^{-1}', 'monthly total precipitation'),
            precip_eff=self._timefield('precip_eff', 'm a^{-1}',
                                       'effective precipitation (post-avalanche)'))

    def _allocate_temperature(self):
        return EnthalpyTemperature(
            t2m=self._timefield('t2m', 'C', 'monthly mean 2 meter temperature'))

    def _allocate_radiation(self):
        return Radiation(
            insol_mean=self._timefield('insol_mean', 'fraction of direct sunlight',
                                       'monthly mean insolation'))
