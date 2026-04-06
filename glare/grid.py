from dataclasses import dataclass, field, fields
import cupy as cp

from glide.field import Field, TimeField, Constant, GridEntity
from glare.operators import ForwardOperators, BackwardOperators

@dataclass
class State:
    smb: TimeField | None = None

@dataclass
class Geometry:
    srf: Field | None = None

@dataclass
class Precipitation:
    precip: TimeField | None = None

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

class TIMGrid:

    def __init__(self,ny: int, nx: int, nt: int, 
            dx: cp.float32, dt: cp.float32,
            x0: cp.float32=cp.float32(0.0),
            y0: cp.float32=cp.float32(0.0),
            crs=None,
            state=None,
            geometry=None,
            precipitation=None,
            temperature=None,
            insolation=None):
        self.ny = ny
        self.nx = nx
        self.nt = nt
        self.dx = cp.float32(dx)
        self.dt = cp.float32(dt)

        self.x0 = cp.float32(x0)
        self.y0 = cp.float32(y0)
        self.crs = crs

        self.x_cell = cp.arange(x0,x0 + dx*nx, dx)
        self.y_cell = cp.arange(y0,y0 - dx*ny, -dx)

        self.state = state if state is not None else self._allocate_state()
        self.geometry = geometry if geometry is not None else self._allocate_geometry()
        self.precipitation = precipitation if precipitation is not None else self._allocate_precipitation()
        self.temperature = temperature if temperature is not None else self._allocate_temperature()
        self.insolation = insolation if insolation is not None else self._allocate_insolation()

        self._forward_operators = None
        self._backward_operators = None
    
    @property
    def forward_operators(self):
        if self._forward_operators is None:
            self._forward_operators = ForwardOperators(self)
        return self._forward_operators   

    @property
    def backward_operators(self):
        if self._backward_operators is None:
            self._backward_operators = BackwardOperators(self)
        return self._backward_operators    

    def _allocate_state(self):
        smb = TimeField(
            data = cp.zeros((self.nt,self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            dt=self.dt,
            grid=self,
            name='smb',
            units='m a^{-1}',
            attrs={'long_name':'monthly surface mass balance'})
        return State(smb=smb)

    def _allocate_geometry(self):
        srf = Field(
            data = cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='srf',
            units='m',
            attrs={'long_name':'surface elevation'})
        return Geometry(srf=srf)

    def _allocate_precipitation(self):
        precip = TimeField(
            data = cp.zeros((self.nt,self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            dt=self.dt,
            grid=self,
            name='precip',
            units='m a^{-1}',
            attrs={'long_name':'monthly total precipitation'})
        return Precipitation(precip=precip)

    def _allocate_temperature(self):
        t2m = TimeField(
            data = cp.zeros((self.nt,self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            dt=self.dt,
            grid=self,
            name='t2m',
            units='C',
            attrs={'long_name':'monthly mean 2 meter temperature'})
        return Temperature(t2m=t2m)

    def _allocate_insolation(self):
        insol_mean = TimeField(
            data = cp.zeros((self.nt,self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            dt=self.dt,
            grid=self,
            name='insol_mean',
            units='fraction of direct sunlight',
            attrs={'long_name':'monthly mean insolation'})

        insol_cos = TimeField(
            data = cp.zeros((self.nt,self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            dt=self.dt,
            grid=self,
            name='insol_cos',
            units='fraction of direct sunlight',
            attrs={'long_name':'Cosine component of fourier decomposition of diurnal temp cycle'})

        insol_sin = TimeField(
            data = cp.zeros((self.nt,self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            dt=self.dt,
            grid=self,
            name='insol_sin',
            units='fraction of direct sunlight',
            attrs={'long_name':'Sine component of fourier decomposition of diurnal temp cycle'})

        return Insolation(insol_mean=insol_mean,
                insol_cos=insol_cos,
                insol_sin=insol_sin)







