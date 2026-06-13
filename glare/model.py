import cupy as cp
from .grid import TIMGrid
from .avalanche import partition_forward, partition_adjoint

class ImprovedTemperatureIndex:
    def __init__(self,grid=None,
            ny=None,nx=None,nt=None,
            dx=None,dt=None,
            x0=cp.float32(0.0),y0=cp.float32(0.0),crs=None,
            avalanche=None):
        if grid is not None:
            self.grid = grid
        elif ny and nx and nt and dx and dt:
            self.grid = TIMGrid(ny,nx,nt,
                dx,dt,
                x0=x0,y0=y0,
                crs=crs)

        # Optional avalanche redistribution operator (an AvalancheOperator or None).
        # When None the snow/rain partition is still applied, but R = I.
        self.avalanche = avalanche

        # Scratch buffers for the partition <-> avalanche pipeline.
        shape = (self.grid.nt, self.grid.ny, self.grid.nx)
        self._snowfall_raw = cp.zeros(shape, dtype=cp.float32)
        self._grad_raw = cp.zeros(shape, dtype=cp.float32)
        self._grad_t2m_acc = cp.zeros(shape, dtype=cp.float32)

    @property
    def _sigma(self):
        return self.grid.temperature.sigma_t2m.value

    def forward(self):
        # Partition total precip into its solid fraction, then (optionally)
        # redistribute it downslope, before the SMB kernel sees it.  This happens
        # once per forcing update, not per timestep.
        partition_forward(
            self.grid.temperature.t2m.data,
            self.grid.precipitation.precip.data,
            self._sigma,
            self._snowfall_raw,
        )
        snowfall = self.grid.precipitation.snowfall.data
        if self.avalanche is not None:
            self.avalanche.forward(self._snowfall_raw, snowfall)
        else:
            snowfall[...] = self._snowfall_raw

        self.grid.forward_operators.compute_forward()

    def backward(self,dJdsmb=None):
        if dJdsmb is not None:
            self.grid.backward_operators.grad_smb[:,:,:] = dJdsmb
        else:
            self.grid.backward_operators.grad_smb.fill(0.0)

        # Kernel adjoint: fills snowfall.grad (= grad of effective solid precip) and
        # t2m.grad with the *melt-only* temperature sensitivity.
        self.grid.backward_operators.compute_gradient()

        # Pull grad back through the avalanche operator (R^T), then through the
        # snow/rain partition to recover grad w.r.t. precip and the accumulation
        # contribution to grad w.r.t. t2m.
        grad_snowfall = self.grid.precipitation.snowfall.grad
        if self.avalanche is not None:
            self.avalanche.adjoint(grad_snowfall, self._grad_raw)
        else:
            self._grad_raw[...] = grad_snowfall

        partition_adjoint(
            self.grid.temperature.t2m.data,
            self.grid.precipitation.precip.data,
            self._sigma,
            self._grad_raw,
            self.grid.precipitation.precip.grad,
            self._grad_t2m_acc,
        )
        # Combine the melt (kernel) and accumulation (partition) T-sensitivities.
        self.grid.temperature.t2m.grad[...] += self._grad_t2m_acc
