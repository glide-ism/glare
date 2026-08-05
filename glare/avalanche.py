"""Avalanche snow-redistribution operator.

A mass-conserving multiple-flow-direction (MFD) routing operator that relocates
solid precipitation off steep, ice-free accumulation terrain onto downslope
deposition zones *before* melt is applied, so the relocated snow ablates at its
destination's PDD/radiation.

The terrain is static, so the operator is a fixed linear map ``q = R s`` whose
adjoint is exactly ``R^T``.  It is applied once per forcing update (to each monthly
slab of the already-partitioned solid-precip field), not per timestep.

This module also provides the snow/rain ``partition`` step, which used to live
inside the SMB kernel.  Moving it out lets the avalanche operator act on the solid
fraction directly: avalanched snow stays snow at its (warmer) destination instead of
being re-classified as rain there.

Differentiation in v1 is with respect to the snow field only; the deposition
parameters ``theta = (s_crit, w_trans, p)`` are fixed scalars set at construction.
"""

from pathlib import Path
import cupy as cp
from cupyx.scipy.special import erf

# Match the device-side approximations in cuda/smb.cu exactly so that with the
# avalanche operator disabled (R = I) the partition reproduces the original
# in-kernel behaviour.
_INV_SQRT2 = 0.7071    # 1/sqrt(2), as written in smb.cu's Phi()
_PHI_NORM = 0.3989     # 1/sqrt(2*pi), as written in smb.cu's phi()


def _phi_cap(z):
    """Standard-normal CDF, matching smb.cu's ``Phi``."""
    return 0.5 * (1.0 + erf(z * _INV_SQRT2))


def _phi_pdf(z):
    """Standard-normal PDF, matching smb.cu's ``phi``."""
    return _PHI_NORM * cp.exp(-0.5 * z * z)


def partition_forward(t2m, precip, sigma, out_raw):
    """Solid fraction of precipitation: ``raw = (1 - Phi(T/sigma)) * precip``.

    ``t2m``, ``precip`` and ``out_raw`` are ``(nt, ny, nx)`` cupy arrays; ``sigma``
    is a scalar.  Writes the solid (snow) precip into ``out_raw`` in place.
    """
    inv_sigma = cp.float32(1.0) / cp.float32(sigma)
    z = t2m * inv_sigma
    phiz = _phi_cap(z)
    out_raw[...] = ((1.0 - phiz) * precip).astype(cp.float32)


def partition_adjoint(t2m, precip, sigma, grad_raw, grad_precip, grad_t2m_acc):
    """Adjoint of :func:`partition_forward`.

    Given ``grad_raw`` (the adjoint of the solid-precip field), writes
    ``grad_precip = (1 - Phi(z)) * grad_raw`` and
    ``grad_t2m_acc = -precip * phi(z) * (1/sigma) * grad_raw`` (the accumulation
    contribution to the temperature gradient that used to be computed inside the
    SMB kernel).  All arrays are ``(nt, ny, nx)``; written in place.
    """
    inv_sigma = cp.float32(1.0) / cp.float32(sigma)
    z = t2m * inv_sigma
    phiz = _phi_cap(z)
    pdf = _phi_pdf(z)
    grad_precip[...] = ((1.0 - phiz) * grad_raw).astype(cp.float32)
    grad_t2m_acc[...] = (-precip * pdf * inv_sigma * grad_raw).astype(cp.float32)


class AvalancheOperator:
    """Fixed linear MFD redistribution operator ``q = R s`` and its adjoint ``R^T``.

    Routing is recomputed in-kernel from the DEM (``grid.geometry.srf``) on every
    pass, so the only state held here is the deposition parameters, the pass count
    and a few scratch buffers.

    Parameters
    ----------
    grid
        A ``TIMGrid``; supplies ``ny``, ``nx``, ``nt``, ``dx`` and the DEM.
    s_crit
        Critical (full-deposition) slope in degrees -- the deposition sigmoid centre.
    w_trans
        Transition width in degrees -- the deposition sigmoid sharpness.
    p
        MFD weighting exponent (flow concentration; ~1-2).
    K
        Number of fixed routing passes (Neumann-series truncation).  At 100 m
        posting K ~ 30-50 covers any plausible runout with negligible residual.
    """

    def __init__(self, grid, s_crit=30.0, w_trans=8.0, p=1.5, K=40,
                 use_fast_math=True):
        self.grid = grid
        self.s_crit = float(s_crit)
        self.w_trans = float(w_trans)
        self.p = float(p)
        self.K = int(K)

        cuda_dir = Path(__file__).parent / "cuda"
        cuda_source = (cuda_dir / "avalanche.cu").read_text()
        options = ("--use_fast_math",) if use_fast_math else ()
        self.kernels = cp.RawModule(code=cuda_source, options=options)
        self._scatter = self.kernels.get_function("avalanche_scatter")
        self._gather = self.kernels.get_function("avalanche_gather")

        ny, nx = grid.ny, grid.nx
        self._mobile_in = cp.zeros((ny, nx), dtype=cp.float32)
        self._mobile_out = cp.zeros((ny, nx), dtype=cp.float32)
        self._q = cp.zeros((ny, nx), dtype=cp.float32)
        self._bar_in = cp.zeros((ny, nx), dtype=cp.float32)
        self._bar_out = cp.zeros((ny, nx), dtype=cp.float32)

    @property
    def _launch(self):
        block = (16, 16)
        grid = ((self.grid.nx + block[0] - 1) // block[0],
                (self.grid.ny + block[1] - 1) // block[1])
        return grid, block

    @property
    def _theta_args(self):
        return (cp.float32(self.grid.dx), cp.float32(self.s_crit),
                cp.float32(self.w_trans), cp.float32(self.p))

    def forward(self, raw, eff):
        """Redistribute ``raw`` -> ``eff`` (both ``(nt, ny, nx)`` cupy arrays).

        Per slab: ``q = sum_{k<K} D A^k s + A^K s`` via K scatter passes followed by
        deposition of the residual mobile snow (so mass is conserved to FP).
        """
        grid, block = self._launch
        srf = self.grid.geometry.srf.data
        for t in range(self.grid.nt):
            self._mobile_in[...] = raw[t]
            self._q.fill(0.0)
            for _ in range(self.K):
                self._mobile_out.fill(0.0)
                self._scatter(grid, block, (
                    srf, self._mobile_in, self._mobile_out, self._q,
                    *self._theta_args, self.grid.ny, self.grid.nx))
                self._mobile_in, self._mobile_out = self._mobile_out, self._mobile_in
            self._q += self._mobile_in        # deposit residual mobile snow
            eff[t] = self._q

    def adjoint(self, grad_eff, grad_raw):
        """Transpose ``R^T``: ``grad_eff`` -> ``grad_raw`` (both ``(nt, ny, nx)``).

        Per slab: ``grad_s = sum_{k<K} (A^T)^k D lambda + (A^T)^K lambda`` via K
        deterministic gather passes, with ``lambda = grad_eff`` injected each pass.
        """
        grid, block = self._launch
        srf = self.grid.geometry.srf.data
        for t in range(self.grid.nt):
            grad_q = cp.ascontiguousarray(grad_eff[t], dtype=cp.float32)
            self._bar_in[...] = grad_q
            for _ in range(self.K):
                self._gather(grid, block, (
                    srf, self._bar_in, self._bar_out, grad_q,
                    *self._theta_args, self.grid.ny, self.grid.nx))
                self._bar_in, self._bar_out = self._bar_out, self._bar_in
            grad_raw[t] = self._bar_in

    def residual(self, raw):
        """Total mobile snow still in transit after K passes (truncation diagnostic).

        Returns the summed magnitude of ``A^K s`` over all slabs -- the mass that the
        residual-deposit step places at a (possibly) wrong location.
        """
        grid, block = self._launch
        srf = self.grid.geometry.srf.data
        total = 0.0
        for t in range(self.grid.nt):
            self._mobile_in[...] = raw[t]
            self._q.fill(0.0)
            for _ in range(self.K):
                self._mobile_out.fill(0.0)
                self._scatter(grid, block, (
                    srf, self._mobile_in, self._mobile_out, self._q,
                    *self._theta_args, self.grid.ny, self.grid.nx))
                self._mobile_in, self._mobile_out = self._mobile_out, self._mobile_in
            total += float(self._mobile_in.sum())
        return total
