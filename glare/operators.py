from pathlib import Path
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
            self.grid.insolation.insol_mean.data, 
            self.grid.insolation.insol_cos.data,
            self.grid.insolation.insol_sin.data, 
            self.grid.temperature.t2m.data, 
            self.grid.precipitation.precip.data,
            self.grid.temperature.mf.value,
            self.grid.insolation.rf.value,
            self.grid.temperature.daily_amp_t2m.value,
            self.grid.temperature.sigma_t2m.value,
            self.grid.temperature.phi_0.value,
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
            self.grid.precipitation.precip.grad,
            self.grad_mf_pixel,
            self.grad_rf_pixel,
            self.grad_smb,
            self.grid.insolation.insol_mean.data, 
            self.grid.insolation.insol_cos.data,
            self.grid.insolation.insol_sin.data, 
            self.grid.temperature.t2m.data, 
            self.grid.precipitation.precip.data,
            self.grid.temperature.mf.value,
            self.grid.insolation.rf.value,
            self.grid.temperature.daily_amp_t2m.value,
            self.grid.temperature.sigma_t2m.value,
            self.grid.temperature.phi_0.value,
            self.grid.ny, self.grid.nx, self.grid.nt
        ))

        self.grid.temperature.mf.grad = self.grad_mf_pixel.sum().item()
        self.grid.insolation.rf.grad = self.grad_rf_pixel.sum().item()



