import cupy as cp
from .grid import TIMGrid

class ImprovedTemperatureIndex:
    def __init__(self,grid=None,
            ny=None,nx=None,nt=None,
            dx=None,dt=None,
            x0=cp.float32(0.0),y0=cp.float32(0.0),crs=None):
        if grid is not None:
            self.grid = grid
        elif ny and nx and nt and dx and dt:
            self.grid = TIMGrid(ny,nx,nt,
                dx,dt,
                x0=x0,y0=y0,
                crs=crs)

    def forward(self):
        self.grid.forward_operators.compute_forward()

    def backward(self,dJdsmb=None):
        if dJdsmb is not None:
            self.grid.backward_operators.grad_smb[:,:,:] = dJdsmb
        else:
            self.grid.backward_operators.grad_smb.fill(0.0)

        self.grid.backward_operators.compute_gradient()



    

        

