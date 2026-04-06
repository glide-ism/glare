import torch
import cupy as cp

class GlareStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx,model,t2m,precip,mf,rf):
        ctx.model = model
        model.grid.temperature.t2m.set(cp.asarray(t2m.data))
        model.grid.precipitation.precip.set(cp.asarray(precip.data))
        model.grid.temperature.mf.set(cp.float32(mf.item()))
        model.grid.insolation.rf.set(cp.float32(rf.item()))

        model.forward()
        ctx.save_for_backward(t2m,precip,mf,rf)
        ctx.model = model 
        return torch.tensor(model.grid.state.smb.data)

    @staticmethod
    def backward(ctx,grad_smb):
        t2m,precip,mf,rf = ctx.saved_tensors
        model = ctx.model

        model.grid.temperature.t2m.set(cp.asarray(t2m.data))
        model.grid.precipitation.precip.set(cp.asarray(precip.data))
        model.grid.temperature.mf.set(cp.float32(mf.item()))
        model.grid.insolation.rf.set(cp.float32(rf.item()))

        model.backward(cp.asarray(grad_smb))

        g_t2m = torch.tensor(model.grid.temperature.t2m.grad)
        g_precip = torch.tensor(model.grid.precipitation.precip.grad)
        g_mf = torch.tensor(model.grid.temperature.mf.grad)
        g_rf = torch.tensor(model.grid.insolation.rf.grad)

        return None, g_t2m, g_precip, g_mf, g_rf




