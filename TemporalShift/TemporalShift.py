import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
so_file = script_dir / "tsm_cuda2.cpython-310-x86_64-linux-gnu.so"
torch.ops.load_library(str(so_file))

tsm_inplace = torch.ops.tsm2.tsm_inplace.default
#backward_ts = torch.ops.tsm2.backward_ts.default

#forward_ts_vect = torch.ops.tsm2.forward_ts_vect.default
#backward_ts_vect = torch.ops.tsm2.backward_ts_vect.default


class InplaceShift(Function):
    @staticmethod
    def forward(ctx, input, fold):
        if not input.is_cuda:
            raise RuntimeError("InplaceShift requires CUDA tensors")

        # Use the compiled wrapper which handles both the operation and autograd registration
        tsm_inplace(input, fold, 1)
        ctx.fold = fold
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        # we call the compiled backward directly if needed
        tsm_inplace(grad_output, ctx.fold, 0)
        return grad_output, None

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        print(f'Using fold div: {self.fold_div}')

    def forward(self, x):
        x = self.shift(x, self.n_segment, self.fold_div, self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=4, inplace=False):
        nt, c, h, w = x.size()
        fold = c // fold_div
        x = x.view(nt // n_segment, n_segment, c, h, w)

        if inplace:
            x = InplaceShift.apply(x, fold)
            return x.view(nt, c, h, w)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]  # shift right
            out[:, :, 2*fold:] = x[:, :, 2*fold:]  # no shift
            return out.view(nt, c, h, w)
