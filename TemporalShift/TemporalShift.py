import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
so_file = script_dir / "tsm_cuda2.cpython-310-x86_64-linux-gnu.so"
torch.ops.load_library(str(so_file))

tsm_inplace = torch.ops.tsm2.tsm_inplace.default


class InplaceShift(Function):
    @staticmethod
    def forward(ctx, input, fold, vect):
        if not input.is_cuda:
            raise RuntimeError("InplaceShift requires CUDA tensors")

        if vect:
            (h,w) = input.size()[-2:]
            if h*w % 4 != 0:
                vect = False # ensures that vect=True is only for cases h*w % 4 == 0

        if vect:
            #vectorized
            tsm_inplace(input, fold, 1, 1)
            ctx.vect = 1
        else:
            #none-vectorized
            tsm_inplace(input, fold, 1, 0)
            ctx.vect = 0

        ctx.fold = fold
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        if ctx.vect == 1:
            #vectorized
            tsm_inplace(grad_output, ctx.fold, 0, 1)
        else:
            #none-vectorized
            tsm_inplace(grad_output, ctx.fold, 0, 0)

        return grad_output, None, None

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False, vect=False):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.vect = vect
        print(f'Using fold div: {self.fold_div}')

    def forward(self, x):
        x = self.shift(x, self.n_segment, self.fold_div, self.inplace, self.vect)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=4, inplace=False, vect=False):
        nt, c, h, w = x.size()
        fold = c // fold_div
        x = x.view(nt // n_segment, n_segment, c, h, w)

        if inplace:
            x = InplaceShift.apply(x, fold, vect)
            return x.view(nt, c, h, w)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]  # shift right
            out[:, :, 2*fold:] = x[:, :, 2*fold:]  # no shift
            return out.view(nt, c, h, w)
