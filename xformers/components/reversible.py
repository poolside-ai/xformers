# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional

import torch
import torch.nn as nn
from torch.autograd.function import Function, once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.checkpoint import get_device_states, set_device_states

from xformers.components import RequiresWrappedInputs

# CREDITS: Code adapted from
# https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
# https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py,
# https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html


# pyre-fixme[13]: `cpu_state` is not initialized in the constructor.
class Deterministic(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.cpu_state: torch.Tensor = torch.get_rng_state()
        self.cuda_in_fwd: bool = False
        self.gpu_devices: List[int] = []
        self.gpu_states: List[torch.Tensor] = []
        self.wrap_inputs = isinstance(net, RequiresWrappedInputs)

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng: bool = False, set_rng: bool = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            # Normal FW run
            if self.wrap_inputs:
                return self.net(inputs=args, **kwargs)
            else:
                return self.net(*args, **kwargs)

        else:  # pragma: no cover  # this is called in the backward pass, not picked up
            # This is analogous to checkpointing, reset the original random state
            rng_devices: List[int] = []
            if self.cuda_in_fwd:
                rng_devices = self.gpu_devices

            with torch.random.fork_rng(devices=rng_devices, enabled=True):
                torch.set_rng_state(self.cpu_state)
                if self.cuda_in_fwd:
                    set_device_states(self.gpu_devices, self.gpu_states)

                if self.wrap_inputs:
                    return self.net(inputs=args, **kwargs)
                else:
                    return self.net(*args, **kwargs)


class ReversibleBlock(nn.Module):
    def __init__(self, f: nn.Module, g: nn.Module, split_dim: int):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.split_dim = split_dim

    @custom_fwd
    @torch.no_grad()
    def forward(self, x: torch.Tensor, f_args={}, g_args={}):
        assert x.dtype == torch.get_autocast_gpu_dtype()
        x1, x2 = torch.chunk(x, 2, dim=self.split_dim)
        x1.add_(self.f(x2, record_rng=self.training, **f_args))
        x2.add_(self.g(x1, record_rng=self.training, **g_args))
        return x

    @custom_bwd
    def backward_pass(
        self, y: torch.Tensor, dy: torch.Tensor, f_args={}, g_args={}
    ) -> None:  # pragma: no cover  # this is covered, but called directly from C++
        assert y.dtype == torch.get_autocast_gpu_dtype()
        # TODO: specify the buffer for gy1 and fy2, support output placement in .f and .g
        y1, y2 = torch.chunk(y, 2, dim=self.split_dim)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=self.split_dim)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            y2.sub_(gy1)
            del gy1

            dy1.add_(y1.grad)
            y1.grad.zero_()

        with torch.enable_grad():
            y2.requires_grad = True
            fy2 = self.f(y2, set_rng=True, **f_args)
            torch.autograd.backward(fy2, dy1)

        with torch.no_grad():
            y1.sub_(fy2)
            del fy2

            dy2.add_(y2.grad)
            y2.grad.zero_()


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs, stream):
        ctx.kwargs = kwargs
        ctx.dtype = torch.get_autocast_gpu_dtype()
        for block in blocks:
            block(x, **kwargs)  # inplace
        if stream is not None:
            # wait for the boilerplate to become available
            torch.cuda.current_stream().wait_stream(stream)
            # must copy synchronously as this tensor will be overwritten in the next block
            stream.boilerplate.copy_(x.detach())
            # wait until the copy finishes
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                x_detached = stream.boilerplate.to("cpu", non_blocking=True)
                # stack CPU tensors so that we can take the previous in backward()
                stream.checkpoints.append(x_detached)
        else:
            x_detached = x.detach().clone()
        ctx.stream = stream
        ctx.save_for_backward(x_detached)
        ctx.blocks = blocks
        return x

    @staticmethod
    @once_differentiable
    def backward(
        ctx, dy
    ):  # pragma: no cover # this is covered, but called directly from C++
        y, = ctx.saved_tensors
        if (stream := ctx.stream) is not None:
            if y is (top := stream.checkpoints.pop() if stream.checkpoints else None):
                # boilerplate is the same as in the last forward()
                top = stream.checkpoints.pop()
            else:
                # wait until the previously issued copy_() finishes
                torch.cuda.current_stream().wait_stream(stream)
            y = stream.boilerplate.clone()
            if top is not None:
                # must start overwriting the boilerplate after clone() finishes
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    stream.boilerplate.copy_(top, non_blocking=True)
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            block.backward_pass(y, dy, **kwargs)

        return dy, None, None, None


class ReversibleSequence(nn.Module):
    def __init__(self, blocks: nn.ModuleList, split_dim: int = 0, checkpoint: bool = False):
        super().__init__()

        # pyre-fixme[23]: Unable to unpack `torch.nn.Module` into 2 values.
        self.blocks = nn.ModuleList([ReversibleBlock(
            f, g, split_dim=split_dim) for f, g in blocks]
        )
        self.checkpoint = checkpoint

    def forward(
        self,
        x,
        arg_route=(True, False),
        input_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {"f_args": f_args, "g_args": g_args}
        if self.checkpoint:
            try:
                stream = x.stream
            except AttributeError:
                stream = torch.cuda.Stream()
                stream.boilerplate = torch.empty_like(x)
                stream.checkpoints = []
        else:
            stream = None
        y = _ReversibleFunction.apply(x, self.blocks, block_kwargs, stream)
        assert y.data_ptr() == x.data_ptr()
        y.stream = stream
        return y


class InputAdapter(nn.Module):
    def __init__(self, split_dim: int = 0):
        super().__init__()
        self.split_dim = split_dim

    def forward(self, x, input_mask: Optional[torch.Tensor] = None):
        y = torch.cat([x, x], dim=self.split_dim)
        # Apply the optional input masking
        if input_mask is not None:
            assert self.split_dim == -1, "TODO for other dimensions"
            if y.dim() - input_mask.dim() > 1:
                input_mask.unsqueeze(0)
            y += input_mask.unsqueeze(-1)
        return y


class OutputAdapter(nn.Module):
    def __init__(self, split_dim: int = 0):
        super().__init__()
        self.split_dim = split_dim

    def forward(self, x, **_):
        return torch.stack(x.chunk(2, dim=self.split_dim)).mean(dim=0)
