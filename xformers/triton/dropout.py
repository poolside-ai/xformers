# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This is heavily inspired by the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

from typing import Optional

import torch
import triton
from torch.amp import custom_bwd, custom_fwd

from xformers.components.activations import Activation, build_activation
from xformers.triton.k_activations import get_triton_activation_index
from xformers.triton.k_dropout import k_dropout_bw, k_dropout_fw

BLOCK_M = 64
BLOCK_N = 128  # NOTE: This should ideally be GPU dependent, big impact on perf


# Helper to handle the SPMD launch grid and error cases
class _dropout(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x, p, bias, activation, scale, trainable_bias, inplace_fwd, inplace_bwd):
        # Soft-flatten an hypothetical 3rd dimension
        x_ = x.reshape(-1, x.shape[-1]).contiguous()
        if not inplace_fwd:
            x_ = x_.clone()
        M, N = x_.shape

        assert bias is None or (bias.dtype == x.dtype and bias.shape[0] == N)
        assert p > 0.0

        def grid(meta):
            return (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"]),
            )

        N_BLOCK_N = triton.cdiv(N, BLOCK_N)

        # Generate one seed per sample
        seeds = torch.randint(1 << 16, (N_BLOCK_N,), device=x.device, dtype=torch.int32)

        # fmt: off
        bias_ptr = bias if bias is not None else x_  # Possibly not being used

        if not k_dropout_fw.cache:
            # we execute the operation inplace, and it will otherwise damage x_
            k_dropout_fw[grid](
                x_.clone().detach(),
                bias_ptr,
                seeds,
                x_.stride(0),
                M, N,
                p,
                USE_BIAS=bias is not None,
                ACTIVATION=activation,
                SCALE=scale,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        k_dropout_fw[grid](
            x_,
            bias_ptr,
            seeds,
            x_.stride(0),
            M, N,
            p,
            USE_BIAS=bias is not None,
            ACTIVATION=activation,
            SCALE=scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        # fmt: on

        ctx.save_for_backward(seeds, bias, x if activation is not None else None)
        ctx.trainable_bias = bias is not None and trainable_bias
        ctx.activation = activation
        ctx.scale = scale
        ctx.p = p
        ctx.inplace = inplace_bwd

        return x_.reshape_as(x)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(
        ctx, grad_out
    ):  # pragma: no cover  # This is covered, but called from C++ and not tracked
        seeds, bias, inputs = ctx.saved_tensors

        # Soft-flatten an hypothetical 3rd dimension
        grad_out_ = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_in = grad_out_ if ctx.inplace else torch.empty_like(grad_out_)

        M, N = grad_out_.shape

        # Optional inputs to compute the activation contribution to the gradient
        assert inputs is not None or ctx.activation is None

        if inputs is None:
            inputs = grad_out_
        elif inputs.ndim > 2:
            inputs = inputs.reshape(-1, N)

        # We split the problem in tiles:
        # - over M there will be a follow-up reduction
        # - over N we compromise in between trying to use as much memory parallelism as possible,
        # (fill in the warps, there are 32 threads per warps, and 4 warps default), and not being too
        # big because of register spilling
        N_BLOCKS_M = triton.cdiv(M, BLOCK_M)

        if ctx.trainable_bias:
            grad_bias = torch.empty(
                (
                    N_BLOCKS_M,
                    N,
                ),
                device=grad_in.device,
                dtype=grad_in.dtype,
            )

        else:
            grad_bias = grad_in  # will not be used

        def grid(meta):
            # NOTE: We use Triton Philox random number generator, which optimally generates 4 blocks for
            # a given seed and offsets. "BLOCK_M" here describes the size of one of these blocks
            # but we need to take this factor of 4 into account when scheduling all the kernels
            return (
                N_BLOCKS_M,
                triton.cdiv(N, meta["BLOCK_N"]),
            )

        # fmt: off
        k_dropout_bw[grid](
            grad_in, grad_bias, grad_out_,
            inputs, bias if bias is not None else inputs,
            seeds,
            grad_out_.stride(0), inputs.stride(0),
            M, N,
            ctx.p,
            INPLACE=ctx.inplace,
            USE_BIAS=bias is not None,
            ACTIVATION=ctx.activation,
            SCALE=ctx.scale,
            TRAINABLE_BIAS=ctx.trainable_bias,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        # fmt: on

        return (
            grad_in.reshape_as(grad_out),
            None,
            torch.sum(grad_bias, dim=0) if ctx.trainable_bias else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def dropout(
    x: torch.Tensor,
    p: float,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[Activation] = None,
    scale: float = 1.0,
    inplace_fwd: bool = False,
    inplace_bwd: bool = False,
):
    """
    Apply dropout on the input tensor.
    Optionally add a bias, the computation will be fused.
    """

    assert p <= 1.0 and p >= 0.0

    if p == 1.0:
        return torch.zeros_like(x)

    # Micro optim, skip dropout
    if p == 0.0:
        x = x + bias if bias is not None else x
        if activation is not None:
            activation_fn = build_activation(activation)
            return activation_fn(x)
        return x

    # The normal triton enabled codepath
    activation_index = get_triton_activation_index(activation)
    return _dropout.apply(
        x,
        float(p),
        bias,
        activation_index,
        scale,
        bias is not None and bias.requires_grad,
        inplace_fwd,
        inplace_bwd,
    )


class FusedDropoutBias(torch.nn.Module):
    """
    A layer which fuses the computation of Dropout(Activation(x))
    in a single GPU kernel
    """

    def __init__(
        self,
        p: float,
        bias_shape: Optional[int],
        activation: Optional[Activation] = None,
        scale: float = 1.0,
        inplace_fwd: bool = False,
        inplace_bwd: bool = False,
    ) -> None:
        super().__init__()

        self.p = float(p)

        assert (
            self.p < 1.0
        ), f"We don't want to drop all the values, most probably p={p} is not properly set"

        self.activation_type = activation
        self.bias = (
            torch.zeros(bias_shape, requires_grad=True)
            if bias_shape is not None
            else None
        )
        self.activation = get_triton_activation_index(self.activation_type)
        self.activation_pytorch = build_activation(self.activation_type)
        self.scale = scale
        self.inplace_fwd = inplace_fwd
        self.inplace_bwd = inplace_bwd

    @torch.no_grad()
    def init_weights(self, *args, **kwargs) -> int:
        if self.bias is not None:
            self.bias.fill_(0.0)
            return 1
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convenience, catch a possible type or device mismatch
        if self.bias is not None:
            self.bias = self.bias.to(dtype=x.dtype, device=x.device)  # type: ignore

        # Train/inference
        p = self.p if self.training else 0.0

        # This kernel is slower than pytorch for small buffers, bypassing it in that case
        perf_check = x.shape[-1] > 512

        # Catch a non-cuda setup, fallback to pytorch
        if not x.is_cuda or not perf_check or p == 0.0:
            x = x + self.bias if self.bias is not None else x
            x = self.activation_pytorch(x) * self.scale
            return torch.nn.functional.dropout(x, p) if p > 0.0 else x

        # The normal, Triton-backed path
        return _dropout.apply(
            x,
            p,
            self.bias,
            self.activation,
            self.scale,
            True,
            self.inplace_fwd,
            self.inplace_bwd,
        )
