# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: the underlying kernel comes straight from the Triton tutorials
# see https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py

import logging
from typing import Optional

import torch
import torch.nn as nn
import triton
from torch.cuda.amp import custom_bwd, custom_fwd

from xformers.triton.k_layer_norm import (
    layer_norm_bwd_dwdb,
    layer_norm_bwd_dx_fused,
    layer_norm_fw,
)

logger = logging.getLogger("xformers")


_triton_layernorm_cast = None  # NOTE: PyTorch keeps layernorm as fp32
_triton_registered_warnings = False


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=_triton_layernorm_cast)
    def forward(ctx, x, weight, bias, eps, grad_dtype):
        # catch eps being too small if the tensors are fp16
        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-5)

        # allocate output
        y = torch.empty_like(x)

        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # allocate mean and std, they'll be used in the backward pass
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        if not x_arg.is_contiguous() or not y.is_contiguous():
            global _triton_registered_warnings
            if not _triton_registered_warnings:
                logger.warning(
                    "Non-contiguous input tensor found. Making it contiguous,"
                    + " but could have perf or trainer implications"
                )

                _triton_registered_warnings = True

            x_arg = x_arg.contiguous()
            y = y.contiguous()

        # heuristics for number of warps.
        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 16)

        # enqueue kernel
        # fmt: off
        layer_norm_fw[(M,)](
            x_arg, y, weight, bias, mean, rstd,
            x_arg.stride(0),
            N,
            eps,
            num_warps=num_warps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            affine=weight is not None
        )
        # fmt: on

        ctx.save_for_backward(x, mean, rstd, weight)
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps
        ctx.grad_dtype = grad_dtype if grad_dtype is not None else x.dtype

        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(
        ctx, dy
    ):  # pragma: no cover  # this is covered, but called directly from C++
        x, mean, rstd, weight = ctx.saved_tensors

        # flatten the batch dimension, if any.
        # We're interested in 'samples' x norm_dimension
        x = x.reshape(-1, x.size(-1))
        M, N = x.size()

        # heuristics for amount of parallel reduction stream for DG/DB
        GROUP_SIZE_M = 32
        if N <= 8192:
            GROUP_SIZE_M = 64
        if N <= 4096:
            GROUP_SIZE_M = 96
        if N <= 2048:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256

        if dy.dtype == torch.float32:
            GROUP_SIZE_M = GROUP_SIZE_M // 2

        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=x.device)
        _dw = torch.zeros((GROUP_SIZE_M, x.size(-1)), dtype=torch.float32, device=x.device)
        _db = torch.zeros_like(_dw)
        dw = torch.empty((x.size(-1),), dtype=ctx.grad_dtype, device=x.device)
        db = torch.empty_like(dw)
        dy = dy.contiguous()
        dx = torch.empty_like(dy)

        # Check the tensor shapes and layouts
        # we suppose in the kernel that they have the same size and are contiguous
        assert (
            dy.numel() == x.numel()
        ), "Something is wrong in the backward graph, possibly because of an inplace operation after the layernorm"

        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        num_warps = min(max(ctx.BLOCK_SIZE_N // 256, 1), 16)

        # fmt: off
        layer_norm_bwd_dx_fused[(M,)](
            dx, dy, _dw, _db, x,
            weight if weight is not None else x,
            mean, rstd,
            locks,
            x.stride(0),
            N,
            affine=weight is not None,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE_N,
            num_warps=num_warps
        )
        # fmt: on

        def grid(meta):
            return [triton.cdiv(N, meta["BLOCK_SIZE_N"])]

        # accumulate partial sums in separate kernel
        # fmt: off
        layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db,
            GROUP_SIZE_M,
            N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=64
        )
        # fmt: on

        dx = dx.reshape_as(dy)
        return dx, dw, db, None, None


class FusedLayerNorm(nn.Module):
    """
    Handle a layer normalization, like torch.nn.LayerNorm_.

    This implementation should be measurably faster than the default PyTorch layernorm (as of PyTorch 1.9),
    both for training and inference workloads.

    .. NOTE: Computations under Torch AMP are kept as float32 by default, one can change this to be (b)float16
        by setting the flag `xformers.triton.k_layer_norm._triton_layernorm_cast = torch.(b)float16`

    .. _torch.nn.LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    """

    def __init__(self, normalized_shape, affine=True, eps=1e-06):
        super().__init__()
        if affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.weight = self.bias = None
        self.epsilon = eps

    def forward(self, x):
        if (grad := self.weight.grad) is not None:
            grad_dtype = grad.dtype
        else:
            grad_dtype = None
        return layer_norm(x, self.weight, self.bias, self.epsilon, grad_dtype)

    def init_weights(self, *args, **kwargs):
        with torch.no_grad():
            if self.weight is not None:
                self.weight.fill_(1.0)

            if self.bias is not None:
                self.bias.fill_(0.0)


def layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-06,
    grad_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    input_dtype = x.dtype

    global _triton_registered_warnings

    r"""Applies normalization over a mini batch of inputs"""

    y = None
    try:
        if (
            not _triton_registered_warnings
            and torch.cuda.is_available()
            and x.is_cuda
            and weight is not None
            and bias is not None
        ):
            y = _LayerNorm.apply(x, weight, bias, eps, grad_dtype)
    except RuntimeError as e:
        # Catch cases where the current GPU does not have enough registers to hold a full tensor line
        # fallback to PyTorch's implementation, which streams the tensor in and out
        if not _triton_registered_warnings:
            _triton_registered_warnings = True
            logger.warning(
                "Triton layernorm kernel register spillover or invalid image caught. "
                "Deactivating this kernel, please file an issue in the xFormers repository"
            )
            logger.warning(e)

    if y is None:
        y = torch.nn.functional.layer_norm(
            x, [x.shape[-1]], weight=weight, bias=bias, eps=eps
        ).to(torch.get_autocast_gpu_dtype())

    assert y.dtype == input_dtype
    return y
