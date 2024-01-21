# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: Inspired by https://github.com/pytorch/text/blob/master/torchtext/nn/modules/multiheadattention.py
# and the MultiHeadAttention implementation from PyTorch


import logging
from dataclasses import InitVar, dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import nn

from xformers import _is_triton_available
from xformers.components.cast_buffers import LinearWeightBias

if _is_triton_available():
    from xformers.triton.layer_norm import FusedLayerNorm
else:
    FusedLayerNorm = torch.nn.LayerNorm

logger = logging.getLogger("xformers")


# this cannot be a dataclass, otherwise the config wrapper converts it to a dict
class InputProjectionBuffers:
    __slots__ = ("q", "k", "v")

    q: LinearWeightBias
    k: LinearWeightBias
    v: LinearWeightBias

    def __init__(self, q: LinearWeightBias, k: LinearWeightBias, v: LinearWeightBias) -> None:
        self.q = q
        self.k = k
        self.v = v

    def __deepcopy__(self, memo: dict) -> "InputProjectionBuffers":
        # never clone self
        return self


@dataclass
class InputProjectionConfig:
    in_features: int
    out_features: int
    bias: bool


class InputProjection(nn.Module):
    """
    Handle all the input projections in one go, opportunistically fuse some operations.
    """

    def __init__(
        self,
        query_proj_params: InputProjectionConfig,
        key_proj_params: Optional[InputProjectionConfig],
        value_proj_params: Optional[InputProjectionConfig],
        use_separate_proj_weight: bool = True,
        cast_buffers: InitVar[InputProjectionBuffers | None] = None,
        matmul: Callable = torch.nn.functional.linear,
        qk_layernorm: bool = False,
    ):
        super().__init__()
        self.streams = [torch.cuda.Stream() for _ in range(3)]

        self.out_features = query_proj_params.out_features

        # Each input gets a seperate projection
        self.q_proj = nn.Linear(
            query_proj_params.in_features,
            query_proj_params.out_features,
            query_proj_params.bias,
        )

        if key_proj_params is not None:
            self.k_proj = nn.Linear(
                key_proj_params.in_features,
                key_proj_params.out_features,
                key_proj_params.bias,
            )
        else:
            logger.info(
                "No Key projection parameters were passed, assuming that the weights"
                + " are shared with the query projection",
            )
            self.k_proj = self.q_proj

        if value_proj_params is not None:
            self.v_proj = nn.Linear(
                value_proj_params.in_features,
                value_proj_params.out_features,
                value_proj_params.bias,
            )
        else:
            logger.info(
                "No Value projection parameters were passed, assuming that the weights"
                + " are shared with the query projection",
            )
            self.v_proj = self.q_proj

        if not use_separate_proj_weight:
            # Compute optimization used at times, share the parameters in between Q/K/V
            with torch.no_grad():
                self.k_proj.weight = self.q_proj.weight
                self.v_proj.weight = self.q_proj.weight

        self.cast_buffers = cast_buffers
        self.matmul = matmul
        if qk_layernorm:
            self.q_ln = FusedLayerNorm(query_proj_params.out_features, bias=False)
            self.k_ln = FusedLayerNorm(
                (
                    key_proj_params.out_features
                    if key_proj_params is not None
                    else query_proj_params.out_features
                ),
                bias=False,
            )
        else:
            self.q_ln = self.k_ln = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # One projection per input tensor

        # NOTE: Would it make sense to catch self attention + shared weights, to skip a projection step ?
        results: list[torch.Tensor] = []
        mainstream = torch.cuda.current_stream()
        for stream, proj, ln, x, cast_buffer_field in zip(
            self.streams,
            [self.q_proj, self.k_proj, self.v_proj],
            [self.q_ln, self.k_ln, None],
            [query, key, value],
            InputProjectionBuffers.__slots__,
        ):
            with torch.cuda.stream(stream):
                stream.wait_stream(mainstream)
                if torch.is_autocast_enabled():
                    if self.cast_buffers is None:
                        dtype = torch.get_autocast_gpu_dtype()
                        weight = proj.weight.to(dtype)
                        if proj.bias is not None:
                            bias = proj.bias.to(dtype)
                        else:
                            bias = None
                    else:
                        lwb = getattr(self.cast_buffers, cast_buffer_field)
                        weight = proj.weight.super_copy(lwb.weight)
                        if proj.bias is not None:
                            bias = proj.bias.super_copy(lwb.bias)
                        else:
                            bias = None
                else:
                    weight, bias = proj.weight, proj.bias
                y = self.matmul(x, weight, bias)
                if ln is not None:
                    y = ln(y)
                results.append(y)
            mainstream.wait_stream(stream)

        return tuple(results)
