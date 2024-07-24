# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import asdict, dataclass, InitVar
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.init import constant_

from xformers import _is_triton_available
from xformers.components.attention import Attention
from xformers.components.cast_buffers import LinearWeightBias
from xformers.components.input_projection import (
    InputProjection,
    InputProjectionBuffers,
    InputProjectionConfig,
)
from xformers.components.lora import LoRAConfig
from xformers.components.positional_embedding import RotaryEmbedding


if _is_triton_available():
    from xformers.triton.dropout import FusedDropoutBias


logger = logging.getLogger("xformers")


# this cannot be a dataclass, otherwise the config wrapper converts it to a dict
class AttentionBuffers:
    __slots__ = ("out_proj", "in_proj")

    out_proj: LinearWeightBias
    in_proj: InputProjectionBuffers

    def __init__(self, out_proj: LinearWeightBias, in_proj: InputProjectionBuffers) -> None:
        self.out_proj = out_proj
        self.in_proj = in_proj

    def __deepcopy__(self, memo: dict) -> "AttentionBuffers":
        # never clone self
        return self


@dataclass
class MultiHeadDispatchConfig:
    dim_model: int
    num_heads: int
    attention: Attention
    bias: bool
    residual_dropout: float
    dim_key: Optional[int]
    dim_value: Optional[int]
    in_proj_container: Optional[InputProjection]
    use_separate_proj_weight: Optional[bool]
    use_rotary_embeddings: Optional[bool]
    out_proj: Optional[nn.Module]
    cast_buffers: AttentionBuffers = None
    matmul: Callable = torch.nn.functional.linear
    qk_layernorm: bool = False
    inplace_dropout: bool = True
    lora: LoRAConfig | None = None

    def __getitem__(self, item):
        return getattr(self, item)


# Simple reshape to indicate the number of heads.
def _expose_heads(t: torch.Tensor, B: int, S: int, H: int, Hs: int):
    return t.view(B, S, H, Hs)


# Move head forward and fold into batch dim. dimensions become (B * nh, S, hs)
def _fold_heads(t: torch.Tensor, B: int, S: int, H: int, Hs: int):
    return _expose_heads(t, B, S, H, Hs).transpose(1, 2).flatten(start_dim=0, end_dim=1)


# Move head forward and fold into batch dim. dimensions become (B, nh, S, hs)
def _split_heads(t: torch.Tensor, B: int, S: int, H: int, Hs: int):
    return _expose_heads(t, B, S, H, Hs).transpose(1, 2)


class MultiHeadDispatch(nn.Module):
    """
    A multi-head masked self-attention dispatch mechanism, with a projection at the end,
    following the architecture proposed in `Attention is all you need`_, Vaswani et al.

    The actual attention mechanism can vary, as well as the projections.
    This can be used to wrap the proposed attention mechanisms and make them multi-head aware,
    but it is optional.

    Args:
        dim_model: The model/embedding dimension
        num_heads: The number of heads being used
        attention: The attention mechanism (needs to be registered to the xformers library)
        bias: Whether to use bias for the projections : (Q, K, V, Output)
        residual_dropout: Amount of dropout on the residual path
        inplace_dropout: Perform dropout forward in-place
        use_separate_proj_weight: Use different weights for the Q, K, V projections
        dim_key: Optionally use a different dimension for the key
        dim_value:  Optionally use a different dimension for the value
        in_proj_container: Optionally provide the input projection module
        use_rotary_embeddings: Use rotary embeddings
        out_proj: Optionally provide the output projection module


    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762v5
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        attention: Attention,
        bias: Tuple[bool, bool, bool, bool] = (True, True, True, True),
        residual_dropout: float = 0.0,
        inplace_dropout: bool = True,
        use_separate_proj_weight: bool = True,
        dim_key: Optional[int] = None,
        dim_value: Optional[int] = None,
        in_proj_container: Optional[InputProjection] = None,
        use_rotary_embeddings: Optional[bool] = False,
        out_proj: Optional[nn.Module] = None,
        cast_buffers: AttentionBuffers | None = None,
        matmul: Callable = torch.nn.functional.linear,
        qk_layernorm: bool = False,
        lora: LoRAConfig | dict | None = None,
        **kwargs,
    ):
        super().__init__()

        if isinstance(bias, bool):
            logger.warning(
                "Single bias value provided for the MHA projections."
                + f" Assuming the same parameter ({bias}) is to be used everywhere"
            )
            bias = (bias, bias, bias, bias)

        assert (
            dim_model % num_heads == 0
        ), f"dim_model {dim_model} must divide by num_heads {num_heads}"
        # static preset for now, each head works on 1/d the embeddings, could be relaxed
        assert num_heads > 0

        # Popular default is that all latent dimensions are the same
        dim_key, dim_value = map(lambda x: x if x else dim_model, (dim_key, dim_value))

        self.num_heads = num_heads
        self.dim_key_head = dim_key // num_heads
        self.dim_value_head = dim_value // num_heads
        self.dim_model = dim_model
        self.attention = attention
        if lora is None:
            lora = LoRAConfig(0, 0)
        elif isinstance(lora, dict):
            lora = LoRAConfig(**lora)

        # key, query, value projections for all heads
        # critical options are
        # - are we sharing weights ?
        # - are we adding biases ?
        if attention.requires_input_projection:
            self.in_proj_container = (
                in_proj_container
                if in_proj_container is not None
                else InputProjection(
                    query_proj_params=InputProjectionConfig(
                        dim_model, dim_key, bias=bias[0], lora=lora,
                    ),
                    key_proj_params=InputProjectionConfig(
                        dim_model, dim_key, bias=bias[1], lora=LoRAConfig(0, 0),
                    ),
                    value_proj_params=InputProjectionConfig(
                        dim_model, dim_value, bias=bias[2], lora=lora,
                    ),
                    use_separate_proj_weight=use_separate_proj_weight,
                    cast_buffers=cast_buffers.in_proj if cast_buffers is not None else None,
                    matmul=matmul,
                    qk_layernorm=qk_layernorm,
                )
            )

        # Optional rotary embeddings
        self.rotary_embeddings = (
            RotaryEmbedding(self.dim_key_head) if use_rotary_embeddings else None
        )

        # Regularization
        self.resid_drop = FusedDropoutBias(
            p=residual_dropout,
            bias_shape=None,
            inplace_fwd=inplace_dropout,
        )

        # Output projection
        self.proj = (
            out_proj if out_proj else nn.Linear(dim_model, dim_model, bias=bias[3])
        )
        if isinstance(self.proj, nn.Linear) and self.proj.bias is not None:
            constant_(self.proj.bias, 0.0)
        self.cast_buffers = cast_buffers
        self.matmul = matmul

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        att_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Expected input dimensions are [batch size, sequence length, embed dim]
        Output dimensions are [batch size, sequence length, embed dim]
        """

        if key is None:
            key = query
        if value is None:
            value = query

        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            max_batch = max((query.shape[0], key.shape[0], value.shape[0]))
            query, key, value = map(
                lambda x: x.expand(max_batch, -1, -1), [query, key, value]
            )

        B, S_Q, _ = query.size()  # Batch x Sequence x Embedding (latent)
        _, S_K, _ = key.size()  # K, Q's sequence length could differ

        # Catch different query and key length but a causal attention
        if S_Q != S_K:
            assert (
                not self.attention.requires_same_k_q_dimensions
            ), "This attention mechanism requires query and key to have the same sequence (context) lengths"

            if hasattr(self.attention, "causal"):
                assert not self.attention.causal, (
                    "Causal attention is not supported when key and query have different sequence lengths.\n"
                    + "In that case causality is ill-determined. Please pad your sequences accordingly"
                )

        kw_mask_args = {}
        if att_mask is not None:
            assert (
                self.attention.supports_attention_mask
            ), "This attention does not support attention masks"
            kw_mask_args["att_mask"] = att_mask

        if key_padding_mask is not None:
            assert (
                self.attention.supports_key_padding_mask
            ), "This attention does not support key padding masks"
            kw_mask_args["key_padding_mask"] = key_padding_mask

        if self.attention.requires_skip_multi_head:
            return self.attention(query, key, value, **kw_mask_args)

        # Calculate query, key, values for all heads in batch
        if self.attention.requires_input_projection:
            q, k, v = self.in_proj_container(query=query, key=key, value=value)
        else:
            k, q, v = key, query, value

        # Check the dimensions properly
        def check(t, name):
            assert (
                t.shape[2] % self.num_heads == 0
            ), f"the {name} embeddings need to be divisible by the number of heads"

        check(q, "projected query")
        check(v, "projected value")
        check(k, "projected key")

        # Optional: rotary embedding, add relative positioning information
        if self.rotary_embeddings:
            # rotary requires the head dimension
            q = _split_heads(q, B, S_Q, self.num_heads, self.dim_key_head)
            k = _split_heads(k, B, S_K, self.num_heads, self.dim_key_head)
            v = _split_heads(v, B, S_K, self.num_heads, self.dim_value_head)

            q, k = self.rotary_embeddings(q=q, k=k)

            if not self.attention.requires_head_dimension:
                q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)

        else:
            # Reshape k/q/v to either expose the heads, or fold the head dimension into the batch
            if self.attention.supports_b_s_h_d:
                reshape_fn = _expose_heads
            elif self.attention.requires_head_dimension:
                reshape_fn = _split_heads
            else:
                reshape_fn = _fold_heads

            q = reshape_fn(q, B, S_Q, self.num_heads, self.dim_key_head)
            k = reshape_fn(k, B, S_K, self.num_heads, self.dim_key_head)
            v = reshape_fn(v, B, S_K, self.num_heads, self.dim_value_head)

        # Self-attend
        y = self.attention(q, k, v, **kw_mask_args)

        # Re-assemble all head outputs side by side
        if not self.attention.supports_b_s_h_d:
            y = (
                y.view(B, self.num_heads, S_Q, self.dim_value_head)
                .transpose(1, 2)
            )
        y = y.flatten(start_dim=2, end_dim=3)

        # Output projection, dropout and good to go
        if torch.is_autocast_enabled():
            if self.cast_buffers is None:
                dtype = torch.get_autocast_gpu_dtype()
                proj_weight = self.proj.weight.to(dtype)
                if self.proj.bias is not None:
                    proj_bias = self.proj.bias.to(dtype)
                else:
                    proj_bias = None
            else:
                lwb = self.cast_buffers.out_proj
                proj_weight = self.proj.weight.super_copy(lwb.weight)
                if self.proj.bias is not None:
                    proj_bias = self.proj.bias.super_copy(lwb.bias)
                else:
                    proj_bias = None
        else:
            proj_weight, proj_bias = self.proj.weight, self.proj.bias
        y = self.resid_drop(self.matmul(y, proj_weight, proj_bias))

        # Return the same sequence size as the input
        return y

    @classmethod
    def from_config(cls, config: MultiHeadDispatchConfig):
        # Generate the class inputs from the config
        fields = asdict(config)

        # Skip all Nones so that default values are used
        fields = {k: v for k, v in fields.items() if v is not None}

        return cls(**fields)
