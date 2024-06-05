import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_trunc_normal_

from xformers.triton.dropout import FusedDropoutBias


@dataclass
class LoRAConfig:
    rank: int
    dropout: float
    init: Literal["ortho", "zero_b", "almost_zero_b", "none"] = "ortho"
    alpha: int = 16


class LoRA(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        config: LoRAConfig,
        matmul: Callable = torch.nn.functional.linear,
    ):
        super().__init__()
        self.high_to_low_a = nn.Linear(in_size, config.rank, bias=False)
        self.low_to_high_b = nn.Linear(config.rank, out_size, bias=False)
        if config.dropout:
            self.dropout = FusedDropoutBias(config.dropout, None, scale=config.alpha / config.rank)
        else:
            self.dropout = lambda x: x * (config.alpha / config.rank)
        self.matmul = matmul
        self.init = config.init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(x)
        y = self.matmul(y, self.high_to_low_a.weight)
        y = self.matmul(y, self.low_to_high_b.weight)
        return y

    def init_weights(self) -> None:
        match self.init:
            case "none":
                pass
            case "ortho":
                dim_in, rank, dim_out = (
                    self.high_to_low_a.in_features,
                    self.high_to_low_a.out_features,
                    self.low_to_high_b.out_features,
                )
                basis = torch.empty(
                    rank,
                    rank,
                    device=self.high_to_low_a.weight.device,
                    dtype=torch.float32,
                )
                torch.nn.init.orthogonal_(basis)
                combinations = torch.rand(dim_in + dim_out, rank // 2, device=basis.device)
                with torch.no_grad():
                    torch.matmul(
                        combinations[:dim_in],
                        basis[: rank // 2],
                        out=self.high_to_low_a.weight.T,
                    )
                    torch.matmul(
                        combinations[dim_in:],
                        basis[rank // 2 :],
                        out=self.low_to_high_b.weight,
                    )
                    # normalize
                    for param in (self.high_to_low_a, self.low_to_high_b):
                        fan_in, fan_out = _calculate_fan_in_and_fan_out(param.weight)
                        std = math.sqrt(2.0 / float(fan_in + fan_out))
                        ratio = (std * math.sqrt(fan_in * fan_out)) / torch.norm(param.weight)
                        param.weight.mul_(ratio)
            case "zero_b" | "almost_zero_b":
                for param, scale in (
                    (self.high_to_low_a, 1),
                    (self.low_to_high_b, 0 if self.init == "zero_b" else 1e-7),
                ):
                    if scale == 0:
                        param.zero_()
                        continue
                    # xavier_normal_2sigma
                    fan_in, fan_out = _calculate_fan_in_and_fan_out(param.weight)
                    std = math.sqrt(2.0 / float(fan_in + fan_out)) * scale
                    _no_grad_trunc_normal_(
                        param.weight,
                        0.0,
                        std / 0.87962566103423978,
                        a=-2,
                        b=2,
                    )
            case _:
                raise AssertionError(f"Unsupported LoRA initialization: {self.init}")
