from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from xformers.triton.dropout import FusedDropoutBias


@dataclass
class LoRAConfig:
    rank: int
    dropout: float
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(x)
        y = self.matmul(y, self.high_to_low_a.weight)
        y = self.matmul(y, self.low_to_high_b.weight)
        return y
