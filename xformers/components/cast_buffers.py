from dataclasses import dataclass

import torch


@dataclass(slots=True)
class LinearWeightBias:
    weight: torch.Tensor
    bias: torch.Tensor | None = None
