import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False

    @property
    def scaling_value(self) -> float:
        """Compute the scaling value for LoRA."""
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class LoRALinear(nn.Module):
    """Linear layer with LoRA support. Can be used as a drop-in replacement for nn.Linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_config: Optional[LoRAConfig] = None,
        base_layer: Optional[nn.Linear] = None,
    ):
        """
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, the layer will not learn an additive bias
            lora_config: LoRA configuration. If None, behaves like a regular Linear layer
            base_layer: Optional pre-existing Linear layer to wrap. If None, creates a new one.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_config = lora_config

        # Create or use existing base layer
        if base_layer is not None:
            self.base_layer = base_layer
        else:
            self.base_layer = nn.Linear(in_features, out_features, bias=bias)

        # Initialize LoRA parameters if config is provided
        if lora_config is not None:
            # LoRA A: [in_features, rank] - initialized with normal distribution (std=0.01)
            # LoRA B: [rank, out_features] - initialized to zero
            # This ensures initial LoRA output is zero, preserving base model behavior at start
            self.lora_A = nn.Parameter(torch.randn(in_features, lora_config.rank) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(lora_config.rank, out_features))
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA."""
        # Base output
        output = self.base_layer(x)

        # Add LoRA contribution if configured
        if self.lora_config is not None and self.lora_A is not None and self.lora_B is not None:
            # LoRA computation: x @ lora_A @ lora_B * scaling
            # lora_A shape: [in_features, rank]
            # lora_B shape: [rank, out_features]
            # x shape: [..., in_features]
            dtype = x.dtype
            lora_output = torch.matmul(x, self.lora_A.to(dtype))
            lora_output = torch.matmul(lora_output, self.lora_B.to(dtype))
            output = output + lora_output * self.lora_config.scaling_value

        return output

    def extra_repr(self) -> str:
        """Extra representation string."""
        s = f"in_features={self.in_features}, out_features={self.out_features}"
        if self.lora_config is not None:
            s += f", lora_rank={self.lora_config.rank}, lora_alpha={self.lora_config.alpha}"
        return s

