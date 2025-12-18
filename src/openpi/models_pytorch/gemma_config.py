"""PyTorch version of Gemma configuration."""

from typing import Literal
import ml_collections

from openpi.models_pytorch.lora import LoRAConfig


Config = ml_collections.ConfigDict
Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant (PyTorch version with PyTorch LoRAConfig)."""
    if variant == "dummy":
        return Config(
            {
                "width": 64,
                "depth": 4,
                "mlp_dim": 128,
                "num_heads": 8,
                "num_kv_heads": 1,
                "head_dim": 16,
            }
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            {
                "width": 1024,
                "depth": 18,
                "mlp_dim": 4096,
                "num_heads": 8,
                "num_kv_heads": 1,
                "head_dim": 256,
            }
        )
    if variant == "gemma_2b":
        return Config(
            {
                "width": 2048,
                "depth": 18,
                "mlp_dim": 16_384,
                "num_heads": 8,
                "num_kv_heads": 1,
                "head_dim": 256,
            }
        )
    if variant == "gemma_2b_lora":
        return Config(
            {
                "width": 2048,
                "depth": 18,
                "mlp_dim": 16_384,
                "num_heads": 8,
                "num_kv_heads": 1,
                "head_dim": 256,
                "lora_configs": {
                    "attn": LoRAConfig(rank=16, alpha=16.0),
                    "ffn": LoRAConfig(rank=16, alpha=16.0),
                },
            }
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            {
                "width": 1024,
                "depth": 18,
                "mlp_dim": 4096,
                "num_heads": 8,
                "num_kv_heads": 1,
                "head_dim": 256,
                "lora_configs": {
                    "attn": LoRAConfig(rank=32, alpha=32.0),
                    "ffn": LoRAConfig(rank=32, alpha=32.0),
                },
            }
        )
    raise ValueError(f"Unknown variant: {variant}")







