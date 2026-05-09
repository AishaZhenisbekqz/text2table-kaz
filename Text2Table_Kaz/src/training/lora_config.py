"""
LoRA configuration for Qwen3.5-4B adaptation.

Implements the configuration from Table 8 of the paper:
  r=32, alpha=64, dropout=0.05
  Target modules: q/k/v/o_proj + gate/up/down_proj (SwiGLU FFN)
"""

from __future__ import annotations

from peft import LoraConfig, TaskType


def get_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """
    Build LoRA configuration.

    Args:
        r: rank (paper: 32). Controls trainable parameter count.
           Trainable params ≈ 2*r*(d_k) * n_layers * n_modules.
        lora_alpha: scaling factor (paper: 64, so alpha/r = 2).
        lora_dropout: dropout on LoRA weights (paper: 0.05).

    Returns:
        peft.LoraConfig object.

    Note:
        Adapters are injected into all linear projections of:
          - Self-attention: q_proj, k_proj, v_proj, o_proj
          - SwiGLU FFN:     gate_proj, up_proj, down_proj
        This covers ~2% of Qwen3.5-4B parameters.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        # Initialize B=0 so Delta_W=0 at training start (paper, Section III-C)
        init_lora_weights=True,
    )
