#!/usr/bin/env python3
"""
Training entry point for dual-regime LoRA fine-tuning.

Usage:
    # Dynamic regime (recommended)
    python scripts/train.py --config configs/dynamic_regime.yaml --seed 42

    # Static regime
    python scripts/train.py --config configs/static_regime.yaml

    # Full replication (3 seeds)
    for seed in 42 123 7; do
        python scripts/train.py --config configs/dynamic_regime.yaml --seed $seed
    done
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.lora_trainer import LoRATrainer, TrainingConfig, LoRAConfig
from src.training.dataset import Text2TableDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Text2Table_Kaz LoRA adapter")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--regime", type=str, default=None,
                        help="Override regime: 'static' or 'dynamic'")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Build training config
    lora_cfg = LoRAConfig(**cfg.get("lora", {}))
    t_cfg = cfg.get("training", {})
    o_cfg = cfg.get("output", {})
    d_cfg = cfg.get("data", {})
    r_cfg = cfg.get("reproducibility", {})

    regime = args.regime or cfg["model"].get("regime", "dynamic")
    seed = args.seed if args.seed is not None else r_cfg.get("seed", 42)

    config = TrainingConfig(
        base_model=cfg["model"]["base_model"],
        regime=regime,
        lora=lora_cfg,
        learning_rate=t_cfg.get("learning_rate", 1e-4),
        num_epochs=t_cfg.get("num_epochs", 3),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 32),
        warmup_steps=t_cfg.get("warmup_steps", 200),
        weight_decay=t_cfg.get("weight_decay", 0.01),
        bf16=t_cfg.get("bf16", True),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        early_stopping_patience=t_cfg.get("early_stopping_patience", 2),
        max_context_length=d_cfg.get("max_context_length", 3000),
        max_generation_length=d_cfg.get("max_generation_length", 1500),
        output_dir=o_cfg.get("output_dir", f"./checkpoints/{regime}"),
        seed=seed,
    )

    logger.info(f"=== Text2Table_Kaz Training ===")
    logger.info(f"Regime:     {regime}")
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Seed:       {seed}")
    logger.info(f"LoRA r={config.lora.r}, α={config.lora.lora_alpha}")

    # Initialize trainer
    trainer = LoRATrainer(config)

    # Load datasets
    logger.info("Loading datasets...")
    train_ds = Text2TableDataset(
        data_path=d_cfg["train_path"],
        tokenizer=trainer.tokenizer,
        regime=regime,
        max_context_length=config.max_context_length,
        max_generation_length=config.max_generation_length,
    )
    val_ds = Text2TableDataset(
        data_path=d_cfg["val_path"],
        tokenizer=trainer.tokenizer,
        regime=regime,
        max_context_length=config.max_context_length,
        max_generation_length=config.max_generation_length,
    )
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Train
    trainer.train(train_ds, val_ds)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
