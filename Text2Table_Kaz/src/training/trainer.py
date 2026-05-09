"""
LoRA fine-tuning trainer for Kazakh text-to-table generation.

Implements Section III-C-3 (Optimization Protocol):
  - Fused AdamW (beta1=0.9, beta2=0.999, weight_decay=0.01)
  - Cosine annealing with linear warmup (Equation 8)
  - Gradient accumulation for effective batch size 32
  - Early stopping (patience=2) on validation loss
  - bfloat16 precision + gradient checkpointing (A100 single GPU)
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup,
)
from peft import get_peft_model

from .dataset import Text2TableDataset
from .lora_config import get_lora_config

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    Orchestrates LoRA fine-tuning for a given regime.

    Args:
        config: dict loaded from configs/static_regime.yaml or dynamic_regime.yaml
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.regime = config["regime"]
        self.model_cfg = config["model"]
        self.lora_cfg = config["lora"]
        self.data_cfg = config["data"]
        self.train_cfg = config["training"]
        self.out_cfg = config["output"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        logger.info(f"Starting {self.regime.upper()} regime training")

        self._load_model_and_tokenizer()
        self._apply_lora()

        train_dataset = self._make_dataset(self.data_cfg["train_file"])
        val_dataset = self._make_dataset(self.data_cfg["val_file"])

        args = self._build_training_args()
        collator = DataCollatorForSeq2Seq(
            self.tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            tokenizer=self.tokenizer,
        )

        logger.info(f"Trainable parameters: {self._count_trainable():,}")
        trainer.train()

        # Save best adapter
        output_dir = Path(self.out_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Adapter saved to {output_dir}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self) -> None:
        base_id = self.model_cfg["base_model"]
        dtype = torch.bfloat16 if self.model_cfg["torch_dtype"] == "bfloat16" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=dtype,
            device_map=self.model_cfg.get("device_map", "auto"),
        )
        if self.model_cfg.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()

        logger.info(f"Loaded base model: {base_id}")

    def _apply_lora(self) -> None:
        lora_config = get_lora_config(
            r=self.lora_cfg["r"],
            lora_alpha=self.lora_cfg["lora_alpha"],
            lora_dropout=self.lora_cfg["lora_dropout"],
        )
        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA adapters injected")

    def _make_dataset(self, path: str) -> Text2TableDataset:
        return Text2TableDataset(
            data_file=path,
            tokenizer=self.tokenizer,
            regime=self.regime,
            max_context_tokens=self.data_cfg["max_context_tokens"],
            max_generation_tokens=self.data_cfg["max_generation_tokens"],
        )

    def _build_training_args(self) -> TrainingArguments:
        t = self.train_cfg
        return TrainingArguments(
            output_dir=self.out_cfg["output_dir"],
            logging_dir=self.out_cfg["logging_dir"],
            num_train_epochs=t["num_epochs"],
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            learning_rate=t["learning_rate"],
            lr_scheduler_type="cosine",
            warmup_steps=t["warmup_steps"],
            weight_decay=t["weight_decay"],
            optim=t["optim"],
            adam_beta1=t["adam_beta1"],
            adam_beta2=t["adam_beta2"],
            bf16=True,
            evaluation_strategy=self.out_cfg.get("eval_strategy", "epoch"),
            save_strategy=self.out_cfg.get("save_strategy", "epoch"),
            load_best_model_at_end=self.out_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=self.out_cfg.get("metric_for_best_model", "eval_loss"),
            greater_is_better=False,
            seed=t["seed"],
            report_to=self.out_cfg.get("report_to", "none"),
            dataloader_num_workers=2,
            remove_unused_columns=False,
        )

    def _count_trainable(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Class method: load config and run
    # ------------------------------------------------------------------

    @classmethod
    def from_config_file(cls, path: str) -> "LoRATrainer":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)
