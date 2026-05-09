"""
LoRA fine-tuning of Qwen3.5-4B for Kazakh text-to-table generation.
Ospan et al. (2024), Section IV.

Configuration:
    r=32, alpha=64, dropout=0.05
    Adapters on: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    ~2% trainable parameters

Training:
    AdamW (fused), lr=1e-4, cosine annealing, 200-step warmup
    bfloat16 + gradient checkpointing (single A100 80GB)
    Early stopping: patience=2 on validation loss
"""

import os
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    # Model
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "./checkpoints"
    regime: str = "dynamic"                  # "static" or "dynamic"

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Optimization
    learning_rate: float = 1e-4
    lr_min: float = 1e-6
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999

    # Schedule
    num_epochs: int = 3
    warmup_steps: int = 200
    gradient_accumulation_steps: int = 32    # effective batch = 32
    per_device_train_batch_size: int = 1

    # Precision
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Token limits
    max_context_length: int = 3000
    max_generation_length: int = 1500        # +300 for dynamic (variable schema)

    # Early stopping
    early_stopping_patience: int = 2

    # Reproducibility
    seed: int = 42

    # Logging
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500


class LoRATrainer:
    """
    Manages LoRA fine-tuning of Qwen3.5-4B.

    Usage:
        config = TrainingConfig(regime="dynamic", seed=42)
        trainer = LoRATrainer(config)
        trainer.train(train_dataset, val_dataset)
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._verify_dependencies()
        self.model, self.tokenizer = self._load_model()

    def _verify_dependencies(self):
        try:
            import torch, transformers, peft
        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}. Run: pip install -r requirements.txt")

    def _load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import get_peft_model, LoraConfig, TaskType

        logger.info(f"Loading base model: {self.config.base_model}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            device_map="auto",
        )

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    def train(self, train_dataset, val_dataset):
        from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            adam_beta1=self.config.beta1,
            adam_beta2=self.config.beta2,
            warmup_steps=self.config.warmup_steps,
            lr_scheduler_type="cosine",
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            seed=self.config.seed,
            optim="adamw_torch_fused",
            dataloader_num_workers=4,
        )

        from .callbacks import EarlyStoppingCallback

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                padding=True,
                pad_to_multiple_of=8,
            ),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            ],
        )

        logger.info(f"Starting training — regime: {self.config.regime}")
        trainer.train()

        # Save final adapter
        best_path = os.path.join(self.config.output_dir, f"{self.config.regime}_best")
        self.model.save_pretrained(best_path)
        self.tokenizer.save_pretrained(best_path)
        logger.info(f"Adapter saved to {best_path}")

        return trainer
