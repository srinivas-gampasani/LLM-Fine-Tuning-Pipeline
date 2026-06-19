"""
pipeline/training/trainer.py

Core LoRA / QLoRA fine-tuning engine.
Handles: model loading, quantization, LoRA injection, training loop, checkpointing.
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import wandb

logger = logging.getLogger(__name__)


@dataclass
class LoRATrainerConfig:
    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    # Quantization
    use_quantization: bool = True
    quant_bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "float16"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    # Training
    output_dir: str = "outputs/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    optim: str = "paged_adamw_32bit"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    report_to: str = "wandb"
    run_name: str = "lora-finetune"
    seed: int = 42
    group_by_length: bool = True
    # Hub
    push_to_hub: bool = False
    hub_repo_id: str = ""
    # W&B
    wandb_project: str = "llm-finetuning-pipeline"


class ModelLoader:
    """Loads and configures base model with optional QLoRA quantization."""

    @staticmethod
    def get_bnb_config(config: LoRATrainerConfig) -> Optional[BitsAndBytesConfig]:
        if not config.use_quantization:
            return None
        compute_dtype = getattr(torch, config.compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=(config.quant_bits == 4),
            load_in_8bit=(config.quant_bits == 8),
            bnb_4bit_quant_type=config.quant_type,
            bnb_4bit_use_double_quant=config.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    @staticmethod
    def load_model(config: LoRATrainerConfig) -> PreTrainedModel:
        torch_dtype = getattr(torch, config.torch_dtype)
        bnb_config = ModelLoader.get_bnb_config(config)

        logger.info(f"Loading model: {config.model_name}")
        logger.info(f"Quantization: {'4-bit NF4 QLoRA' if config.use_quantization else 'FP16 LoRA'}")

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype if not config.use_quantization else None,
            device_map=config.device_map,
            trust_remote_code=False,
        )
        model.config.use_cache = False  # Required for gradient checkpointing
        model.config.pretraining_tp = 1

        if config.use_quantization:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.gradient_checkpointing,
            )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Base model loaded — {total_params / 1e9:.2f}B parameters")
        return model

    @staticmethod
    def load_tokenizer(config: LoRATrainerConfig) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=False,
            padding_side="right",  # Important for causal LM training
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Tokenizer loaded — vocab size: {tokenizer.vocab_size:,}")
        return tokenizer


class LoRAInjector:
    """Injects LoRA adapters into the base model."""

    @staticmethod
    def inject(model: PreTrainedModel, config: LoRATrainerConfig) -> PeftModel:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=config.target_modules,
            inference_mode=False,
        )

        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = 100 * trainable / total

        logger.info(f"LoRA injected:")
        logger.info(f"  Trainable parameters : {trainable:,} ({pct:.3f}%)")
        logger.info(f"  Total parameters     : {total:,}")
        logger.info(f"  Frozen parameters    : {total - trainable:,}")
        logger.info(f"  LoRA rank (r)        : {config.lora_r}")
        logger.info(f"  LoRA alpha           : {config.lora_alpha}")
        logger.info(f"  Target modules       : {config.target_modules}")

        model.print_trainable_parameters()
        return model


class LoRAFinetuner:
    """
    End-to-end fine-tuning orchestrator.
    Usage:
        tuner = LoRAFinetuner(config)
        tuner.train(dataset_dict)
    """

    def __init__(self, config: LoRATrainerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def setup(self):
        """Load model, tokenizer, inject LoRA."""
        self.tokenizer = ModelLoader.load_tokenizer(self.config)
        self.model = ModelLoader.load_model(self.config)
        self.model = LoRAInjector.inject(self.model, self.config)
        return self

    def _get_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=False,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
            seed=self.config.seed,
            group_by_length=self.config.group_by_length,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            push_to_hub=self.config.push_to_hub,
            hub_model_id=self.config.hub_repo_id if self.config.push_to_hub else None,
        )

    def train(self, datasets: DatasetDict) -> Dict[str, Any]:
        """Run full training loop and return metrics."""
        if self.model is None:
            raise RuntimeError("Call .setup() before .train()")

        os.makedirs(self.config.output_dir, exist_ok=True)
        training_args = self._get_training_args()

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info(f"  Model       : {self.config.model_name}")
        logger.info(f"  Mode        : {'QLoRA 4-bit' if self.config.use_quantization else 'LoRA FP16'}")
        logger.info(f"  Train steps : {len(datasets['train']) // (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps) * self.config.num_train_epochs}")
        logger.info("=" * 60)

        train_result = trainer.train()
        metrics = train_result.metrics

        trainer.save_model(self.config.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info(f"Training complete. Model saved to: {self.config.output_dir}")
        return metrics

    def save_adapter(self, output_path: str):
        """Save only the LoRA adapter weights (much smaller than full model)."""
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"LoRA adapter saved to: {output_path}")

    def merge_and_save(self, output_path: str):
        """Merge LoRA weights into base model and save full merged model."""
        logger.info("Merging LoRA adapter into base model weights...")
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Merged model saved to: {output_path}")
        return merged_model
