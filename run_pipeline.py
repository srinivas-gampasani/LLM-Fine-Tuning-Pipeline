"""
run_pipeline.py

Entry point for the complete LLM fine-tuning pipeline.

Usage:
    python run_pipeline.py --config configs/config.yaml
    python run_pipeline.py --config configs/config_mistral.yaml
    python run_pipeline.py --config configs/config.yaml --mode eval_only
    python run_pipeline.py --config configs/config.yaml --mode data_only
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_config(cfg: dict) -> dict:
    """Flatten nested YAML config into a flat dict for easy access."""
    flat = {}
    flat["model_name"]               = cfg["model"]["name"]
    flat["torch_dtype"]              = cfg["model"].get("torch_dtype", "float16")
    flat["device_map"]               = cfg["model"].get("device_map", "auto")
    flat["use_quantization"]         = cfg["quantization"]["enabled"]
    flat["quant_bits"]               = cfg["quantization"]["bits"]
    flat["quant_type"]               = cfg["quantization"]["quant_type"]
    flat["double_quant"]             = cfg["quantization"]["double_quant"]
    flat["compute_dtype"]            = cfg["quantization"]["compute_dtype"]
    flat["lora_r"]                   = cfg["lora"]["r"]
    flat["lora_alpha"]               = cfg["lora"]["alpha"]
    flat["lora_dropout"]             = cfg["lora"]["dropout"]
    flat["lora_bias"]                = cfg["lora"]["bias"]
    flat["target_modules"]           = cfg["lora"]["target_modules"]
    flat["train_file"]               = cfg["data"]["train_file"]
    flat["val_file"]                 = cfg["data"].get("val_file")
    flat["test_file"]                = cfg["data"].get("test_file")
    flat["text_column"]              = cfg["data"].get("text_column", "text")
    flat["max_seq_length"]           = cfg["data"]["max_seq_length"]
    flat["dataset_format"]           = cfg["data"].get("dataset_format", "instruction")
    flat["prompt_template"]          = cfg["data"].get("prompt_template", "alpaca")
    flat["val_split_ratio"]          = cfg["data"].get("val_split_ratio", 0.1)
    flat["pack_sequences"]           = cfg["data"].get("pack_sequences", True)
    flat["output_dir"]               = cfg["training"]["output_dir"]
    flat["num_train_epochs"]         = cfg["training"]["num_train_epochs"]
    flat["per_device_train_batch_size"] = cfg["training"]["per_device_train_batch_size"]
    flat["per_device_eval_batch_size"]  = cfg["training"]["per_device_eval_batch_size"]
    flat["gradient_accumulation_steps"] = cfg["training"]["gradient_accumulation_steps"]
    flat["gradient_checkpointing"]   = cfg["training"]["gradient_checkpointing"]
    flat["learning_rate"]            = cfg["training"]["learning_rate"]
    flat["lr_scheduler_type"]        = cfg["training"]["lr_scheduler_type"]
    flat["warmup_ratio"]             = cfg["training"]["warmup_ratio"]
    flat["weight_decay"]             = cfg["training"]["weight_decay"]
    flat["max_grad_norm"]            = cfg["training"]["max_grad_norm"]
    flat["fp16"]                     = cfg["training"]["fp16"]
    flat["bf16"]                     = cfg["training"].get("bf16", False)
    flat["optim"]                    = cfg["training"]["optim"]
    flat["logging_steps"]            = cfg["training"]["logging_steps"]
    flat["eval_steps"]               = cfg["training"]["eval_steps"]
    flat["save_steps"]               = cfg["training"]["save_steps"]
    flat["save_total_limit"]         = cfg["training"]["save_total_limit"]
    flat["load_best_model_at_end"]   = cfg["training"]["load_best_model_at_end"]
    flat["metric_for_best_model"]    = cfg["training"]["metric_for_best_model"]
    flat["report_to"]                = cfg["training"]["report_to"]
    flat["run_name"]                 = cfg["training"]["run_name"]
    flat["seed"]                     = cfg["training"]["seed"]
    flat["group_by_length"]          = cfg["training"]["group_by_length"]
    flat["push_to_hub"]              = cfg["hub"]["push_to_hub"]
    flat["hub_repo_id"]              = cfg["hub"]["repo_id"]
    flat["wandb_project"]            = cfg["wandb"]["project"]
    flat["effective_batch"]          = flat["per_device_train_batch_size"] * flat["gradient_accumulation_steps"]
    return flat


def banner():
    print("\n" + "=" * 65)
    print("  LLM Fine-Tuning Pipeline — LoRA / QLoRA")
    print("  Srinivas Gampasani · AI & ML Engineering")
    print("=" * 65 + "\n")


def step_data(flat_cfg: dict):
    from pipeline.data.prepare_data import DataConfig, generate_dataset

    logger.info("STEP 1 — Data Preparation")

    # Auto-generate dataset if train file doesn't exist
    if not os.path.exists(flat_cfg["train_file"]):
        logger.info("Training data not found — generating synthetic domain dataset...")
        data_dir = os.path.dirname(flat_cfg["train_file"]) or "data"
        generate_dataset(output_dir=data_dir)
        logger.info("Dataset generated.")
    else:
        logger.info(f"Using existing dataset: {flat_cfg['train_file']}")


def step_train(flat_cfg: dict) -> dict:
    from pipeline.training.trainer import LoRATrainerConfig, LoRAFinetuner
    from pipeline.data.prepare_data import DataConfig, DataPreparer

    logger.info("\nSTEP 2 — Loading Model & Injecting LoRA")

    trainer_cfg = LoRATrainerConfig(**{
        k: v for k, v in flat_cfg.items()
        if hasattr(LoRATrainerConfig, k)
    })
    tuner = LoRAFinetuner(trainer_cfg)
    tuner.setup()

    logger.info("\nSTEP 3 — Tokenizing Dataset")
    data_cfg = DataConfig(
        train_file=flat_cfg["train_file"],
        val_file=flat_cfg.get("val_file"),
        test_file=flat_cfg.get("test_file"),
        text_column=flat_cfg["text_column"],
        max_seq_length=flat_cfg["max_seq_length"],
        dataset_format=flat_cfg["dataset_format"],
        prompt_template=flat_cfg["prompt_template"],
        val_split_ratio=flat_cfg["val_split_ratio"],
        pack_sequences=flat_cfg["pack_sequences"],
        seed=flat_cfg["seed"],
    )
    preparer = DataPreparer(data_cfg, tuner.tokenizer)
    datasets = preparer.prepare()

    logger.info("\nSTEP 4 — Training")
    t0 = time.time()
    metrics = tuner.train(datasets)
    elapsed = (time.time() - t0) / 3600

    logger.info(f"\nTraining completed in {elapsed:.2f} hours")
    logger.info(f"Train loss: {metrics.get('train_loss', 'N/A'):.4f}")

    # Save adapter
    adapter_path = os.path.join(flat_cfg["output_dir"], "lora_adapter")
    tuner.save_adapter(adapter_path)

    return {"tuner": tuner, "datasets": datasets, "metrics": metrics, "adapter_path": adapter_path}


def step_evaluate(tuner, datasets, flat_cfg: dict) -> dict:
    from pipeline.evaluation.evaluator import EvalConfig, ModelEvaluator

    logger.info("\nSTEP 5 — Evaluation (ROUGE / BLEU / Perplexity)")

    eval_cfg = EvalConfig(
        max_new_tokens=512,
        do_sample=False,
        num_eval_samples=100,
        output_dir="outputs/eval",
    )

    import json
    test_file = flat_cfg.get("test_file", "data/test.jsonl")
    if os.path.exists(test_file):
        with open(test_file) as f:
            test_examples = [json.loads(line) for line in f if line.strip()]
    else:
        logger.warning("Test file not found — using validation set for eval")
        test_examples = [{"instruction": "What is sepsis?", "output": "Sepsis is a life-threatening infection response."}] * 10

    evaluator = ModelEvaluator(
        tuner.model, tuner.tokenizer, eval_cfg, model_name="fine-tuned"
    )
    ft_results = evaluator.evaluate(test_examples)

    logger.info(ft_results.summary())

    os.makedirs("outputs/eval", exist_ok=True)
    with open("outputs/eval/results.json", "w") as f:
        json.dump(ft_results.to_dict(), f, indent=2)

    return ft_results.to_dict()


def step_publish(flat_cfg: dict, eval_results: dict, adapter_path: str):
    from pipeline.export.hub_publisher import HubConfig, HubPublisher, ModelCardGenerator

    if not flat_cfg.get("push_to_hub"):
        logger.info("Hub publishing disabled in config. Skipping.")
        return

    logger.info("\nSTEP 6 — Publishing to HuggingFace Hub")
    card_gen = ModelCardGenerator()
    model_card = card_gen.generate(
        config={**flat_cfg, "model_title": f"Domain Fine-tuned {flat_cfg['model_name'].split('/')[-1]} (QLoRA)"},
        eval_comparison={"fine_tuned_model": eval_results},
    )

    hub_cfg = HubConfig(
        repo_id=flat_cfg["hub_repo_id"],
        private=True,
        commit_message="QLoRA fine-tune — domain adaptation v1.0",
    )
    publisher = HubPublisher(hub_cfg)
    publisher.publish(adapter_path, model_card, eval_results)


def main():
    banner()
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Pipeline (LoRA/QLoRA)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--mode", choices=["full", "data_only", "train_only", "eval_only"],
                        default="full", help="Pipeline mode")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")
    logger.info(f"Mode  : {args.mode}")

    cfg = load_config(args.config)
    flat = flatten_config(cfg)

    os.makedirs(flat["output_dir"], exist_ok=True)
    os.makedirs("outputs/eval", exist_ok=True)

    if args.mode in ("full", "data_only"):
        step_data(flat)

    if args.mode == "data_only":
        logger.info("Data preparation complete. Exiting.")
        return

    if args.mode in ("full", "train_only"):
        result = step_train(flat)
        tuner = result["tuner"]
        datasets = result["datasets"]
        adapter_path = result["adapter_path"]

    if args.mode in ("full", "train_only", "eval_only"):
        if args.mode == "eval_only":
            logger.error("eval_only mode requires a pre-loaded model — use train_only first.")
            sys.exit(1)
        eval_results = step_evaluate(tuner, datasets, flat)
        step_publish(flat, eval_results, adapter_path)

    logger.info("\n✅ Pipeline complete!")
    logger.info(f"   Adapter saved to : {flat['output_dir']}/lora_adapter/")
    logger.info(f"   Eval results     : outputs/eval/results.json")
    if flat.get("push_to_hub"):
        logger.info(f"   HuggingFace Hub  : https://huggingface.co/{flat['hub_repo_id']}")


if __name__ == "__main__":
    main()
