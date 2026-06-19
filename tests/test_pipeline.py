"""
tests/test_pipeline.py

Comprehensive test suite for the LLM Fine-Tuning Pipeline.
Tests: data preparation, ROUGE/BLEU calculators, evaluator, hub publisher, config loading.
Run with: pytest tests/ -v
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Data Preparation Tests ────────────────────────────────────────────────────

class TestPromptFormatter:

    def setup_method(self):
        from pipeline.data.prepare_data import PromptFormatter
        self.PromptFormatter = PromptFormatter

    def test_alpaca_with_input(self):
        fmt = self.PromptFormatter("alpaca")
        result = fmt.format({
            "instruction": "Summarize this",
            "input": "Some context here",
            "output": "Summary output",
        })
        assert "### Instruction:" in result
        assert "### Input:" in result
        assert "### Response:" in result
        assert "Summary output" in result

    def test_alpaca_without_input(self):
        fmt = self.PromptFormatter("alpaca")
        result = fmt.format({
            "instruction": "What is sepsis?",
            "input": "",
            "output": "Sepsis is...",
        })
        assert "### Instruction:" in result
        assert "### Input:" not in result
        assert "Sepsis is..." in result

    def test_llama2_format(self):
        fmt = self.PromptFormatter("llama2")
        result = fmt.format({
            "instruction": "Explain LoRA",
            "output": "LoRA is...",
        })
        assert "[INST]" in result
        assert "[/INST]" in result
        assert "LoRA is..." in result

    def test_mistral_format(self):
        fmt = self.PromptFormatter("mistral")
        result = fmt.format({
            "instruction": "What is QLoRA?",
            "output": "QLoRA uses...",
        })
        assert "[INST]" in result
        assert "QLoRA uses..." in result

    def test_chatml_format(self):
        fmt = self.PromptFormatter("chatml")
        result = fmt.format({
            "system": "You are helpful.",
            "instruction": "Hello",
            "output": "Hi!",
        })
        assert "<|im_start|>" in result
        assert "<|im_end|>" in result

    def test_invalid_template_raises(self):
        with pytest.raises(ValueError):
            self.PromptFormatter("nonexistent_template")


class TestDataGeneration:

    def test_generate_dataset_creates_files(self, tmp_path):
        from pipeline.data.prepare_data import generate_dataset
        generate_dataset(output_dir=str(tmp_path), n_train=50, n_val=10, n_test=10)
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "val.jsonl").exists()
        assert (tmp_path / "test.jsonl").exists()
        assert (tmp_path / "data_card.json").exists()

    def test_generated_train_has_correct_count(self, tmp_path):
        from pipeline.data.prepare_data import generate_dataset
        generate_dataset(output_dir=str(tmp_path), n_train=80, n_val=10, n_test=10)
        with open(tmp_path / "train.jsonl") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 80

    def test_generated_examples_have_required_fields(self, tmp_path):
        from pipeline.data.prepare_data import generate_dataset
        generate_dataset(output_dir=str(tmp_path), n_train=20, n_val=5, n_test=5)
        with open(tmp_path / "train.jsonl") as f:
            for line in f:
                ex = json.loads(line)
                assert "instruction" in ex
                assert "output" in ex
                assert len(ex["instruction"]) > 5
                assert len(ex["output"]) > 10

    def test_data_card_json_valid(self, tmp_path):
        from pipeline.data.prepare_data import generate_dataset
        generate_dataset(output_dir=str(tmp_path), n_train=30, n_val=5, n_test=5)
        with open(tmp_path / "data_card.json") as f:
            card = json.load(f)
        assert card["splits"]["train"] == 30
        assert card["splits"]["val"] == 5


# ── ROUGE Calculator Tests ────────────────────────────────────────────────────

class TestROUGECalculator:

    def setup_method(self):
        from pipeline.evaluation.evaluator import ROUGECalculator
        self.rouge = ROUGECalculator()

    def test_identical_texts_score_one(self):
        text = "the patient has severe sepsis requiring immediate treatment"
        scores = self.rouge.compute_batch([text], [text])
        assert scores["rouge1"] == pytest.approx(1.0, abs=0.001)
        assert scores["rouge2"] == pytest.approx(1.0, abs=0.001)
        assert scores["rougeL"] == pytest.approx(1.0, abs=0.001)

    def test_completely_different_texts_score_zero(self):
        pred = "apple orange banana fruit"
        ref  = "quantum physics electron neutron"
        scores = self.rouge.compute_batch([pred], [ref])
        assert scores["rouge1"] == 0.0
        assert scores["rouge2"] == 0.0

    def test_partial_overlap(self):
        pred = "the patient has sepsis"
        ref  = "the patient has severe sepsis and requires antibiotics"
        scores = self.rouge.compute_batch([pred], [ref])
        assert 0.0 < scores["rouge1"] < 1.0
        assert 0.0 < scores["rougeL"] < 1.0

    def test_empty_prediction(self):
        scores = self.rouge.compute_batch([""], ["some reference text here"])
        assert scores["rouge1"] == 0.0

    def test_batch_averages_correctly(self):
        preds = ["the cat sat", "a b c"]
        refs  = ["the cat sat", "x y z"]
        scores = self.rouge.compute_batch(preds, refs)
        # First pair: 1.0, Second pair: 0.0 → avg ≈ 0.5
        assert abs(scores["rouge1"] - 0.5) < 0.1

    def test_rouge_order(self):
        pred = "a b c d e"
        ref  = "a b c d e f g"
        scores = self.rouge.compute_batch([pred], [ref])
        # rouge2 ≤ rouge1 always
        assert scores["rouge2"] <= scores["rouge1"] + 0.001


# ── BLEU Calculator Tests ─────────────────────────────────────────────────────

class TestBLEUCalculator:

    def setup_method(self):
        from pipeline.evaluation.evaluator import BLEUCalculator
        self.bleu = BLEUCalculator()

    def test_identical_text_high_score(self):
        text = "the quick brown fox jumps over the lazy dog"
        scores = self.bleu.compute([text], [text], max_n=4)
        assert scores["bleu1"] > 0.8

    def test_no_overlap_zero_score(self):
        scores = self.bleu.compute(["hello world"], ["foo bar baz"], max_n=2)
        assert scores["bleu1"] == 0.0

    def test_bleu4_leq_bleu1(self):
        pred = "the patient was diagnosed with sepsis"
        ref  = "the patient has been diagnosed with severe sepsis"
        scores = self.bleu.compute([pred], [ref], max_n=4)
        assert scores["bleu4"] <= scores["bleu1"]

    def test_batch_bleu(self):
        preds = ["the cat sat on the mat", "hello world today"]
        refs  = ["the cat sat on the mat", "goodbye world yesterday"]
        scores = self.bleu.compute(preds, refs, max_n=2)
        assert 0.0 < scores["bleu1"] <= 1.0


# ── EvalResults Tests ─────────────────────────────────────────────────────────

class TestEvalResults:

    def test_to_dict_all_fields(self):
        from pipeline.evaluation.evaluator import EvalResults
        r = EvalResults(
            rouge1_f1=0.72, rouge2_f1=0.51, rougeL_f1=0.68,
            bleu1=0.65, bleu4=0.30, bertscore_f1=0.88,
            perplexity=12.5, avg_latency_ms=320.0, num_samples=100,
            model_name="test-model",
        )
        d = r.to_dict()
        assert d["rouge1_f1"] == 0.72
        assert d["model_name"] == "test-model"
        assert "perplexity" in d

    def test_summary_contains_key_info(self):
        from pipeline.evaluation.evaluator import EvalResults
        r = EvalResults(rouge1_f1=0.75, model_name="ft-model", num_samples=50)
        summary = r.summary()
        assert "ft-model" in summary
        assert "ROUGE-1" in summary
        assert "0.7500" in summary


# ── Config Loading Tests ──────────────────────────────────────────────────────

class TestConfigLoading:

    def test_load_yaml_config(self, tmp_path):
        config_content = """
model:
  name: "meta-llama/Llama-2-7b-hf"
  torch_dtype: "float16"
  device_map: "auto"
quantization:
  enabled: true
  bits: 4
  quant_type: "nf4"
  double_quant: true
  compute_dtype: "float16"
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules: ["q_proj", "v_proj"]
data:
  train_file: "data/train.jsonl"
  max_seq_length: 2048
  dataset_format: "instruction"
  prompt_template: "alpaca"
  val_split_ratio: 0.1
  pack_sequences: true
training:
  output_dir: "outputs"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  learning_rate: 0.0002
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  weight_decay: 0.001
  max_grad_norm: 1.0
  fp16: true
  bf16: false
  optim: "paged_adamw_32bit"
  logging_steps: 10
  eval_steps: 100
  save_steps: 100
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  report_to: "none"
  run_name: "test-run"
  seed: 42
  group_by_length: true
hub:
  push_to_hub: false
  repo_id: "test/model"
  private: true
  commit_message: "test"
  save_adapter_only: true
wandb:
  project: "test"
  tags: []
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        assert cfg["model"]["name"] == "meta-llama/Llama-2-7b-hf"
        assert cfg["quantization"]["bits"] == 4
        assert cfg["lora"]["r"] == 16
        assert cfg["training"]["learning_rate"] == 0.0002


# ── Model Card Tests ──────────────────────────────────────────────────────────

class TestModelCard:

    def test_model_card_generates_valid_markdown(self):
        from pipeline.export.hub_publisher import ModelCardGenerator
        gen = ModelCardGenerator()
        card = gen.generate(
            config={
                "model_name": "meta-llama/Llama-2-7b-hf",
                "hub_repo_id": "test/model",
                "use_quantization": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "num_train_epochs": 3,
                "learning_rate": 2e-4,
                "effective_batch": 16,
                "optim": "paged_adamw_32bit",
                "gpu_memory": "~6GB",
                "domain": "healthcare",
                "domain_description": "Medical QA",
                "train_size": 800,
                "val_size": 100,
                "test_size": 100,
                "max_seq_length": 2048,
                "train_hours": 2.5,
                "model_title": "Test Model",
            },
        )
        assert "# Test Model" in card
        assert "LoRA" in card or "QLoRA" in card
        assert "```python" in card
        assert "PeftModel" in card
        assert "meta-llama/Llama-2-7b-hf" in card


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
