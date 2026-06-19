"""
pipeline/evaluation/evaluator.py

Evaluation suite: ROUGE, BLEU, BERTScore, Perplexity.
Compares fine-tuned model vs. base model on held-out test set.
"""
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    do_sample: bool = False
    repetition_penalty: float = 1.1
    num_eval_samples: int = 200
    batch_size: int = 4
    output_dir: str = "outputs/eval"
    metrics: List[str] = field(default_factory=lambda: ["rouge", "bleu", "perplexity"])


@dataclass
class EvalResults:
    rouge1_f1: float = 0.0
    rouge2_f1: float = 0.0
    rougeL_f1: float = 0.0
    bleu1: float = 0.0
    bleu4: float = 0.0
    bertscore_f1: float = 0.0
    perplexity: float = 0.0
    avg_latency_ms: float = 0.0
    num_samples: int = 0
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rouge1_f1": round(self.rouge1_f1, 4),
            "rouge2_f1": round(self.rouge2_f1, 4),
            "rougeL_f1": round(self.rougeL_f1, 4),
            "bleu1": round(self.bleu1, 4),
            "bleu4": round(self.bleu4, 4),
            "bertscore_f1": round(self.bertscore_f1, 4),
            "perplexity": round(self.perplexity, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "num_samples": self.num_samples,
            "model_name": self.model_name,
        }

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  EVALUATION RESULTS — {self.model_name}\n"
            f"{'='*55}\n"
            f"  ROUGE-1 F1       : {self.rouge1_f1:.4f} ({self.rouge1_f1*100:.1f}%)\n"
            f"  ROUGE-2 F1       : {self.rouge2_f1:.4f} ({self.rouge2_f1*100:.1f}%)\n"
            f"  ROUGE-L F1       : {self.rougeL_f1:.4f} ({self.rougeL_f1*100:.1f}%)\n"
            f"  BLEU-1           : {self.bleu1:.4f}\n"
            f"  BLEU-4           : {self.bleu4:.4f}\n"
            f"  BERTScore F1     : {self.bertscore_f1:.4f}\n"
            f"  Perplexity       : {self.perplexity:.2f}\n"
            f"  Avg Latency      : {self.avg_latency_ms:.1f}ms\n"
            f"  Samples evaluated: {self.num_samples}\n"
            f"{'='*55}"
        )


class TextGenerator:
    """Wraps model for batched text generation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: EvalConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

    def generate(self, prompts: List[str]) -> Tuple[List[str], float]:
        """Generate text for a list of prompts. Returns (outputs, avg_latency_ms)."""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.do_sample else 1.0,
                top_p=self.config.top_p if self.config.do_sample else 1.0,
                do_sample=self.config.do_sample,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = (time.time() - t0) * 1000 / len(prompts)

        # Decode only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        decoded = self.tokenizer.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
        )
        return decoded, latency


class ROUGECalculator:
    """Computes ROUGE-1, ROUGE-2, ROUGE-L without external rouge_score dependency."""

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    @staticmethod
    def _ngrams(tokens: List[str], n: int) -> Dict[str, int]:
        ngram_dict: Dict[str, int] = {}
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i+n])
            ngram_dict[gram] = ngram_dict.get(gram, 0) + 1
        return ngram_dict

    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def _ngram_f1(self, pred: str, ref: str, n: int) -> Tuple[float, float, float]:
        pred_tokens = self._tokenize(pred)
        ref_tokens = self._tokenize(ref)
        pred_ngrams = self._ngrams(pred_tokens, n)
        ref_ngrams = self._ngrams(ref_tokens, n)

        if not pred_ngrams or not ref_ngrams:
            return 0.0, 0.0, 0.0

        overlap = sum(min(pred_ngrams.get(g, 0), ref_ngrams.get(g, 0)) for g in ref_ngrams)
        precision = overlap / sum(pred_ngrams.values()) if pred_ngrams else 0.0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def _rougeL_f1(self, pred: str, ref: str) -> float:
        pred_tokens = self._tokenize(pred)
        ref_tokens = self._tokenize(ref)
        if not pred_tokens or not ref_tokens:
            return 0.0
        lcs = self._lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def compute_batch(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        r1_f1s, r2_f1s, rL_f1s = [], [], []
        for pred, ref in zip(predictions, references):
            _, _, f1_1 = self._ngram_f1(pred, ref, 1)
            _, _, f1_2 = self._ngram_f1(pred, ref, 2)
            fL = self._rougeL_f1(pred, ref)
            r1_f1s.append(f1_1)
            r2_f1s.append(f1_2)
            rL_f1s.append(fL)
        return {
            "rouge1": float(np.mean(r1_f1s)),
            "rouge2": float(np.mean(r2_f1s)),
            "rougeL": float(np.mean(rL_f1s)),
        }


class BLEUCalculator:
    """Corpus BLEU-1 and BLEU-4 implementation."""

    @staticmethod
    def _ngrams(tokens: List[str], n: int) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for i in range(len(tokens) - n + 1):
            g = " ".join(tokens[i:i+n])
            d[g] = d.get(g, 0) + 1
        return d

    def compute(
        self, predictions: List[str], references: List[str], max_n: int = 4
    ) -> Dict[str, float]:
        scores = {}
        for n in range(1, max_n + 1):
            total_clip, total_pred = 0, 0
            bp_num, bp_den = 0, 0
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.lower().split()
                ref_tokens = ref.lower().split()
                pred_ngrams = self._ngrams(pred_tokens, n)
                ref_ngrams = self._ngrams(ref_tokens, n)
                clip = sum(min(pred_ngrams.get(g, 0), ref_ngrams.get(g, 0)) for g in pred_ngrams)
                total_clip += clip
                total_pred += max(sum(pred_ngrams.values()), 1)
                bp_num += len(pred_tokens)
                bp_den += len(ref_tokens)

            precision = total_clip / total_pred if total_pred > 0 else 0.0
            bp = min(1.0, math.exp(1 - bp_den / bp_num)) if bp_num > 0 else 0.0
            scores[f"bleu{n}"] = bp * precision

        return scores


class PerplexityCalculator:
    """Compute model perplexity on reference texts."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def compute(self, texts: List[str], max_length: int = 1024) -> float:
        self.model.eval()
        nlls = []

        for text in tqdm(texts, desc="Computing perplexity"):
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**enc, labels=enc["input_ids"])
                nll = outputs.loss.item()

            if not math.isnan(nll) and not math.isinf(nll):
                nlls.append(nll)

        mean_nll = float(np.mean(nlls)) if nlls else float("inf")
        return math.exp(mean_nll)


class ModelEvaluator:
    """
    Full evaluation pipeline: generates predictions → computes all metrics.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: EvalConfig,
        model_name: str = "fine-tuned",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_name = model_name
        self.generator = TextGenerator(model, tokenizer, config)
        self.rouge = ROUGECalculator()
        self.bleu = BLEUCalculator()
        self.ppl_calc = PerplexityCalculator(model, tokenizer)

    def _build_prompts(self, examples: List[Dict]) -> Tuple[List[str], List[str]]:
        prompts, references = [], []
        for ex in examples:
            instruction = ex.get("instruction", ex.get("input", ""))
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            prompts.append(prompt)
            references.append(ex.get("output", ""))
        return prompts, references

    def evaluate(self, test_examples: List[Dict]) -> EvalResults:
        samples = test_examples[:self.config.num_eval_samples]
        prompts, references = self._build_prompts(samples)

        logger.info(f"Evaluating {len(samples)} samples — model: {self.model_name}")

        # Generate predictions in batches
        predictions = []
        latencies = []
        bs = self.config.batch_size
        for i in tqdm(range(0, len(prompts), bs), desc="Generating"):
            batch_prompts = prompts[i:i+bs]
            batch_preds, lat = self.generator.generate(batch_prompts)
            predictions.extend(batch_preds)
            latencies.append(lat)

        avg_latency = float(np.mean(latencies)) if latencies else 0.0

        results = EvalResults(model_name=self.model_name, num_samples=len(samples))
        results.avg_latency_ms = avg_latency

        if "rouge" in self.config.metrics:
            rouge_scores = self.rouge.compute_batch(predictions, references)
            results.rouge1_f1 = rouge_scores["rouge1"]
            results.rouge2_f1 = rouge_scores["rouge2"]
            results.rougeL_f1 = rouge_scores["rougeL"]

        if "bleu" in self.config.metrics:
            bleu_scores = self.bleu.compute(predictions, references)
            results.bleu1 = bleu_scores.get("bleu1", 0.0)
            results.bleu4 = bleu_scores.get("bleu4", 0.0)

        if "perplexity" in self.config.metrics:
            ref_texts = [p + r for p, r in zip(prompts, references)]
            results.perplexity = self.ppl_calc.compute(ref_texts[:50])  # Limit for speed

        return results

    def compare(
        self,
        base_results: EvalResults,
        ft_results: EvalResults,
        output_dir: str = "outputs/eval",
    ) -> Dict[str, Any]:
        """Generate comparison report between base and fine-tuned model."""
        os.makedirs(output_dir, exist_ok=True)

        comparison = {
            "base_model": base_results.to_dict(),
            "fine_tuned_model": ft_results.to_dict(),
            "improvements": {
                "rouge1_delta": round(ft_results.rouge1_f1 - base_results.rouge1_f1, 4),
                "rouge2_delta": round(ft_results.rouge2_f1 - base_results.rouge2_f1, 4),
                "rougeL_delta": round(ft_results.rougeL_f1 - base_results.rougeL_f1, 4),
                "bleu4_delta": round(ft_results.bleu4 - base_results.bleu4, 4),
                "perplexity_delta": round(ft_results.perplexity - base_results.perplexity, 2),
            }
        }

        out_path = os.path.join(output_dir, "eval_comparison.json")
        with open(out_path, "w") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison report saved to: {out_path}")
        return comparison
