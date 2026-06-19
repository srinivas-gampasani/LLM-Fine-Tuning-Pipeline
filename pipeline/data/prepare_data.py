"""
pipeline/data/prepare_data.py

Data curation, formatting, and preprocessing for LLM fine-tuning.
Supports: Alpaca, ChatML, Llama2, Mistral prompt templates.
"""
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


# ── Prompt Templates ──────────────────────────────────────────────────────────

PROMPT_TEMPLATES = {
    "alpaca": {
        "with_input": (
            "Below is an instruction that describes a task, paired with an input that provides "
            "further context. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        ),
        "without_input": (
            "Below is an instruction that describes a task. Write a response that appropriately "
            "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
        ),
    },
    "chatml": {
        "format": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    },
    "llama2": {
        "format": (
            "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST] {output} </s>"
        ),
        "default_system": (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as "
            "possible, while being safe. Your answers should not include any harmful, unethical, "
            "racist, sexist, toxic, dangerous, or illegal content."
        ),
    },
    "mistral": {
        "format": "<s>[INST] {instruction} [/INST] {output}</s>"
    },
}


@dataclass
class DataConfig:
    train_file: str
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    text_column: str = "text"
    max_seq_length: int = 2048
    dataset_format: str = "instruction"   # instruction | text | chat
    prompt_template: str = "alpaca"
    val_split_ratio: float = 0.1
    pack_sequences: bool = True
    seed: int = 42


class PromptFormatter:
    """Converts raw dataset examples into formatted prompt strings."""

    def __init__(self, template: str = "alpaca"):
        self.template = template
        if template not in PROMPT_TEMPLATES:
            raise ValueError(f"Unknown template '{template}'. Choose from: {list(PROMPT_TEMPLATES.keys())}")

    def format(self, example: Dict[str, Any]) -> str:
        tmpl = PROMPT_TEMPLATES[self.template]

        if self.template == "alpaca":
            if example.get("input", "").strip():
                return tmpl["with_input"].format(
                    instruction=example.get("instruction", ""),
                    input=example.get("input", ""),
                    output=example.get("output", ""),
                )
            return tmpl["without_input"].format(
                instruction=example.get("instruction", ""),
                output=example.get("output", ""),
            )

        elif self.template == "chatml":
            return tmpl["format"].format(
                system=example.get("system", "You are a helpful assistant."),
                instruction=example.get("instruction", example.get("input", "")),
                output=example.get("output", ""),
            )

        elif self.template == "llama2":
            return tmpl["format"].format(
                system=example.get("system", tmpl["default_system"]),
                instruction=example.get("instruction", example.get("input", "")),
                output=example.get("output", ""),
            )

        elif self.template == "mistral":
            return tmpl["format"].format(
                instruction=example.get("instruction", example.get("input", "")),
                output=example.get("output", ""),
            )

        raise ValueError(f"Unhandled template: {self.template}")


class DataPreparer:
    """
    Full data preparation pipeline:
      1. Load JSONL / CSV / HuggingFace dataset
      2. Format with prompt template
      3. Tokenize + truncate
      4. (Optional) Pack sequences for efficiency
      5. Return train/val/test DatasetDicts
    """

    def __init__(self, config: DataConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.formatter = PromptFormatter(config.prompt_template)

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_raw(self) -> DatasetDict:
        splits = {}

        for split, path in [("train", self.config.train_file),
                             ("validation", self.config.val_file),
                             ("test", self.config.test_file)]:
            if not path or not os.path.exists(path):
                continue
            ext = Path(path).suffix.lower()
            if ext == ".jsonl" or ext == ".json":
                ds = load_dataset("json", data_files=path, split="train")
            elif ext == ".csv":
                ds = load_dataset("csv", data_files=path, split="train")
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            splits[split] = ds
            logger.info(f"Loaded {split}: {len(ds):,} examples from {path}")

        if "train" not in splits:
            raise FileNotFoundError(f"Training file not found: {self.config.train_file}")

        if "validation" not in splits:
            logger.info(f"No val file found — splitting {self.config.val_split_ratio:.0%} from train")
            train_val = splits["train"].train_test_split(
                test_size=self.config.val_split_ratio,
                seed=self.config.seed
            )
            splits["train"] = train_val["train"]
            splits["validation"] = train_val["test"]

        return DatasetDict(splits)

    # ── Formatting ───────────────────────────────────────────────────────────

    def _format_example(self, example: Dict) -> Dict:
        if self.config.dataset_format == "text":
            text = example.get(self.config.text_column, "")
        else:
            text = self.formatter.format(example)
        return {"text": text}

    # ── Tokenization ─────────────────────────────────────────────────────────

    def _tokenize(self, examples: Dict) -> Dict:
        result = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    # ── Sequence Packing ─────────────────────────────────────────────────────

    def _pack_sequences(self, dataset: Dataset) -> Dataset:
        """
        Concatenate short sequences and split at max_seq_length boundaries.
        This maximises GPU utilisation by eliminating padding waste.
        """
        logger.info("Packing sequences...")
        all_input_ids = []
        all_attention_masks = []

        for ex in dataset:
            all_input_ids.extend(ex["input_ids"] + [self.tokenizer.eos_token_id])
            all_attention_masks.extend(ex["attention_mask"] + [1])

        packed_ids, packed_masks = [], []
        for i in range(0, len(all_input_ids), self.config.max_seq_length):
            chunk_ids = all_input_ids[i:i + self.config.max_seq_length]
            chunk_masks = all_attention_masks[i:i + self.config.max_seq_length]
            if len(chunk_ids) == self.config.max_seq_length:
                packed_ids.append(chunk_ids)
                packed_masks.append(chunk_masks)

        logger.info(f"Packed into {len(packed_ids):,} full-length sequences")
        return Dataset.from_dict({
            "input_ids": packed_ids,
            "attention_mask": packed_masks,
            "labels": packed_ids,
        })

    # ── Main ─────────────────────────────────────────────────────────────────

    def prepare(self) -> DatasetDict:
        raw = self.load_raw()

        # Format prompts
        formatted = raw.map(
            self._format_example,
            remove_columns=raw["train"].column_names,
            desc="Formatting prompts",
        )

        # Tokenize
        tokenized = formatted.map(
            self._tokenize,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

        # Pack (train only)
        if self.config.pack_sequences:
            tokenized["train"] = self._pack_sequences(tokenized["train"])

        logger.info(f"Train: {len(tokenized['train']):,} | Val: {len(tokenized['validation']):,}")
        return tokenized


# ── Dataset Generation (synthetic domain data for demo) ──────────────────────

DOMAIN_QA_PAIRS = [
    {
        "instruction": "What is the recommended first-line antibiotic for community-acquired pneumonia in outpatient adults?",
        "output": "For outpatient adults with community-acquired pneumonia (CAP) and no comorbidities, the recommended first-line treatment is amoxicillin 1g PO TID for 5 days. For those with comorbidities (COPD, diabetes, heart/liver/renal disease), a respiratory fluoroquinolone (levofloxacin 750mg daily or moxifloxacin 400mg daily) or a beta-lactam plus macrolide combination is preferred. Doxycycline 100mg BID is an alternative for penicillin-allergic patients.",
    },
    {
        "instruction": "Explain the mechanism of action of GLP-1 receptor agonists for type 2 diabetes management.",
        "output": "GLP-1 receptor agonists (e.g., semaglutide, liraglutide) work by mimicking the incretin hormone glucagon-like peptide-1. They bind to GLP-1 receptors on pancreatic beta cells, stimulating glucose-dependent insulin secretion. Simultaneously, they suppress glucagon release from alpha cells, slow gastric emptying to reduce postprandial glucose spikes, and act on the hypothalamus to reduce appetite. These mechanisms collectively lower HbA1c (typically 1–2%), promote weight loss (3–15 kg), and have demonstrated cardiovascular benefits in trials (LEADER, SUSTAIN-6).",
    },
    {
        "instruction": "What are the diagnostic criteria for septic shock per Sepsis-3 definition?",
        "output": "Per the Sepsis-3 consensus (Singer et al., JAMA 2016), septic shock is defined as a subset of sepsis with circulatory and cellular/metabolic dysfunction associated with higher mortality risk. Clinical criteria require ALL three: (1) underlying sepsis (life-threatening organ dysfunction due to dysregulated host response to infection); (2) vasopressor requirement to maintain MAP ≥ 65 mmHg despite adequate fluid resuscitation; (3) serum lactate > 2 mmol/L (>18 mg/dL) despite adequate fluid resuscitation. The in-hospital mortality for septic shock exceeds 40%.",
    },
    {
        "instruction": "What is the Wells score and how is it used in DVT diagnosis?",
        "output": "The Wells score is a clinical prediction rule for pre-test probability of deep vein thrombosis (DVT). Scoring: Active cancer (+1), paralysis/paresis/recent plaster cast (+1), bedridden >3 days or major surgery within 12 weeks (+1), localized tenderness along deep venous system (+1), entire leg swollen (+1), calf swelling >3cm vs asymptomatic side (+1), pitting edema (affected side only) (+1), collateral superficial veins (+1), prior DVT (+1), alternative diagnosis at least as likely (-2). Interpretation: Score ≥2 = high probability → proceed to duplex ultrasound; Score <2 = low probability → D-dimer test; if D-dimer negative, DVT excluded.",
    },
    {
        "instruction": "Describe appropriate heparin dosing for acute pulmonary embolism.",
        "output": "For acute PE, unfractionated heparin (UFH) is initiated with a weight-based bolus of 80 units/kg IV (max 10,000 units), followed by infusion at 18 units/kg/hour. Adjust to maintain aPTT 60–100 seconds (corresponding to anti-Xa 0.3–0.7 IU/mL). Alternatively, LMWH: enoxaparin 1 mg/kg SC q12h or 1.5 mg/kg SC q24h (avoid in severe renal impairment, CrCl <30). For massive PE with hemodynamic compromise, systemic thrombolysis with alteplase 100 mg IV over 2 hours should be considered. Transition to oral anticoagulation (DOAC preferred) once hemodynamically stable.",
    },
    {
        "instruction": "What is HIPAA's minimum necessary standard and how does it apply to clinical staff?",
        "output": "The HIPAA Minimum Necessary Standard (45 CFR §164.502(b)) requires covered entities and business associates to limit the use, disclosure, and requests of protected health information (PHI) to the minimum necessary to accomplish the intended purpose. For clinical staff, this means: accessing only the medical records of patients under your direct care; not reviewing records of family members, coworkers, or celebrities even if you have system access; limiting the scope of information shared in verbal handoffs to relevant clinical details only; and ensuring faxes/emails include only the PHI fields required for the specific task. Violations can result in sanctions ranging from verbal warning to termination and OIG referral.",
    },
    {
        "instruction": "How does LoRA (Low-Rank Adaptation) reduce the number of trainable parameters during LLM fine-tuning?",
        "output": "LoRA (Hu et al., 2021) addresses the high computational cost of full fine-tuning by decomposing weight updates into low-rank matrices. For a pretrained weight matrix W ∈ R^(d×k), LoRA freezes W and injects trainable matrices A ∈ R^(d×r) and B ∈ R^(r×k) where r << min(d,k). The adapted forward pass becomes: h = Wx + (BA)x * (α/r). During training, only A and B are updated — reducing trainable parameters from d×k to r×(d+k). For a 7B model with r=16, this reduces trainable params from ~7B to ~15M (99.8% reduction), while maintaining ~96% of full fine-tune performance. At inference, BA can be merged into W with zero latency cost.",
    },
    {
        "instruction": "What is QLoRA and how does it achieve 60% GPU memory reduction compared to full fine-tuning?",
        "output": "QLoRA (Dettmers et al., 2023) combines 4-bit NormalFloat (NF4) quantization with LoRA adapters to dramatically reduce memory requirements. Three key innovations: (1) 4-bit NF4 quantization — stores base model weights in 4-bit NF4 format (optimally distributed for normally distributed neural network weights), reducing base model memory by ~75% vs FP16; (2) Double quantization — quantizes the quantization constants themselves, saving an additional 0.37 bits/parameter; (3) Paged optimizers — uses NVIDIA unified memory to page optimizer states between GPU and CPU RAM, preventing OOM during gradient spikes. Combined effect: a Llama 2 7B fine-tune requires ~48GB GPU RAM in full FP16 vs ~6GB with QLoRA (87.5% reduction). Training uses BFloat16 compute with dequantization at each layer during the forward/backward pass.",
    },
    {
        "instruction": "Explain the difference between ROUGE-1, ROUGE-2, and ROUGE-L metrics for text generation evaluation.",
        "output": "ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between generated text and reference summaries. ROUGE-1 measures unigram (single word) overlap between candidate and reference — captures vocabulary coverage. ROUGE-2 measures bigram (two consecutive words) overlap — captures phrasal fluency and coherence. ROUGE-L measures the Longest Common Subsequence (LCS) — captures sentence-level structural similarity without requiring consecutive matches. All variants compute Precision (what fraction of candidate n-grams appear in reference), Recall (what fraction of reference n-grams appear in candidate), and F1. For fine-tuning evaluation, ROUGE-L F1 ≥ 0.40 is generally considered good for abstractive summarization tasks; for instruction following, ROUGE-1 F1 > 0.50 indicates strong performance.",
    },
    {
        "instruction": "What are the key considerations for selecting LoRA rank (r) for fine-tuning?",
        "output": "LoRA rank r controls the capacity of the adapter. Tradeoffs: Lower r (4–8) — fewer parameters, less expressive, suitable for simple domain adaptation or when data is limited (<1K examples); risk of underfitting for complex tasks. Medium r (16–32) — good balance; r=16 is the most common choice, covers most instruction-tuning tasks well. Higher r (64–128) — approaches full fine-tune expressiveness; suitable for large datasets and significant domain shift; increased memory and risk of overfitting on small datasets. General guidance: start with r=16, alpha=32; monitor train/eval loss gap for overfitting; reduce r if overfitting, increase if underfitting. For code generation, r=32+ often outperforms r=16. The paper recommends alpha=2r as a safe default scaling.",
    },
    {
        "instruction": "What is gradient checkpointing and when should it be used in LLM training?",
        "output": "Gradient checkpointing (also called activation recomputation) is a memory-saving technique that trades compute for memory. In normal training, all intermediate activations are stored during the forward pass for use in backward pass gradient computation — consuming O(L) memory where L is model depth. With gradient checkpointing, only checkpoint activations at certain layers are stored; others are recomputed during the backward pass from the nearest checkpoint. This reduces activation memory from O(L) to O(√L) at the cost of ~30–40% more compute. Use gradient_checkpointing=True whenever: GPU VRAM is insufficient for full batch; using QLoRA/LoRA on large models (7B+); batch_size > 1 with long sequences (>2048 tokens). Not needed when memory is abundant — it slows training unnecessarily. Always combine with gradient_accumulation_steps to maintain effective batch size.",
    },
    {
        "instruction": "How do you prevent catastrophic forgetting when fine-tuning LLMs?",
        "output": "Catastrophic forgetting occurs when fine-tuning on a narrow domain erases the model's general capabilities. Mitigation strategies: (1) LoRA/PEFT — by freezing base weights and training only adapters, general knowledge is preserved by design; most effective approach. (2) Replay/data mixing — include 10–30% of general instruction-following data (e.g., OpenHermes, ShareGPT) mixed with domain data in training batches. (3) Low learning rate — use lr ≤ 2e-4 with cosine decay; aggressive learning rates overwrite more pretrained knowledge. (4) Fewer epochs — 1–3 epochs for small datasets; more epochs increase forgetting risk. (5) Elastic Weight Consolidation (EWC) — adds a regularization term penalizing changes to weights important for prior tasks (computationally expensive for LLMs). (6) Evaluation monitoring — track general benchmarks (MMLU, HellaSwag) alongside domain metrics during training to detect degradation early.",
    },
]


def generate_dataset(
    output_dir: str = "pipeline/data",
    n_train: int = 800,
    n_val: int = 100,
    n_test: int = 100,
    seed: int = 42,
):
    """Generate a synthetic domain QA dataset for demonstration."""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Augment base pairs by paraphrasing instructions slightly
    augmented = []
    for item in DOMAIN_QA_PAIRS:
        augmented.append(item)
        # Add instruction variants
        prefixes = ["Please ", "Can you ", "I need to know: ", ""]
        suffixes = ["", " Please be detailed.", " Provide a clinical perspective.", " Use evidence-based guidelines."]
        for _ in range(n_train // len(DOMAIN_QA_PAIRS)):
            augmented.append({
                "instruction": random.choice(prefixes) + item["instruction"] + random.choice(suffixes),
                "output": item["output"],
                "input": "",
            })

    random.shuffle(augmented)
    all_data = augmented[:n_train + n_val + n_test]

    splits = {
        "train": all_data[:n_train],
        "val": all_data[n_train:n_train + n_val],
        "test": all_data[n_train + n_val:n_train + n_val + n_test],
    }

    for split_name, examples in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        logger.info(f"Wrote {len(examples)} examples to {out_path}")

    # Write data card
    card = {
        "dataset_name": "domain-instruction-tuning",
        "description": "Medical + AI/ML domain instruction-following dataset",
        "splits": {k: len(v) for k, v in splits.items()},
        "format": "alpaca",
        "columns": ["instruction", "input", "output"],
        "sources": ["clinical guidelines", "pharmacology references", "ML papers"],
        "generated": True,
    }
    with open(os.path.join(output_dir, "data_card.json"), "w") as f:
        json.dump(card, f, indent=2)

    print(f"\nDataset generated:")
    for split, examples in splits.items():
        print(f"  {split}: {len(examples):,} examples")
    return splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_dataset(output_dir="data")
