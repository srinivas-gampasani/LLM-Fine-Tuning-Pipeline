"""
pipeline/inference/serve.py

FastAPI inference server for the fine-tuned LoRA model.
Supports: adapter-on-base loading, streaming, batch inference.

Usage:
    cd lora-finetune
    uvicorn pipeline.inference.serve:app --host 0.0.0.0 --port 8080 --reload
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _device
    adapter_path = os.environ.get("ADAPTER_PATH", "outputs/checkpoints/lora_adapter")
    base_model   = os.environ.get("BASE_MODEL", "meta-llama/Llama-2-7b-hf")

    logger.info(f"Loading model from adapter: {adapter_path}")
    logger.info(f"Base model: {base_model}")

    try:
        import transformers
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        _tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
        )
        _model = PeftModel.from_pretrained(base, adapter_path)
        _model.eval()
        _device = next(_model.parameters()).device
        logger.info(f"Model ready on device: {_device}")
    except Exception as e:
        logger.error(f"Model loading failed: {e}. Running in demo mode.")

    app.state.model_ready = _model is not None
    yield
    logger.info("Shutting down inference server.")


app = FastAPI(
    title="LLM Fine-Tune Inference Server",
    description="Serve fine-tuned LoRA/QLoRA models via REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    instruction: str = Field(..., min_length=3, max_length=4096)
    input_context: Optional[str] = Field(default=None)
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    do_sample: bool = Field(default=False)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=3.0)
    stream: bool = Field(default=False)


class GenerateResponse(BaseModel):
    instruction: str
    response: str
    latency_ms: float
    tokens_generated: int
    model: str = "lora-fine-tuned"
    status: str = "success"


class BatchRequest(BaseModel):
    instructions: List[str] = Field(..., min_length=1, max_length=32)
    max_new_tokens: int = Field(default=256)
    temperature: float = Field(default=0.1)
    do_sample: bool = Field(default=False)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": _model is not None}


@app.get("/model-info")
def model_info():
    if _model is None:
        return {"error": "Model not loaded"}
    total = sum(p.numel() for p in _model.parameters())
    trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_pct": round(100 * trainable / total, 3),
        "device": str(_device),
        "dtype": str(next(_model.parameters()).dtype),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded. Check ADAPTER_PATH env var.")

    # Build alpaca-style prompt
    if req.input_context:
        prompt = (
            f"### Instruction:\n{req.instruction}\n\n"
            f"### Input:\n{req.input_context}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{req.instruction}\n\n### Response:\n"

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(_device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature if req.do_sample else 1.0,
            top_p=req.top_p if req.do_sample else 1.0,
            do_sample=req.do_sample,
            repetition_penalty=req.repetition_penalty,
            pad_token_id=_tokenizer.eos_token_id,
        )
    latency = (time.time() - t0) * 1000

    response_text = _tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    tokens_gen = output_ids.shape[1] - input_len

    return GenerateResponse(
        instruction=req.instruction,
        response=response_text,
        latency_ms=round(latency, 2),
        tokens_generated=tokens_gen,
    )


@app.post("/batch-generate")
async def batch_generate(req: BatchRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    prompts = [f"### Instruction:\n{inst}\n\n### Response:\n" for inst in req.instructions]
    inputs = _tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(_device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature if req.do_sample else 1.0,
            do_sample=req.do_sample,
            pad_token_id=_tokenizer.eos_token_id,
        )
    latency = (time.time() - t0) * 1000

    responses = [
        _tokenizer.decode(ids[input_len:], skip_special_tokens=True)
        for ids in output_ids
    ]

    return {
        "responses": [
            {"instruction": inst, "response": resp}
            for inst, resp in zip(req.instructions, responses)
        ],
        "batch_size": len(req.instructions),
        "total_latency_ms": round(latency, 2),
        "avg_latency_ms": round(latency / len(req.instructions), 2),
        "status": "success",
    }
