FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git curl wget vim \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data outputs/checkpoints outputs/eval

EXPOSE 8080

CMD ["uvicorn", "pipeline.inference.serve:app", "--host", "0.0.0.0", "--port", "8080"]
