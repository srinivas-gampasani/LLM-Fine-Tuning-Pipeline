#!/bin/bash
# ============================================================
#  LLM Fine-Tuning Pipeline — Quick Start
# ============================================================
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

echo ""
echo "============================================================"
echo "  LLM Fine-Tuning Pipeline — LoRA / QLoRA"
echo "  Srinivas Gampasani · AI & ML Engineering"
echo "============================================================"
echo ""

# Check GPU
if command -v nvidia-smi &>/dev/null; then
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}WARNING: No GPU detected. Training will be very slow on CPU.${NC}"
fi
echo ""

# Check .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}ACTION REQUIRED: Edit .env and set HF_TOKEN and WANDB_API_KEY${NC}"
    read -p "Press Enter once you've set your tokens..."
fi

# Install
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -r requirements.txt -q
echo -e "${GREEN}Done.${NC}"
echo ""

echo "Choose mode:"
echo "  1) full         — data + train + eval + publish"
echo "  2) data_only    — generate dataset only"
echo "  3) train_only   — train (assumes data exists)"
echo ""
read -p "Enter mode [1]: " mode_choice
case "$mode_choice" in
    2) MODE="data_only" ;;
    3) MODE="train_only" ;;
    *) MODE="full" ;;
esac

echo ""
echo -e "${BLUE}Config: configs/config.yaml | Mode: $MODE${NC}"
echo ""
python run_pipeline.py --config configs/config.yaml --mode "$MODE"
