#!/bin/bash
# CataLIST Stage 2 — CPU-only installer (fallback)
# Use this if no NVIDIA GPU is available
# Run: bash install_cpu.sh

set -e

echo "================================================"
echo "  CataLIST Stage 2 — CPU-only install"
echo "================================================"

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[1/4] Creating CPU conda environment..."
conda env create -f "$SCRIPT_DIR/environment_cpu.yml" --force

echo "[2/4] Installing eumine_databridge package..."
conda run -n catallist_stage2_cpu pip install -e "$REPO_ROOT" --quiet

echo "[3/4] Adding matfed-api-template..."
MATFED="$REPO_ROOT/../hackathon_ref/matfed-api-template"
if [ -d "$MATFED" ]; then
    conda run -n catallist_stage2_cpu pip install -e "$MATFED" --quiet
fi

echo "[4/4] Running CPU health check..."
conda run -n catallist_stage2_cpu python "$SCRIPT_DIR/verify.py" \
    --model_path "$REPO_ROOT/models/full_retrain" \
    --cpu_only

echo ""
echo "================================================"
echo "  CPU installation complete."
echo "  Activate with: conda activate catallist_stage2_cpu"
echo "  NOTE: Inference will be slower on CPU (~2 min/structure)"
echo "================================================"
