#!/bin/bash
# CataLIST Stage 2 — GPU installer
# Run: bash install.sh
# Time: ~10 minutes

set -e

echo "================================================"
echo "  CataLIST Stage 2 — Installing environment"
echo "================================================"

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found."
    echo "Install miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[1/4] Creating conda environment (catallist_stage2)..."
conda env create -f "$SCRIPT_DIR/environment.yml" --force

echo "[2/4] Installing eumine_databridge package..."
conda run -n catallist_stage2 pip install -e "$REPO_ROOT" --quiet

echo "[3/4] Adding matfed-api-template to path..."
MATFED="$REPO_ROOT/../hackathon_ref/matfed-api-template"
if [ -d "$MATFED" ]; then
    conda run -n catallist_stage2 pip install -e "$MATFED" --quiet
    echo "  matfed-api-template installed."
else
    echo "  WARNING: matfed-api-template not found at $MATFED"
    echo "  Clone it manually if needed."
fi

echo "[4/4] Running health check..."
conda run -n catallist_stage2 python "$SCRIPT_DIR/verify.py" \
    --model_path "$REPO_ROOT/models/full_retrain"

echo ""
echo "================================================"
echo "  Installation complete."
echo "  Activate with: conda activate catallist_stage2"
echo "================================================"
