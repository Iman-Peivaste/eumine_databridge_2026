#!/bin/bash
# Run official MatFed API compliance tests against LISTEuMINePredictor.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MATFED_TESTS="$ROOT/../hackathon_ref/matfed-api-template"

echo "============================================"
echo "MatFed API v1 Compliance Test Runner"
echo "============================================"
echo "Predictor : eumine_databridge.matfed.predictor.LISTEuMINePredictor"
echo "Tests     : $MATFED_TESTS/tests/test_interface.py"
echo ""

export MY_PREDICTOR="eumine_databridge.matfed.predictor.LISTEuMINePredictor"

if [ -d "$ROOT/models/full_retrain/alignn_ef_full" ]; then
  export MATFED_MODEL_PATH="$ROOT/models/full_retrain"
else
  export MATFED_MODEL_PATH="$ROOT/models/ensemble"
fi

export PYTHONPATH="$ROOT/src:$MATFED_TESTS:$PYTHONPATH"

echo "Model path: $MATFED_MODEL_PATH"
echo ""

cd "$MATFED_TESTS"
pytest tests/test_interface.py -v --tb=short --no-header

echo ""
echo "============================================"
echo "If all tests passed: submission is compliant"
echo "============================================"
