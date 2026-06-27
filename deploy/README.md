# CataLIST — Stage 2 Deployment Guide

Get the model running in 5 minutes on any machine.

## Requirements

- Linux or macOS (Windows untested)
- conda or miniconda installed
- 8 GB RAM minimum
- NVIDIA GPU recommended (CPU works, slower)

## Quick Start

### GPU machine (recommended)
```bash
git clone https://github.com/Iman-Peivaste/eumine_databridge_2026.git
cd eumine_databridge_2026
bash deploy/install.sh
```

### CPU-only machine (fallback)
```bash
bash deploy/install_cpu.sh
```

## Verify Everything Works

```bash
conda activate catallist_stage2
python deploy/verify.py --model_path models/full_retrain
```

Expected output:
```
7/7 checks passed
STATUS: READY FOR STAGE 2 SPRINT
```

## Federation Sprint Usage

```python
from eumine_databridge.matfed.predictor import LISTEuMINePredictor
from eumine_databridge.matfed.federation import FederatedEnsemble

# Load our model
our_model = LISTEuMINePredictor()
our_model.load_model("models/full_retrain")

# Add partner team models
fed = FederatedEnsemble()
fed.add_predictor(our_model, "CataLIST")
# fed.add_predictor(their_model, "TakeMe2Romania")
# fed.add_predictor(their_model2, "ProphX")

# Fit on calibration set (provided by organizers at sprint)
result = fed.fit(
    cal_structures=cal_structures,
    cal_ef=cal_ef,
    cal_bg=cal_bg,
    n_trials=200,
)

# Predict on test structures
predictions = fed.predict(test_structures)
```

## CPU Inference Times (approximate)

| Operation | GPU (A4000) | CPU only |
|---|---|---|
| Load model | 8s | 12s |
| Predict 150 structures | 30s | 4 min |
| Federation fit (200 trials) | 3 min | 8 min |
| Full sprint pipeline | 10 min | 20 min |

## Troubleshooting

**ALIGNN import error:** `pip install alignn==2026.4.2`

**MACE CUDA mismatch:** Use `install_cpu.sh` instead

**matfed_api not found:**
```bash
cd ../hackathon_ref/matfed-api-template && pip install -e .
```

**Out of memory:** Use `--cpu_only` flag in verify.py
