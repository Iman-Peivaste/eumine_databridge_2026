"""
Smoke test — validates all critical dependencies load correctly.
Run with: python scripts/smoke_test.py
Expected: all checks print OK, GPU is detected, no import errors.
"""

import sys
import os

print("="*60)
print("EuMINe DataBridge — Environment Smoke Test")
print("="*60)

# --- Python version ---
print(f"\n[1] Python: {sys.version}")

# --- PyTorch + CUDA ---
import torch
print(f"\n[2] PyTorch version : {torch.__version__}")
print(f"    CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    CUDA version    : {torch.version.cuda}")
    print(f"    GPU device      : {torch.cuda.get_device_name(0)}")
    print(f"    VRAM total      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("    WARNING: CUDA not available — GPU training disabled")

# --- PyMatgen ---
from pymatgen.core import Structure
print(f"\n[3] pymatgen OK")

# --- MP API ---
from mp_api.client import MPRester
from dotenv import load_dotenv
load_dotenv()
mp_key = os.getenv("MP_API_KEY")
if mp_key:
    print(f"\n[4] MP API key loaded from .env (length: {len(mp_key)})")
else:
    print(f"\n[4] WARNING: MP_API_KEY not found in .env")

# --- JARVIS ---
from jarvis.core.atoms import Atoms as JarvisAtoms
print(f"\n[5] jarvis-tools OK")

# --- ALIGNN ---
try:
    import alignn
    print(f"\n[6] ALIGNN OK: version {alignn.__version__}")
except Exception as e:
    print(f"\n[6] ALIGNN ERROR: {e}")

# --- MACE ---
try:
    from mace.calculators import mace_mp
    print(f"\n[7] MACE-torch OK")
except Exception as e:
    print(f"\n[7] MACE ERROR: {e}")

# --- Torch Geometric ---
import torch_geometric
print(f"\n[8] torch-geometric: {torch_geometric.__version__}")

# --- scikit-learn ---
import sklearn
print(f"\n[9] scikit-learn: {sklearn.__version__}")

# --- XGBoost ---
import xgboost
print(f"\n[10] xgboost: {xgboost.__version__}")

# --- W&B ---
import wandb
print(f"\n[11] wandb: {wandb.__version__}")

# --- Optuna ---
import optuna
print(f"\n[12] optuna: {optuna.__version__}")

# --- Scoring formula validation ---
print(f"\n[13] Scoring formula check:")
BASELINE_MAE_EF = 0.2378
BASELINE_MAE_BG = 0.6414

def score_property(mae, baseline_mae):
    if mae < baseline_mae:
        return 10 + 10 * (baseline_mae - mae) / (baseline_mae - 0.01)
    else:
        return max(0, 10 * (1 - (mae - baseline_mae) / baseline_mae))

# Reproduce organizer test result: MAE_EF=0.1547, MAE_BG=0.4122 → 27.28 pts
score_ef = score_property(0.1547, BASELINE_MAE_EF)
score_bg = score_property(0.4122, BASELINE_MAE_BG)
total = score_ef + score_bg
print(f"    Reproducing OrganizerTest score:")
print(f"    EF score: {score_ef:.4f}")
print(f"    BG score: {score_bg:.4f}")
print(f"    Total   : {total:.4f}  (expected: 27.28)")
assert abs(total - 27.28) < 0.1, f"Scoring formula mismatch: got {total}"
print(f"    Scoring formula: VERIFIED")

print("\n" + "="*60)
print("Smoke test complete.")
print("="*60)
