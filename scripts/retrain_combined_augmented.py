"""
Combined augmented retrain — v4.

Trains ALIGNN on all available data with per-source sample weights:
  1. Bridge Dataset train+val   (850 structures)  — weight 3.0
  2. Semiconductor augmentation (S2-1)             — weight 2.0
  3. Layered/2D structures      (S2-2A)            — weight 2.5
  4. Rare earth + complex oxide (S2-2B)            — weight 2.0

Key differences vs retrain_augmented.py (v3):
  - ALIGNN BG: 8 layers (more capacity for larger, harder dataset)
  - BG epochs: 600, EF epochs: 400, patience: 80 both
  - JARVIS->MP EF correction applied to ALL JARVIS-sourced data
  - JARVIS->MP BG correction skipped for layered structures
    (OptB88vdW BG is more accurate for layered/2D materials)
  - Saves to models/combined_retrain/
  - model_id: ALIGNN_MACE_ensemble_v4_combined

Run:
    python scripts/retrain_combined_augmented.py
"""

import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from pymatgen.core import Structure
from tqdm import tqdm
import pandas as pd

from eumine_databridge.data.loader import BridgeDataset
from eumine_databridge.data.combined_loader import CombinedDataset
from eumine_databridge.data.harmonizer import DatabaseHarmonizer
from eumine_databridge.models.alignn_config import get_ef_config, get_bg_config
from eumine_databridge.models.alignn_model import ALIGNNFineTuner
from eumine_databridge.models.alignn_data import structures_to_alignn_dataset
from eumine_databridge.models.mace_model import MACEPredictor
from eumine_databridge.models.ensemble import WeightedEnsemble, CalibrationLayer
from eumine_databridge.utils.metrics import compute_full_score

DATA    = ROOT / "data"
RAW     = DATA / "raw"
AUG     = DATA / "augmented"
MODELS  = ROOT / "models"
OUT_DIR = MODELS / "combined_retrain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load Bridge Dataset ────────────────────────────────────────────────────
print("\n[1] Loading Bridge Dataset (train + val)...")
train_ds = BridgeDataset(
    csv_path=RAW / "bridge_dataset_train.csv",
    structures_dir=RAW / "train_structures",
    split="train",
)
val_ds = BridgeDataset(
    csv_path=RAW / "bridge_dataset_val.csv",
    structures_dir=RAW / "val_structures",
    split="val",
)
combined = CombinedDataset(train_ds, val_ds)
bridge_structures = combined.get_structures()
bridge_ef, bridge_ef_ids = combined.get_targets("formation_energy_per_atom")
bridge_bg, bridge_bg_ids = combined.get_targets("band_gap")
print(f"  Bridge Dataset: {len(bridge_structures)} structures")

# ── 2. Load harmonizer ────────────────────────────────────────────────────────
print("\n[2] Loading JARVIS→MP harmonizer...")
harmonizer = DatabaseHarmonizer()
harmonizer.load(DATA / "processed" / "harmonizer_params.json")

# ── 3. Load augmented datasets ────────────────────────────────────────────────
print("\n[3] Loading augmented datasets...")

def load_aug_csv_and_cifs(
    csv_path: Path,
    cif_dir: Path,
    source_label: str,
    apply_ef_correction: bool = True,
    apply_bg_correction: bool = True,
):
    """Load a CSV+CIF augmented dataset, apply JARVIS→MP corrections."""
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found — skipping {source_label}")
        return [], [], [], []

    df = pd.read_csv(csv_path)
    structures, ef_vals, bg_vals, ids = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  Loading {source_label}"):
        mat_id   = str(row["material_id"])
        cif_path = cif_dir / f"{mat_id}.cif"
        if not cif_path.exists():
            continue
        try:
            struct = Structure.from_file(str(cif_path))
        except Exception:
            continue

        ef = float(row["formation_energy_per_atom"])
        bg = float(row["band_gap"])

        # Apply JARVIS→MP scale corrections for JARVIS-sourced entries
        src = str(row.get("source", ""))
        if "JARVIS" in src:
            if apply_ef_correction:
                ef = harmonizer.correct_jarvis_ef(np.array([ef]))[0]
            if apply_bg_correction:
                bg = harmonizer.correct_jarvis_bg(np.array([bg]))[0]

        structures.append(struct)
        ef_vals.append(ef)
        bg_vals.append(bg)
        ids.append(mat_id)

    print(f"    {source_label}: {len(structures)} structures loaded")
    return structures, ef_vals, bg_vals, ids


# 3a. Semiconductor augmentation (S2-1)
semi_structs, semi_ef, semi_bg, semi_ids = load_aug_csv_and_cifs(
    csv_path=AUG / "augmentation_dataset.csv",
    cif_dir=AUG / "structures",
    source_label="Semiconductor (S2-1)",
    apply_ef_correction=True,
    apply_bg_correction=True,
)

# 3b. Layered/2D structures (S2-2A) — EF corrected, BG NOT corrected
layered_structs, layered_ef, layered_bg, layered_ids = load_aug_csv_and_cifs(
    csv_path=AUG / "layered_structures" / "layered_dataset.csv",
    cif_dir=AUG / "layered_structures",
    source_label="Layered/2D (S2-2A)",
    apply_ef_correction=True,
    apply_bg_correction=False,  # OptB88vdW BG more accurate for layered
)

# 3c. Rare earth + complex oxides (S2-2B)
re_structs, re_ef, re_bg, re_ids = load_aug_csv_and_cifs(
    csv_path=AUG / "rare_earth_structures" / "rare_earth_dataset.csv",
    cif_dir=AUG / "rare_earth_structures",
    source_label="Rare earth + oxide (S2-2B)",
    apply_ef_correction=True,
    apply_bg_correction=True,
)

# ── 4. Combine with per-source sample weights ─────────────────────────────────
print("\n[4] Combining datasets with sample weights...")

all_structures = (
    bridge_structures + semi_structs + layered_structs + re_structs
)
all_ef  = bridge_ef  + semi_ef  + layered_ef  + re_ef
all_bg  = bridge_bg  + semi_bg  + layered_bg  + re_bg
all_ids = bridge_ef_ids + semi_ids + layered_ids + re_ids

n_bridge  = len(bridge_structures)
n_semi    = len(semi_structs)
n_layered = len(layered_structs)
n_re      = len(re_structs)

sample_weights = np.concatenate([
    np.full(n_bridge,  3.0),   # Bridge Dataset — closest to test distribution
    np.full(n_semi,    2.0),   # Semiconductor augmentation
    np.full(n_layered, 2.5),   # Layered/2D — Stage 2 OOD target
    np.full(n_re,      2.0),   # Rare earth + complex oxide
])

print(f"\n  Training set composition:")
print(f"    Bridge Dataset       : {n_bridge:5d}  (weight 3.0)")
print(f"    Semiconductor aug    : {n_semi:5d}  (weight 2.0)")
print(f"    Layered/2D           : {n_layered:5d}  (weight 2.5)")
print(f"    Rare earth + oxide   : {n_re:5d}  (weight 2.0)")
print(f"    Total                : {len(all_structures):5d}")

# ── 5. Internal val split for early stopping ──────────────────────────────────
INTERNAL_VAL_FRAC = 0.08
n_val_int = int(len(all_structures) * INTERNAL_VAL_FRAC)
rng = np.random.default_rng(42)
idx = np.arange(len(all_structures))
rng.shuffle(idx)
val_idx = idx[:n_val_int]
tr_idx  = idx[n_val_int:]

tr_structs  = [all_structures[i] for i in tr_idx]
vl_structs  = [all_structures[i] for i in val_idx]
tr_ef       = [all_ef[i]  for i in tr_idx]
vl_ef       = [all_ef[i]  for i in val_idx]
tr_bg       = [all_bg[i]  for i in tr_idx]
vl_bg       = [all_bg[i]  for i in val_idx]
tr_ids      = [all_ids[i] for i in tr_idx]
vl_ids      = [all_ids[i] for i in val_idx]

print(f"\n  Internal split: {len(tr_structs)} train / {len(vl_structs)} val")

# ── 6. Train ALIGNN EF ────────────────────────────────────────────────────────
print("\n[5] Training ALIGNN EF (4 layers, 400 epochs)...")
ef_cfg = get_ef_config()
ef_cfg.output_dir       = OUT_DIR / "alignn_ef_combined"
ef_cfg.wandb_run_name   = "alignn_ef_v4_combined"
ef_cfg.epochs           = 400
ef_cfg.patience         = 80

ef_trainer = ALIGNNFineTuner(ef_cfg)
ef_trainer.setup(
    train_dataset=structures_to_alignn_dataset(
        tr_structs, tr_ef, tr_ids, cutoff=ef_cfg.cutoff,
    ),
    val_dataset=structures_to_alignn_dataset(
        vl_structs, vl_ef, vl_ids, cutoff=ef_cfg.cutoff,
    ),
)
ef_trainer.train()

# ── 7. Train ALIGNN BG ────────────────────────────────────────────────────────
print("\n[6] Training ALIGNN BG (8 layers, 600 epochs)...")
bg_cfg = get_bg_config()
bg_cfg.output_dir       = OUT_DIR / "alignn_bg_combined"
bg_cfg.wandb_run_name   = "alignn_bg_v4_combined"
bg_cfg.epochs           = 600
bg_cfg.patience         = 80
bg_cfg.alignn_layers    = 8
bg_cfg.hidden_features  = 256

bg_trainer = ALIGNNFineTuner(bg_cfg)
bg_trainer.setup(
    train_dataset=structures_to_alignn_dataset(
        tr_structs, tr_bg, tr_ids, cutoff=bg_cfg.cutoff,
    ),
    val_dataset=structures_to_alignn_dataset(
        vl_structs, vl_bg, vl_ids, cutoff=bg_cfg.cutoff,
    ),
)
bg_trainer.train()

# ── 8. Evaluate on official val set ──────────────────────────────────────────
print("\n[7] Evaluating on official val set (150 samples)...")
ef_trainer.load_best_model()
bg_trainer.load_best_model()

val_structs_official = val_ds.get_structures()
val_ef_official, _   = val_ds.get_targets("formation_energy_per_atom")
val_bg_official, _   = val_ds.get_targets("band_gap")

alignn_ef_val = np.array(ef_trainer.predict(
    val_structs_official, [e.material_id for e in val_ds.entries]
))
alignn_bg_val = np.array(bg_trainer.predict(
    val_structs_official, [e.material_id for e in val_ds.entries]
))

print("  Loading MACE artifacts...")
mace = MACEPredictor(model_name="medium")
mace.load_model()
mace.load_artifacts(MODELS / "full_retrain" / "mace_artifacts")

mace_ef_val = mace.predict_ef(val_structs_official)
mace_bg_val = mace.predict_bg(val_structs_official)
mace_ef_val = np.where(np.isnan(mace_ef_val), alignn_ef_val, mace_ef_val)
mace_bg_val = np.where(np.isnan(mace_bg_val), alignn_bg_val, mace_bg_val)

ensemble = WeightedEnsemble()
ensemble.fit(
    alignn_ef=alignn_ef_val, mace_ef=mace_ef_val,
    true_ef=np.array(val_ef_official),
    alignn_bg=alignn_bg_val, mace_bg=mace_bg_val,
    true_bg=np.array(val_bg_official),
    n_trials=300,
)
ef_ens_val, bg_ens_val = ensemble.predict(
    alignn_ef_val, mace_ef_val, alignn_bg_val, mace_bg_val,
)
calibrator = CalibrationLayer()
calibrator.fit(ef_ens_val, np.array(val_ef_official),
               bg_ens_val, np.array(val_bg_official))
ef_cal, bg_cal = calibrator.calibrate(ef_ens_val, bg_ens_val)

mae_ef = float(np.mean(np.abs(ef_cal - val_ef_official)))
mae_bg = float(np.mean(np.abs(bg_cal - val_bg_official)))
score  = compute_full_score(mae_ef, mae_bg)

# Class breakdown
bg_true   = np.array(val_bg_official)
mask_metal = bg_true == 0
mask_semi  = (bg_true > 0) & (bg_true < 3)
mask_wide  = bg_true >= 3

# Layered proxy: structures with any of the top layered elements
# (rough heuristic; real layered flag not in Bridge Dataset labels)
def _class_mae(pred, true, mask):
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - true[mask])))

print(f"\n{'='*60}")
print(f"COMBINED MODEL (v4) — OFFICIAL VAL RESULTS")
print(f"{'='*60}")
print(f"  Overall EF MAE : {mae_ef:.4f} eV/atom")
print(f"  Overall BG MAE : {mae_bg:.4f} eV")
print(f"  EF score       : {score['score_ef']:.2f} / 20")
print(f"  BG score       : {score['score_bg']:.2f} / 20")
print(f"  TOTAL          : {score['total_performance_score']:.2f} / 40")
print(f"\n  Breakdown by material class:")
print(f"    Metals   EF MAE : {_class_mae(ef_cal, bg_true*0+np.array(val_ef_official), mask_metal):.4f}"
      f"   BG MAE: {_class_mae(bg_cal, bg_true, mask_metal):.4f}  (n={mask_metal.sum()})")
print(f"    Semis    EF MAE : {_class_mae(ef_cal, np.array(val_ef_official), mask_semi):.4f}"
      f"   BG MAE: {_class_mae(bg_cal, bg_true, mask_semi):.4f}  (n={mask_semi.sum()})")
print(f"    Wide-gap EF MAE : {_class_mae(ef_cal, np.array(val_ef_official), mask_wide):.4f}"
      f"   BG MAE: {_class_mae(bg_cal, bg_true, mask_wide):.4f}  (n={mask_wide.sum()})")
print(f"\n  vs v3 (aug semi only): EF 0.0461 / BG 0.0158 / 38.32 (val, optimistic)")
print(f"{'='*60}")

# Save artifacts
ensemble.save(OUT_DIR / "ensemble_weights.json")
calibrator.save(OUT_DIR / "calibration")

# ── 9. Generate test predictions ─────────────────────────────────────────────
print("\n[8] Generating test predictions...")
test_cifs = sorted((RAW / "test_structures").glob("*.cif"))
test_ids, test_structs = [], []
for cif in tqdm(test_cifs, desc="Loading test CIFs"):
    try:
        test_structs.append(Structure.from_file(str(cif)))
        test_ids.append(cif.stem)
    except Exception:
        continue

alignn_ef_test = np.array(ef_trainer.predict(test_structs, test_ids))
alignn_bg_test = np.array(bg_trainer.predict(test_structs, test_ids))
mace_ef_test   = mace.predict_ef(test_structs)
mace_bg_test   = mace.predict_bg(test_structs)
mace_ef_test   = np.where(np.isnan(mace_ef_test), alignn_ef_test, mace_ef_test)
mace_bg_test   = np.where(np.isnan(mace_bg_test), alignn_bg_test, mace_bg_test)

ef_ens_test, bg_ens_test = ensemble.predict(
    alignn_ef_test, mace_ef_test, alignn_bg_test, mace_bg_test,
)
ef_final, bg_final = calibrator.calibrate(ef_ens_test, bg_ens_test)
bg_final = np.clip(bg_final, 0.0, None)

pred_path = ROOT / "submissions" / "CataLIST" / "predictions_test.json"
pred_path.parent.mkdir(parents=True, exist_ok=True)
with open(pred_path, "w") as f:
    json.dump({
        "team_name":          "CataLIST",
        "model_id":           "ALIGNN_MACE_ensemble_v4_combined",
        "matfed_api_version": "1.0",
        "predictions": [
            {
                "material_id":               test_ids[i],
                "formation_energy_per_atom": float(ef_final[i]),
                "band_gap":                  float(bg_final[i]),
            }
            for i in range(len(test_ids))
        ],
    }, f, indent=2)

print(f"  Saved {len(test_ids)} predictions → {pred_path}")
print(f"  model_id: ALIGNN_MACE_ensemble_v4_combined")
print(f"\nDone. Submit as PR 3 if val score > 38.32/40")
