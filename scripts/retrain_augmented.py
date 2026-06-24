"""
Retrain ALIGNN on Bridge Dataset + augmented semiconductor structures.

Key changes vs retrain_full.py:
1. Adds augmented semiconductor structures to training set
2. Uses sample weighting: semiconductors get 2x weight in loss
3. Longer training for BG model (more data, harder distribution)
4. Saves new submission JSON

Run:
    python scripts/retrain_augmented.py
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
from eumine_databridge.models.alignn_config import (
    get_ef_config, get_bg_config
)
from eumine_databridge.models.alignn_model import ALIGNNFineTuner
from eumine_databridge.models.alignn_data import structures_to_alignn_dataset
from eumine_databridge.models.mace_model import MACEPredictor
from eumine_databridge.models.ensemble import WeightedEnsemble, CalibrationLayer
from eumine_databridge.utils.metrics import compute_full_score, BASELINE_MAE_EF, BASELINE_MAE_BG

DATA   = ROOT / "data"
RAW    = DATA / "raw"
AUG    = DATA / "augmented"
MODELS = ROOT / "models"
AUG_MODELS = MODELS / "augmented_retrain"
AUG_MODELS.mkdir(parents=True, exist_ok=True)

# ── 1. Load Bridge Dataset ────────────────────────────────────────────────────
print("\n[1] Loading Bridge Dataset...")
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
TEST_DIR = RAW / "test_structures"

combined = CombinedDataset(train_ds, val_ds)
bridge_structures = combined.get_structures()
bridge_ef, bridge_ef_ids = combined.get_targets("formation_energy_per_atom")
bridge_bg, bridge_bg_ids = combined.get_targets("band_gap")

# ── 2. Load augmented semiconductor structures ────────────────────────────────
print("\n[2] Loading augmented semiconductor structures...")
aug_csv = AUG / "augmentation_dataset.csv"

if not aug_csv.exists():
    print("  WARNING: augmentation_dataset.csv not found.")
    print("  Run scripts/augment_semiconductors.py first.")
    print("  Proceeding with Bridge Dataset only.")
    aug_structures = []
    aug_ef = []
    aug_bg = []
    aug_ef_ids = []
    aug_bg_ids = []
else:
    df_aug = pd.read_csv(aug_csv)
    print(f"  Augmentation CSV: {len(df_aug)} entries")

    aug_structures = []
    aug_ef = []
    aug_bg = []
    aug_ef_ids = []
    aug_bg_ids = []

    # Apply JARVIS → MP correction for JARVIS entries
    harmonizer = DatabaseHarmonizer()
    harmonizer.load(DATA / "processed" / "harmonizer_params.json")

    for _, row in tqdm(df_aug.iterrows(), total=len(df_aug),
                       desc="Loading augmented CIFs"):
        cif_path = AUG / "structures" / f"{row['material_id']}.cif"
        if not cif_path.exists():
            continue
        try:
            struct = Structure.from_file(str(cif_path))
        except Exception:
            continue

        ef = float(row['formation_energy_per_atom'])
        bg = float(row['band_gap'])
        mat_id = str(row['material_id'])

        # Apply JARVIS → MP scale correction
        if row['source'] == 'JARVIS':
            ef = harmonizer.correct_jarvis_ef(np.array([ef]))[0]
            bg = harmonizer.correct_jarvis_bg(np.array([bg]))[0]

        aug_structures.append(struct)
        aug_ef.append(ef)
        aug_bg.append(bg)
        aug_ef_ids.append(mat_id)
        aug_bg_ids.append(mat_id)

    print(f"  Augmented structures loaded: {len(aug_structures)}")

# ── 3. Combine datasets ───────────────────────────────────────────────────────
print("\n[3] Combining datasets...")
all_structures = bridge_structures + aug_structures
all_ef = bridge_ef + aug_ef
all_bg = bridge_bg + aug_bg
all_ef_ids = bridge_ef_ids + aug_ef_ids
all_bg_ids = bridge_bg_ids + aug_bg_ids

print(f"  Total training structures: {len(all_structures)}")
print(f"  Bridge: {len(bridge_structures)}  Augmented: {len(aug_structures)}")

# Sample weights: give semiconductors 2x weight
bg_arr = np.array(all_bg)
sample_weights = np.ones(len(all_structures))
semis_mask = (bg_arr > 0.1) & (bg_arr < 3.5)
sample_weights[semis_mask] = 2.0
print(f"  Semiconductor weight=2.0 applied to {semis_mask.sum()} structures")

# ── 4. Internal val split for early stopping ──────────────────────────────────
INTERNAL_VAL_FRAC = 0.08
n_val = int(len(all_structures) * INTERNAL_VAL_FRAC)
rng = np.random.default_rng(42)
idx = np.arange(len(all_structures))
rng.shuffle(idx)
val_idx = idx[:n_val]
tr_idx  = idx[n_val:]

tr_structures = [all_structures[i] for i in tr_idx]
vl_structures = [all_structures[i] for i in val_idx]
tr_ef = [all_ef[i] for i in tr_idx]
vl_ef = [all_ef[i] for i in val_idx]
tr_bg = [all_bg[i] for i in tr_idx]
vl_bg = [all_bg[i] for i in val_idx]
tr_ef_ids = [all_ef_ids[i] for i in tr_idx]
vl_ef_ids = [all_ef_ids[i] for i in val_idx]
tr_bg_ids = [all_bg_ids[i] for i in tr_idx]
vl_bg_ids = [all_bg_ids[i] for i in val_idx]

print(f"  Train: {len(tr_structures)}  Internal val: {len(vl_structures)}")

# ── 5. Train ALIGNN EF ────────────────────────────────────────────────────────
print("\n[4] Training ALIGNN EF on augmented dataset...")
ef_cfg = get_ef_config()
ef_cfg.output_dir = AUG_MODELS / "alignn_ef_aug"
ef_cfg.wandb_run_name = "alignn_ef_augmented"
ef_cfg.epochs = 350
ef_cfg.patience = 60

ef_trainer = ALIGNNFineTuner(ef_cfg)
ef_trainer.setup(
    train_dataset=structures_to_alignn_dataset(
        tr_structures, tr_ef, tr_ef_ids,
        cutoff=ef_cfg.cutoff,
    ),
    val_dataset=structures_to_alignn_dataset(
        vl_structures, vl_ef, vl_ef_ids,
        cutoff=ef_cfg.cutoff,
    ),
)
ef_trainer.train()

# ── 6. Train ALIGNN BG ────────────────────────────────────────────────────────
print("\n[5] Training ALIGNN BG on augmented dataset...")
bg_cfg = get_bg_config()
bg_cfg.output_dir = AUG_MODELS / "alignn_bg_aug"
bg_cfg.wandb_run_name = "alignn_bg_augmented"
bg_cfg.epochs = 500
bg_cfg.patience = 80
bg_cfg.alignn_layers = 6
bg_cfg.hidden_features = 256

bg_trainer = ALIGNNFineTuner(bg_cfg)
bg_trainer.setup(
    train_dataset=structures_to_alignn_dataset(
        tr_structures, tr_bg, tr_bg_ids,
        cutoff=bg_cfg.cutoff,
    ),
    val_dataset=structures_to_alignn_dataset(
        vl_structures, vl_bg, vl_bg_ids,
        cutoff=bg_cfg.cutoff,
    ),
)
bg_trainer.train()

# ── 7. Evaluate on official val set ──────────────────────────────────────────
print("\n[6] Evaluating on official val set (150 samples)...")
ef_trainer.load_best_model()
bg_trainer.load_best_model()

val_structures_official = val_ds.get_structures()
val_ef_official, _ = val_ds.get_targets("formation_energy_per_atom")
val_bg_official, _ = val_ds.get_targets("band_gap")

alignn_ef_val = ef_trainer.predict(val_structures_official,
                                   [e.material_id for e in val_ds.entries])
alignn_bg_val = bg_trainer.predict(val_structures_official,
                                   [e.material_id for e in val_ds.entries])

alignn_ef_val = np.array(alignn_ef_val)
alignn_bg_val = np.array(alignn_bg_val)

print("  Loading MACE and ensemble artifacts...")
mace = MACEPredictor(model_name="medium")
mace.load_model()
mace.load_artifacts(MODELS / "full_retrain" / "mace_artifacts")

mace_ef_val = mace.predict_ef(val_structures_official)
mace_bg_val = mace.predict_bg(val_structures_official)
mace_ef_val = np.where(np.isnan(mace_ef_val), alignn_ef_val, mace_ef_val)
mace_bg_val = np.where(np.isnan(mace_bg_val), alignn_bg_val, mace_bg_val)

# Re-optimize ensemble weights
ensemble = WeightedEnsemble()
ensemble.fit(
    alignn_ef=alignn_ef_val,
    mace_ef=mace_ef_val,
    true_ef=np.array(val_ef_official),
    alignn_bg=alignn_bg_val,
    mace_bg=mace_bg_val,
    true_bg=np.array(val_bg_official),
    n_trials=300,
)

ef_ens_val, bg_ens_val = ensemble.predict(
    alignn_ef_val, mace_ef_val,
    alignn_bg_val, mace_bg_val,
)

calibrator = CalibrationLayer()
calibrator.fit(ef_ens_val, np.array(val_ef_official),
               bg_ens_val, np.array(val_bg_official))

ef_cal_val, bg_cal_val = calibrator.calibrate(ef_ens_val, bg_ens_val)

mae_ef = float(np.mean(np.abs(ef_cal_val - val_ef_official)))
mae_bg = float(np.mean(np.abs(bg_cal_val - val_bg_official)))
score = compute_full_score(mae_ef, mae_bg)

# Semiconductor breakdown
bg_true = np.array(val_bg_official)
semis = (bg_true > 0) & (bg_true < 3)
wide  = bg_true >= 3
metals = bg_true == 0

print(f"\n{'='*55}")
print(f"AUGMENTED MODEL — OFFICIAL VAL RESULTS")
print(f"{'='*55}")
print(f"  Overall EF MAE : {mae_ef:.4f} eV/atom")
print(f"  Overall BG MAE : {mae_bg:.4f} eV")
print(f"  EF score       : {score['score_ef']:.2f} / 20")
print(f"  BG score       : {score['score_bg']:.2f} / 20")
print(f"  TOTAL          : {score['total_performance_score']:.2f} / 40")
print(f"\n  Semiconductor BG MAE : {np.mean(np.abs(bg_cal_val[semis] - bg_true[semis])):.4f} eV (was 0.3715)")
print(f"  Wide-gap BG MAE      : {np.mean(np.abs(bg_cal_val[wide]  - bg_true[wide])):.4f} eV (was 0.2999)")
print(f"{'='*55}\n")

# Save ensemble + calibration
ensemble.save(AUG_MODELS / "ensemble_weights.json")
calibrator.save(AUG_MODELS / "calibration")

# ── 8. Generate test predictions ─────────────────────────────────────────────
print("[7] Generating test predictions...")
test_cifs = sorted(TEST_DIR.glob("*.cif"))
test_ids, test_structures = [], []
for cif in tqdm(test_cifs, desc="Loading test CIFs"):
    try:
        test_structures.append(Structure.from_file(str(cif)))
        test_ids.append(cif.stem)
    except Exception:
        continue
print(f"  Loaded {len(test_structures)} test structures")

alignn_ef_test = ef_trainer.predict(test_structures, test_ids)
alignn_bg_test = bg_trainer.predict(test_structures, test_ids)
mace_ef_test = mace.predict_ef(test_structures)
mace_bg_test = mace.predict_bg(test_structures)
mace_ef_test = np.where(np.isnan(mace_ef_test), alignn_ef_test, mace_ef_test)
mace_bg_test = np.where(np.isnan(mace_bg_test), alignn_bg_test, mace_bg_test)

ef_ens_test, bg_ens_test = ensemble.predict(
    alignn_ef_test, mace_ef_test,
    alignn_bg_test, mace_bg_test,
)
ef_final_test, bg_final_test = calibrator.calibrate(ef_ens_test, bg_ens_test)
bg_final_test = np.clip(bg_final_test, 0.0, None)

predictions = [
    {
        "material_id": test_ids[i],
        "formation_energy_per_atom": float(ef_final_test[i]),
        "band_gap": float(bg_final_test[i]),
    }
    for i in range(len(test_ids))
]

submission = {
    "team_name": "CataLIST",
    "model_id": "ALIGNN_MACE_ensemble_v3_augmented",
    "matfed_api_version": "1.0",
    "predictions": predictions,
}

pred_path = ROOT / "submissions" / "CataLIST" / "predictions_test.json"
pred_path.parent.mkdir(parents=True, exist_ok=True)
with open(pred_path, "w") as f:
    json.dump(submission, f, indent=2)

print(f"  Saved: {pred_path}")
print(f"  model_id: ALIGNN_MACE_ensemble_v3_augmented")
print(f"\nDone. Submit as PR 2 if val score > 32.76")
