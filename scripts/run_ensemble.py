"""
Build and evaluate the full ALIGNN + MACE ensemble.

Run:
    python scripts/run_ensemble.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("WANDB_MODE", "disabled")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from eumine_databridge.data.loader import BridgeDataset
from eumine_databridge.models.alignn_config import get_bg_config, get_ef_config
from eumine_databridge.models.alignn_data import structures_to_alignn_dataset
from eumine_databridge.models.alignn_model import ALIGNNFineTuner
from eumine_databridge.models.ensemble import CalibrationLayer, WeightedEnsemble
from eumine_databridge.models.mace_model import MACEPredictor
from eumine_databridge.utils.metrics import (
    BASELINE_MAE_BG,
    BASELINE_MAE_EF,
    compute_full_score,
    compute_metrics,
)

DATA = ROOT / "data" / "raw"
MODELS = ROOT / "models"
ENSEMBLE_DIR = MODELS / "ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

print("\n[1] Loading Bridge Dataset splits...")
train_ds = BridgeDataset(
    csv_path=DATA / "bridge_dataset_train.csv",
    structures_dir=DATA / "train_structures",
    split="train",
)
val_ds = BridgeDataset(
    csv_path=DATA / "bridge_dataset_val.csv",
    structures_dir=DATA / "val_structures",
    split="val",
)
test_ds = BridgeDataset(
    csv_path=DATA / "bridge_dataset_test.csv",
    structures_dir=DATA / "test_structures",
    split="test",
)

train_structures = train_ds.get_structures()
val_structures = val_ds.get_structures()
test_structures = test_ds.get_structures()
test_ids = [e.material_id for e in test_ds.entries]

train_ef, train_ef_ids = train_ds.get_targets("formation_energy_per_atom")
train_bg, train_bg_ids = train_ds.get_targets("band_gap")
val_ef, val_ef_ids = val_ds.get_targets("formation_energy_per_atom")
val_bg, val_bg_ids = val_ds.get_targets("band_gap")

print(f"  Train: {len(train_structures)} structures")
print(f"  Val  : {len(val_structures)} structures")
print(f"  Test : {len(test_structures)} structures")

print("\n[2] Running ALIGNN inference on val set...")
ef_config = get_ef_config()
ef_config.output_dir = MODELS / "alignn_ef"
alignn_ef_trainer = ALIGNNFineTuner(ef_config)
alignn_ef_trainer.setup(
    train_dataset=structures_to_alignn_dataset(
        train_structures,
        train_ef,
        train_ef_ids,
        cutoff=ef_config.cutoff,
        max_neighbors=ef_config.max_neighbors,
    ),
    val_dataset=structures_to_alignn_dataset(
        val_structures,
        val_ef,
        val_ef_ids,
        cutoff=ef_config.cutoff,
        max_neighbors=ef_config.max_neighbors,
    ),
)
alignn_ef_trainer.load_best_model()
_, alignn_ef_val, _ = alignn_ef_trainer._validate()
alignn_ef_val = np.array(alignn_ef_val)

bg_config = get_bg_config()
bg_config.output_dir = MODELS / "alignn_bg"
alignn_bg_trainer = ALIGNNFineTuner(bg_config)
alignn_bg_trainer.setup(
    train_dataset=structures_to_alignn_dataset(
        train_structures,
        train_bg,
        train_bg_ids,
        cutoff=bg_config.cutoff,
        max_neighbors=bg_config.max_neighbors,
    ),
    val_dataset=structures_to_alignn_dataset(
        val_structures,
        val_bg,
        val_bg_ids,
        cutoff=bg_config.cutoff,
        max_neighbors=bg_config.max_neighbors,
    ),
)
alignn_bg_trainer.load_best_model()
_, alignn_bg_val, _ = alignn_bg_trainer._validate()
alignn_bg_val = np.array(alignn_bg_val)

val_ef_arr = np.array(val_ef)
val_bg_arr = np.array(val_bg)
print(f"  ALIGNN EF val MAE : {np.mean(np.abs(alignn_ef_val - val_ef_arr)):.4f}")
print(f"  ALIGNN BG val MAE : {np.mean(np.abs(alignn_bg_val - val_bg_arr)):.4f}")

print("\n  Running ALIGNN on test set...")
alignn_ef_test = alignn_ef_trainer.predict(test_structures, test_ids)
alignn_bg_test = alignn_bg_trainer.predict(test_structures, test_ids)

print("\n[3] Running MACE-MP-0 inference...")
mace = MACEPredictor(model_name="medium")
mace.load_model()
mace.fit_references(train_structures, train_ef)
mace.fit_bg_head(train_structures, train_bg)

print("\n  MACE val set inference...")
mace_ef_val = mace.predict_ef(val_structures)
mace_bg_val = mace.predict_bg(val_structures)

nan_ef = np.isnan(mace_ef_val)
nan_bg = np.isnan(mace_bg_val)
if nan_ef.sum() > 0:
    print(f"  WARNING: {nan_ef.sum()} NaN EF predictions — using ALIGNN fallback")
    mace_ef_val[nan_ef] = alignn_ef_val[nan_ef]
if nan_bg.sum() > 0:
    print(f"  WARNING: {nan_bg.sum()} NaN BG predictions — using ALIGNN fallback")
    mace_bg_val[nan_bg] = alignn_bg_val[nan_bg]

print(f"  MACE EF val MAE : {np.mean(np.abs(mace_ef_val - val_ef_arr)):.4f}")
print(f"  MACE BG val MAE : {np.mean(np.abs(mace_bg_val - val_bg_arr)):.4f}")

print("\n  MACE test set inference...")
mace_ef_test = mace.predict_ef(test_structures)
mace_bg_test = mace.predict_bg(test_structures)
mace_ef_test = np.where(np.isnan(mace_ef_test), alignn_ef_test, mace_ef_test)
mace_bg_test = np.where(np.isnan(mace_bg_test), alignn_bg_test, mace_bg_test)

mace.save(ENSEMBLE_DIR / "mace_artifacts")

print("\n[4] Optimizing ensemble weights with Optuna...")
ensemble = WeightedEnsemble()
ensemble.fit(
    alignn_ef=alignn_ef_val,
    mace_ef=mace_ef_val,
    true_ef=val_ef_arr,
    alignn_bg=alignn_bg_val,
    mace_bg=mace_bg_val,
    true_bg=val_bg_arr,
    n_trials=300,
)
ensemble.save(ENSEMBLE_DIR / "ensemble_weights.json")

ef_ensemble_val, bg_ensemble_val = ensemble.predict(
    alignn_ef_val, mace_ef_val, alignn_bg_val, mace_bg_val
)
ef_ensemble_test, bg_ensemble_test = ensemble.predict(
    alignn_ef_test, mace_ef_test, alignn_bg_test, mace_bg_test
)

print("\n[5] Fitting calibration layer...")
calibrator = CalibrationLayer()
calibrator.fit(
    ef_predictions=ef_ensemble_val,
    ef_targets=val_ef_arr,
    bg_predictions=bg_ensemble_val,
    bg_targets=val_bg_arr,
)
calibrator.save(ENSEMBLE_DIR / "calibration")

ef_final_val, bg_final_val = calibrator.calibrate(
    ef_ensemble_val, bg_ensemble_val
)
ef_final_test, bg_final_test = calibrator.calibrate(
    ef_ensemble_test, bg_ensemble_test
)

print("\n[6] Final evaluation on validation set...")
mae_ef_final = float(np.mean(np.abs(ef_final_val - val_ef_arr)))
mae_bg_final = float(np.mean(np.abs(bg_final_val - val_bg_arr)))
final_score = compute_full_score(mae_ef_final, mae_bg_final)
ef_metrics = compute_metrics(
    ef_final_val,
    val_ef_arr,
    property_name="formation_energy_per_atom",
    baseline_mae=BASELINE_MAE_EF,
)
bg_metrics = compute_metrics(
    bg_final_val,
    val_bg_arr,
    property_name="band_gap",
    baseline_mae=BASELINE_MAE_BG,
)

print(f"\n{'='*60}")
print("FINAL ENSEMBLE RESULTS")
print(f"{'='*60}")
print("\nFormation Energy:")
print(f"  ALIGNN alone  : {np.mean(np.abs(alignn_ef_val - val_ef_arr)):.4f} eV/atom")
print(f"  MACE alone    : {np.mean(np.abs(mace_ef_val - val_ef_arr)):.4f} eV/atom")
print(f"  Ensemble      : {np.mean(np.abs(ef_ensemble_val - val_ef_arr)):.4f} eV/atom")
print(f"  + Calibration : {mae_ef_final:.4f} eV/atom")
print(f"  Score         : {ef_metrics['hackathon_score']:.2f} / 20")
print("\nBand Gap:")
print(f"  ALIGNN alone  : {np.mean(np.abs(alignn_bg_val - val_bg_arr)):.4f} eV")
print(f"  MACE alone    : {np.mean(np.abs(mace_bg_val - val_bg_arr)):.4f} eV")
print(f"  Ensemble      : {np.mean(np.abs(bg_ensemble_val - val_bg_arr)):.4f} eV")
print(f"  + Calibration : {mae_bg_final:.4f} eV")
print(f"  Score         : {bg_metrics['hackathon_score']:.2f} / 20")
print(f"\n{'─'*60}")
print(f"  EF score      : {final_score['score_ef']:.2f} / 20")
print(f"  BG score      : {final_score['score_bg']:.2f} / 20")
print(f"  TOTAL         : {final_score['total_performance_score']:.2f} / 40")
print(f"  vs ALIGNN only: {final_score['total_performance_score'] - 34.12:+.2f}")
print(f"  Qualifies     : {final_score['qualifies_for_stage2']}")
print(f"{'='*60}\n")

print("\n[7] Saving test predictions...")
predictions = []
for i, mat_id in enumerate(test_ids):
    predictions.append({
        "material_id": mat_id,
        "formation_energy_per_atom": float(ef_final_test[i]),
        "band_gap": float(np.clip(bg_final_test[i], 0.0, None)),
    })

output = {
    "team_name": "CataLIST",
    "model_id": "ALIGNN_MACE_ensemble_v1",
    "matfed_api_version": "1.0",
    "predictions": predictions,
    "val_mae_ef": mae_ef_final,
    "val_mae_bg": mae_bg_final,
    "val_score": final_score["total_performance_score"],
}

pred_path = ROOT / "submissions" / "CataLIST" / "predictions_test.json"
pred_path.parent.mkdir(parents=True, exist_ok=True)
with open(pred_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"  Test predictions saved to {pred_path}")
print(f"  {len(predictions)} materials predicted")

np.save(ENSEMBLE_DIR / "alignn_ef_val.npy", alignn_ef_val)
np.save(ENSEMBLE_DIR / "alignn_bg_val.npy", alignn_bg_val)
np.save(ENSEMBLE_DIR / "mace_ef_val.npy", mace_ef_val)
np.save(ENSEMBLE_DIR / "mace_bg_val.npy", mace_bg_val)
np.save(ENSEMBLE_DIR / "ef_final_val.npy", ef_final_val)
np.save(ENSEMBLE_DIR / "bg_final_val.npy", bg_final_val)
np.save(ENSEMBLE_DIR / "val_ef_targets.npy", val_ef_arr)
np.save(ENSEMBLE_DIR / "val_bg_targets.npy", val_bg_arr)
print(f"\nAll ensemble artifacts saved to {ENSEMBLE_DIR}")
