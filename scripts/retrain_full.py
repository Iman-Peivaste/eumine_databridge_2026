"""
Full-data retraining pipeline (Step 4B).

Run:
    python scripts/retrain_full.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("WANDB_MODE", "disabled")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from eumine_databridge.data.combined_loader import CombinedDataset
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
FULL_MODELS = MODELS / "full_retrain"
FULL_MODELS.mkdir(parents=True, exist_ok=True)

INTERNAL_VAL_FRAC = 0.1


def _alignn_dataset(structures, targets, ids, cfg):
    return structures_to_alignn_dataset(
        structures=structures,
        targets=targets,
        material_ids=ids,
        cutoff=cfg.cutoff,
        max_neighbors=cfg.max_neighbors,
    )


print("\n[1] Loading datasets...")
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

combined = CombinedDataset(train_ds, val_ds)
all_structures = combined.get_structures()
all_ef = combined.get_target_array("formation_energy_per_atom")
all_bg = combined.get_target_array("band_gap")
all_ef_ids = combined.get_material_ids()
all_bg_ids = all_ef_ids

test_structures = test_ds.get_structures()
test_ids = [e.material_id for e in test_ds.entries]

print(f"Combined: {len(all_structures)} structures")
print(f"Test    : {len(test_structures)} structures")

# ── 2. Five-fold CV ───────────────────────────────────────────────────────────
print("\n[2] Running 5-fold cross-validation for OOF predictions...")
folds = combined.get_cv_folds(n_folds=5, seed=42)

oof_ef = np.zeros(len(all_structures))
oof_bg = np.zeros(len(all_structures))
oof_ef_mace = np.zeros(len(all_structures))
oof_bg_mace = np.zeros(len(all_structures))
fold_ef_maes: list[float] = []
fold_bg_maes: list[float] = []
fold_mace_ef_maes: list[float] = []
fold_mace_bg_maes: list[float] = []

for fold_idx, (tr_idx, vl_idx) in enumerate(folds):
    print(f"\n{'─' * 50}")
    print(f"FOLD {fold_idx + 1} / 5")
    print(f"{'─' * 50}")

    fold_dir = FULL_MODELS / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tr_structures = [all_structures[i] for i in tr_idx]
    vl_structures = [all_structures[i] for i in vl_idx]
    tr_ef = all_ef[tr_idx].tolist()
    vl_ef = all_ef[vl_idx].tolist()
    tr_bg = all_bg[tr_idx].tolist()
    vl_bg = all_bg[vl_idx].tolist()
    tr_ef_ids = [all_ef_ids[i] for i in tr_idx]
    vl_ef_ids = [all_ef_ids[i] for i in vl_idx]
    tr_bg_ids = [all_bg_ids[i] for i in tr_idx]
    vl_bg_ids = [all_bg_ids[i] for i in vl_idx]

    print(f"\n  [Fold {fold_idx + 1}] ALIGNN EF...")
    ef_cfg = get_ef_config()
    ef_cfg.output_dir = fold_dir / "alignn_ef"
    ef_cfg.wandb_run_name = f"alignn_ef_fold{fold_idx + 1}"
    ef_cfg.epochs = 300
    ef_cfg.patience = 50
    ef_trainer = ALIGNNFineTuner(ef_cfg)
    ef_trainer.setup(
        train_dataset=_alignn_dataset(tr_structures, tr_ef, tr_ef_ids, ef_cfg),
        val_dataset=_alignn_dataset(vl_structures, vl_ef, vl_ef_ids, ef_cfg),
    )
    ef_trainer.train()
    ef_trainer.load_best_model()
    _, fold_ef_preds, _ = ef_trainer._validate()
    fold_ef_preds = np.array(fold_ef_preds)
    oof_ef[vl_idx] = fold_ef_preds
    fold_ef_mae = float(np.mean(np.abs(fold_ef_preds - vl_ef)))
    fold_ef_maes.append(fold_ef_mae)
    print(f"  Fold {fold_idx + 1} EF MAE: {fold_ef_mae:.4f}")

    print(f"\n  [Fold {fold_idx + 1}] ALIGNN BG...")
    bg_cfg = get_bg_config()
    bg_cfg.output_dir = fold_dir / "alignn_bg"
    bg_cfg.wandb_run_name = f"alignn_bg_fold{fold_idx + 1}"
    bg_cfg.epochs = 400
    bg_cfg.patience = 60
    bg_trainer = ALIGNNFineTuner(bg_cfg)
    bg_trainer.setup(
        train_dataset=_alignn_dataset(tr_structures, tr_bg, tr_bg_ids, bg_cfg),
        val_dataset=_alignn_dataset(vl_structures, vl_bg, vl_bg_ids, bg_cfg),
    )
    bg_trainer.train()
    bg_trainer.load_best_model()
    _, fold_bg_preds, _ = bg_trainer._validate()
    fold_bg_preds = np.array(fold_bg_preds)
    oof_bg[vl_idx] = fold_bg_preds
    fold_bg_mae = float(np.mean(np.abs(fold_bg_preds - vl_bg)))
    fold_bg_maes.append(fold_bg_mae)
    print(f"  Fold {fold_idx + 1} BG MAE: {fold_bg_mae:.4f}")

    print(f"\n  [Fold {fold_idx + 1}] MACE inference...")
    mace_fold = MACEPredictor(model_name="medium")
    mace_fold.load_model()
    mace_fold.fit_references(tr_structures, tr_ef)
    mace_fold.fit_bg_head(tr_structures, tr_bg)
    fold_mace_ef = mace_fold.predict_ef(vl_structures)
    fold_mace_bg = mace_fold.predict_bg(vl_structures)
    fold_mace_ef = np.where(np.isnan(fold_mace_ef), fold_ef_preds, fold_mace_ef)
    fold_mace_bg = np.where(np.isnan(fold_mace_bg), fold_bg_preds, fold_mace_bg)
    oof_ef_mace[vl_idx] = fold_mace_ef
    oof_bg_mace[vl_idx] = fold_mace_bg
    fold_mace_ef_mae = float(np.mean(np.abs(fold_mace_ef - vl_ef)))
    fold_mace_bg_mae = float(np.mean(np.abs(fold_mace_bg - vl_bg)))
    fold_mace_ef_maes.append(fold_mace_ef_mae)
    fold_mace_bg_maes.append(fold_mace_bg_mae)
    print(f"  Fold {fold_idx + 1} MACE EF MAE: {fold_mace_ef_mae:.4f}")
    print(f"  Fold {fold_idx + 1} MACE BG MAE: {fold_mace_bg_mae:.4f}")

    del ef_trainer, bg_trainer, mace_fold
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

oof_ef_mae = float(np.mean(np.abs(oof_ef - all_ef)))
oof_bg_mae = float(np.mean(np.abs(oof_bg - all_bg)))
oof_score = compute_full_score(oof_ef_mae, oof_bg_mae)

print(f"\n{'=' * 60}")
print("5-FOLD OOF SUMMARY (ALIGNN only, pre-ensemble)")
print(f"{'=' * 60}")
print("Per-fold ALIGNN MAE:")
for i, (ef_m, bg_m) in enumerate(zip(fold_ef_maes, fold_bg_maes), 1):
    print(f"  Fold {i}: EF={ef_m:.4f}  BG={bg_m:.4f}")
print(f"\n  OOF EF MAE : {oof_ef_mae:.4f} eV/atom")
print(f"  OOF BG MAE : {oof_bg_mae:.4f} eV")
print(f"  OOF Score  : {oof_score['total_performance_score']:.2f} / 40")
print(f"{'=' * 60}")

np.save(FULL_MODELS / "oof_ef_alignn.npy", oof_ef)
np.save(FULL_MODELS / "oof_bg_alignn.npy", oof_bg)
np.save(FULL_MODELS / "oof_ef_mace.npy", oof_ef_mace)
np.save(FULL_MODELS / "oof_bg_mace.npy", oof_bg_mace)
np.save(FULL_MODELS / "all_ef_targets.npy", all_ef)
np.save(FULL_MODELS / "all_bg_targets.npy", all_bg)

# ── 3. Ensemble on OOF ────────────────────────────────────────────────────────
print("\n[3] Optimizing ensemble weights on OOF predictions...")
ensemble = WeightedEnsemble()
ensemble.fit(
    alignn_ef=oof_ef,
    mace_ef=oof_ef_mace,
    true_ef=all_ef,
    alignn_bg=oof_bg,
    mace_bg=oof_bg_mace,
    true_bg=all_bg,
    n_trials=300,
)
ensemble.save(FULL_MODELS / "ensemble_weights.json")
oof_ef_ensemble, oof_bg_ensemble = ensemble.predict(
    oof_ef, oof_ef_mace, oof_bg, oof_bg_mace
)

# ── 4. Calibration on OOF ensemble ──────────────────────────────────────────
print("\n[4] Fitting calibration on OOF ensemble predictions...")
calibrator = CalibrationLayer()
calibrator.fit(
    ef_predictions=oof_ef_ensemble,
    ef_targets=all_ef,
    bg_predictions=oof_bg_ensemble,
    bg_targets=all_bg,
)
calibrator.save(FULL_MODELS / "calibration")
oof_ef_cal, oof_bg_cal = calibrator.calibrate(oof_ef_ensemble, oof_bg_ensemble)
oof_ef_cal_mae = float(np.mean(np.abs(oof_ef_cal - all_ef)))
oof_bg_cal_mae = float(np.mean(np.abs(oof_bg_cal - all_bg)))
oof_final_score = compute_full_score(oof_ef_cal_mae, oof_bg_cal_mae)

print(f"\n{'=' * 60}")
print("OOF SCORE AFTER ENSEMBLE + CALIBRATION")
print(f"{'=' * 60}")
print(f"  OOF EF MAE  : {oof_ef_cal_mae:.4f} eV/atom")
print(f"  OOF BG MAE  : {oof_bg_cal_mae:.4f} eV")
print(f"  EF score    : {oof_final_score['score_ef']:.2f} / 20")
print(f"  BG score    : {oof_final_score['score_bg']:.2f} / 20")
print(f"  TOTAL (OOF) : {oof_final_score['total_performance_score']:.2f} / 40")
print(f"{'=' * 60}")

# ── 5. Full ALIGNN retrain ────────────────────────────────────────────────────
print("\n[5] Retraining ALIGNN on full combined dataset (850 samples)...")
rng = np.random.default_rng(42)
all_idx = np.arange(len(all_structures))
rng.shuffle(all_idx)
n_val_internal = int(len(all_structures) * INTERNAL_VAL_FRAC)
internal_val_idx = all_idx[:n_val_internal]
internal_tr_idx = all_idx[n_val_internal:]

int_tr_structures = [all_structures[i] for i in internal_tr_idx]
int_vl_structures = [all_structures[i] for i in internal_val_idx]
int_tr_ef = all_ef[internal_tr_idx].tolist()
int_vl_ef = all_ef[internal_val_idx].tolist()
int_tr_bg = all_bg[internal_tr_idx].tolist()
int_vl_bg = all_bg[internal_val_idx].tolist()
int_tr_ef_ids = [all_ef_ids[i] for i in internal_tr_idx]
int_vl_ef_ids = [all_ef_ids[i] for i in internal_val_idx]
int_tr_bg_ids = [all_bg_ids[i] for i in internal_tr_idx]
int_vl_bg_ids = [all_bg_ids[i] for i in internal_val_idx]

print(
    f"\n  Full ALIGNN EF ({len(internal_tr_idx)} train, "
    f"{len(internal_val_idx)} internal val)..."
)
ef_full_cfg = get_ef_config()
ef_full_cfg.output_dir = FULL_MODELS / "alignn_ef_full"
ef_full_cfg.wandb_run_name = "alignn_ef_full_retrain"
ef_full_cfg.epochs = 350
ef_full_cfg.patience = 60
ef_full_trainer = ALIGNNFineTuner(ef_full_cfg)
ef_full_trainer.setup(
    train_dataset=_alignn_dataset(
        int_tr_structures, int_tr_ef, int_tr_ef_ids, ef_full_cfg
    ),
    val_dataset=_alignn_dataset(
        int_vl_structures, int_vl_ef, int_vl_ef_ids, ef_full_cfg
    ),
)
ef_full_trainer.train()

print(
    f"\n  Full ALIGNN BG ({len(internal_tr_idx)} train, "
    f"{len(internal_val_idx)} internal val)..."
)
bg_full_cfg = get_bg_config()
bg_full_cfg.output_dir = FULL_MODELS / "alignn_bg_full"
bg_full_cfg.wandb_run_name = "alignn_bg_full_retrain"
bg_full_cfg.epochs = 450
bg_full_cfg.patience = 70
bg_full_trainer = ALIGNNFineTuner(bg_full_cfg)
bg_full_trainer.setup(
    train_dataset=_alignn_dataset(
        int_tr_structures, int_tr_bg, int_tr_bg_ids, bg_full_cfg
    ),
    val_dataset=_alignn_dataset(
        int_vl_structures, int_vl_bg, int_vl_bg_ids, bg_full_cfg
    ),
)
bg_full_trainer.train()

# ── 6. MACE full refit ────────────────────────────────────────────────────────
print("\n[6] Refitting MACE on full dataset...")
mace_full = MACEPredictor(model_name="medium")
mace_full.load_model()
mace_full.fit_references(all_structures, all_ef.tolist())
mace_full.fit_bg_head(all_structures, all_bg.tolist())
mace_full.save(FULL_MODELS / "mace_artifacts")

# ── 7. Test predictions ───────────────────────────────────────────────────────
print("\n[7] Generating final test predictions...")
ef_full_trainer.load_best_model()
bg_full_trainer.load_best_model()
alignn_ef_test = ef_full_trainer.predict(test_structures, test_ids)
alignn_bg_test = bg_full_trainer.predict(test_structures, test_ids)
mace_ef_test = mace_full.predict_ef(test_structures)
mace_bg_test = mace_full.predict_bg(test_structures)
mace_ef_test = np.where(np.isnan(mace_ef_test), alignn_ef_test, mace_ef_test)
mace_bg_test = np.where(np.isnan(mace_bg_test), alignn_bg_test, mace_bg_test)
ef_ensemble_test, bg_ensemble_test = ensemble.predict(
    alignn_ef_test, mace_ef_test, alignn_bg_test, mace_bg_test
)
ef_final_test, bg_final_test = calibrator.calibrate(
    ef_ensemble_test, bg_ensemble_test
)
bg_final_test = np.clip(bg_final_test, 0.0, None)

# ── 8. Save submission ──────────────────────────────────────────────────────
print("\n[8] Saving test predictions...")
predictions = [
    {
        "material_id": test_ids[i],
        "formation_energy_per_atom": float(ef_final_test[i]),
        "band_gap": float(bg_final_test[i]),
    }
    for i in range(len(test_ids))
]
submission = {
    "team_name": "LIST_EuMINe",
    "model_id": "ALIGNN_MACE_ensemble_v2_fullretrain",
    "matfed_api_version": "1.0",
    "predictions": predictions,
    "oof_mae_ef": oof_ef_cal_mae,
    "oof_mae_bg": oof_bg_cal_mae,
    "oof_score": oof_final_score["total_performance_score"],
}
pred_path = ROOT / "submissions" / "LIST_EuMINe" / "predictions_test.json"
pred_path.parent.mkdir(parents=True, exist_ok=True)
with open(pred_path, "w") as f:
    json.dump(submission, f, indent=2)
print(f"  Saved: {pred_path}")
print(f"  {len(predictions)} test predictions")

# ── 9. Final report ───────────────────────────────────────────────────────────
print(f"\n{'#' * 60}")
print("# STEP 4B — FINAL REPORT")
print(f"{'#' * 60}")
print(f"\nPer-fold ALIGNN MAE (EF / BG):")
for i, (ef_m, bg_m) in enumerate(zip(fold_ef_maes, fold_bg_maes), 1):
    print(f"  Fold {i}: {ef_m:.4f} / {bg_m:.4f}")
print(f"\nOOF EF MAE  (honest): {oof_ef_cal_mae:.4f} eV/atom")
print(f"OOF BG MAE  (honest): {oof_bg_cal_mae:.4f} eV")
print(f"\nHackathon score estimate:")
print(f"  EF : {oof_final_score['score_ef']:.2f} / 20")
print(f"  BG : {oof_final_score['score_bg']:.2f} / 20")
print(f"  TOTAL (OOF): {oof_final_score['total_performance_score']:.2f} / 40")
print(
    f"\nvs Step 4 val score (35.96): "
    f"{oof_final_score['total_performance_score'] - 35.96:+.2f}"
)
print(f"\nOptimal ensemble weights:")
print(
    f"  EF: ALIGNN={ensemble.weights_ef['alignn']:.3f} "
    f"MACE={ensemble.weights_ef['mace']:.3f}"
)
print(
    f"  BG: ALIGNN={ensemble.weights_bg['alignn']:.3f} "
    f"MACE={ensemble.weights_bg['mace']:.3f}"
)
print(f"\nArtifacts:")
print(f"  {FULL_MODELS}/alignn_ef_full/best_model.pt")
print(f"  {FULL_MODELS}/alignn_bg_full/best_model.pt")
print(f"  {FULL_MODELS}/mace_artifacts/")
print(f"  {FULL_MODELS}/ensemble_weights.json")
print(f"  {FULL_MODELS}/calibration/")
print(f"  {pred_path}  (v2)")
print(f"{'#' * 60}\n")

if os.getenv("WANDB_MODE", "disabled") != "disabled":
    try:
        import wandb

        run = wandb.init(
            project=os.getenv(
                "WANDB_PROJECT", "eumine_databridge_2026"
            ),
            name="full_retrain_final",
            job_type="evaluation",
        )
        wandb.log({
            "oof_ef_mae": oof_ef_cal_mae,
            "oof_bg_mae": oof_bg_cal_mae,
            "oof_score_ef": oof_final_score["score_ef"],
            "oof_score_bg": oof_final_score["score_bg"],
            "oof_total_score": oof_final_score["total_performance_score"],
        })
        wandb.finish()
    except Exception as exc:
        print(f"W&B logging skipped: {exc}")
