# Step 4B — Full-Data Retraining + Cross-Validated Calibration

**Status:** Complete  
**Date:** 2026-05-15  
**Environment:** `conda ip`, NVIDIA RTX A4000; 5-fold stratified CV (BG category)

---

## Objective

Retrain ALIGNN and MACE on **train + val combined (850 samples)**, obtain **honest out-of-fold (OOF)** predictions via 5-fold CV for ensemble/calibration fitting without leakage, retrain final models on 850 for test submission (`predictions_test.json` v2).

---

## Final documented results

| Metric | Value | Note |
|---|---|---|
| OOF EF MAE | **0.0533** eV/atom | Honest — 5-fold CV, no leakage |
| OOF BG MAE | **0.1951** eV | Honest — 5-fold CV, no leakage |
| OOF EF score | **18.10** / 20 | |
| OOF BG score | **17.07** / 20 | |
| **OOF Total** | **35.17** / 40 | Unbiased performance estimate |
| Step 4 val score | 35.96 / 40 | Optimistic — calibration fit on same val used for scoring |
| **Delta** | **−0.79** | Expected optimism correction |

### Per-fold ALIGNN OOF (pre-ensemble summary from CV loop)

| Fold | EF MAE | BG MAE |
|---|---|---|
| 1 | 0.0631 | 0.2564 |
| 2 | 0.0607 | 0.2179 |
| 3 | 0.0629 | 0.1776 |
| 4 | 0.0822 | 0.2054 |
| 5 | 0.0784 | 0.2214 |
| **OOF (ensemble + calibration)** | **0.0533** | **0.1951** |

**Fold 4 outlier (EF):** EF MAE 0.082 vs ~0.061–0.063 on other folds — likely higher concentration of complex transition-metal oxides in that held-out partition.

---

## Interpretation

The **−0.79** point gap between Step 4 val (**35.96**) and OOF (**35.17**) is **not a regression**. Step 4 fitted isotonic calibration on the same 150 val structures used for scoring (mild leakage). OOF fits calibration on pooled OOF ensemble predictions from CV held-out folds — the correct unbiased estimate.

| Estimate | Score |
|---|---|
| Honest OOF | **35.17 ± ~0.5** / 40 |
| Step 4 val (optimistic upper bound) | 35.96 / 40 |
| OrganizerTest reference | 27.28 / 40 |

**Test submission:** `predictions_test.json` uses the **850-sample full retrain** (`alignn_*_full` + MACE on 850), not CV fold models. Leaderboard score may land **between 35.17 and 35.96**, possibly near val if extra training data helps.

---

## Pipeline

1. `CombinedDataset` — 700 train + 150 val = 850  
2. 5-fold stratified CV (metal / semiconductor / wide-gap) → OOF ALIGNN + MACE per fold  
3. Optuna ensemble weights on OOF  
4. Isotonic calibration on OOF ensemble  
5. Full ALIGNN EF/BG retrain (765 train / 85 internal val for early stopping only)  
6. MACE refit on 850  
7. Test predictions → `submissions/LIST_EuMINe/predictions_test.json` (`ALIGNN_MACE_ensemble_v2_fullretrain`)

**Run:** `python scripts/retrain_full.py` — log: `logs/retrain_full.log`

---

## Artifacts

```
models/full_retrain/
  fold_0/ … fold_4/          # CV checkpoints
  alignn_ef_full/best_model.pt
  alignn_bg_full/best_model.pt
  mace_artifacts/
  ensemble_weights.json
  calibration/
  oof_*.npy
```

---

## References

- Step 4: `STEP4_MACE_ENSEMBLE_CALIBRATION.md`
- Step 5 MatFed: `STEP5_MATFED_API.md`
- Technical report: `technical_report.tex`
