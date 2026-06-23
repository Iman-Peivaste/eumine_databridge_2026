# Step 5 — MatFed API v1 Implementation (Federation Readiness)

**Status:** Complete (checkpoint; validated on Step 4 artifacts)  
**Date:** 2026-05-15  
**Environment:** `conda ip`, CUDA; hackathon ref at `~/EuMINe/hackathon_ref/matfed-api-template`

---

## Objective

Implement a **MatFed API v1**–compliant predictor that wraps the full ensemble pipeline (ALIGNN + MACE + Optuna weights + isotonic calibration), pass all official compliance tests, and expose rich `describe()` metadata for jury federation scoring.

Step 5 does **not** require Step 4B (`full_retrain`) to be finished: artifact paths are fixed; the predictor auto-resolves `models/full_retrain/` when available, else `models/ensemble/` + Step 3 ALIGNN checkpoints.

---

## Checkpoint summary

| Item | Status |
|---|---|
| `LISTEuMINePredictor` implemented | Done |
| `_init_model_only()` added to `ALIGNNFineTuner` | Done |
| Our compliance suite (10 tests) | **10/10 passed** |
| Official hackathon suite (5 tests) | **5/5 passed** |
| Sample CIF end-to-end predictions | Working |
| `describe()` metadata | Complete |
| JSON schema validation | Passing |

**Federation readiness (hackathon scoring):**

| Component | Points | Status |
|---|---|---|
| Automated MatFed compliance (`pytest tests/test_interface.py`) | **15** | Secured — 5/5 official tests pass with `LISTEuMINePredictor` |
| Jury review of `describe()` quality | **5** | Strong — architecture, training provenance, uncertainty method, institution |
| **Effective federation block** | **~20/20** | Ready for Stage 2 federation sprint |

---

## Official compliance tests (hackathon reference)

Source: `hackathon_ref/matfed-api-template/tests/test_interface.py`

| # | Test name | What it checks |
|---|---|---|
| 1 | `test_predictor_is_abstract` | `MatFedPredictor` is abstract (cannot instantiate base class) |
| 2 | `test_predict_returns_list` | `predict(structures)` returns a `list` with **len == n_structures** |
| 3 | `test_predict_required_keys` | Each dict contains `formation_energy_per_atom`, `band_gap`, `model_id`, `data_sources_used` |
| 4 | `test_describe_required_fields` | `describe()` includes `team_name`, `model_type`, `api_version`, `data_sources` |
| 5 | `test_json_schema_valid` | Output passes `matfed_api.validate_predictions()` (JSON schema) |

**Fixture behavior:** `tests/conftest.py` imports the class from `MY_PREDICTOR` (default: example `RandomForestPredictor`). Our runner sets:

```bash
export MY_PREDICTOR="eumine_databridge.matfed.predictor.LISTEuMINePredictor"
export MATFED_MODEL_PATH="<models/full_retrain or models/ensemble>"
export PYTHONPATH="$ROOT/src:$MATFED_TESTS:$PYTHONPATH"
```

`LISTEuMINePredictor.__init__` auto-calls `load_model()` when `MATFED_MODEL_PATH` is set, because the official fixture does not call `load_model()` explicitly.

**Pre-check on example impl:** Running the official suite against the bundled `RandomForestPredictor` fails without `matminer` in `ip`; that is an environment gap on the reference example, not a spec issue. Our predictor has no `matminer` dependency.

---

## Extended compliance suite (project)

Source: `tests/test_matfed_compliance.py` (10 tests)

Mirrors the official five tests plus:

- Physical range checks (EF ∈ [−6, 3] eV/atom, BG ∈ [0, 15] eV)
- `api_version == "1.0"`
- Single-structure and empty-list `predict()` behavior
- Explicit schema validation import

**Result (Step 4 artifacts, `MATFED_MODEL_PATH=models/ensemble`):** **10/10 passed** (~46 s).

**Official re-run:** `bash scripts/run_matfed_compliance.sh` → **5/5 passed** (~27 s).

---

## Predictor architecture

```
MatFedPredictor (abstract)
    └── LISTEuMINePredictor
            load_model(model_path)
                ├── ALIGNN EF  (fine-tuned checkpoint)
                ├── ALIGNN BG  (fine-tuned checkpoint)
                ├── MACE-MP-0  (medium + ref energies + GBM BG head)
                ├── WeightedEnsemble (ensemble_weights.json)
                └── CalibrationLayer (isotonic EF + BG)
            predict(structures) → List[Dict]
            describe() → Dict (jury / federation metadata)
```

**Inference flow per structure batch:**

1. ALIGNN EF/BG `predict()` (graph inference, order-preserving)
2. MACE `predict_ef` / `predict_bg` (energy + embedding head)
3. NaN MACE → ALIGNN fallback (inference failures only)
4. Weighted blend (Optuna weights from CV or val)
5. Isotonic calibration + `clip(BG, 0)`
6. Optional uncertainty: `|ALIGNN − MACE|` per property

---

## Sample CIF end-to-end run

Structures: `hackathon_ref/matfed-api-template/tests/sample_structures/test_001.cif` … `test_005.cif`  
Artifacts: `models/ensemble/` (Step 4; `model_version: v1_ensemble`)

| Material | EF (eV/atom) | BG (eV) | Unc EF | Unc BG |
|---|---|---|---|---|
| struct_01 | −0.1009 | 1.6058 | 0.1305 | 0.5205 |
| struct_02 | 0.1476 | 0.0000 | 0.4043 | 0.4043 |
| struct_03 | **−1.7537** | 2.4625 | **2.4600** | 0.1936 |
| struct_04 | 0.0337 | 0.0000 | 0.1322 | 1.6297 |
| struct_05 | −0.1328 | 0.0000 | 0.0524 | 1.0415 |

### Technical note: struct_03 high EF uncertainty

**struct_03** has **Unc EF = 2.46 eV/atom** — ALIGNN and MACE disagree by ~2.46 eV/atom on that structure. This is a deliberate feature of our uncertainty proxy (`|ALIGNN_EF − MACE_EF|`), not noise:

- When both models agree, uncertainty is low (e.g. struct_01: 0.13).
- When they diverge strongly, the prediction is **less reliable** and should be flagged for federation / human review.

Likely causes: unusual composition or geometry **far from the Bridge training distribution** (850 labeled structures). Worth citing in the **Technical Report** as a concrete example of epistemic disagreement between line-graph (ALIGNN) and equivariant-potential (MACE) inductive biases.

BG uncertainty on struct_03 remains moderate (0.19) because the ensemble is ~99% ALIGNN-weighted for BG.

---

## `describe()` metadata (jury-facing)

`LISTEuMINePredictor.describe()` returns a structured dict including:

| Section | Content |
|---|---|
| Identity | `team_name`, `institution` (LIST), `contact`, `repository` |
| API | `api_version: "1.0"`, `model_version` (`v1_ensemble` or `v2_fullretrain`) |
| Model | `model_type`, `properties_predicted`, `data_sources` |
| Architecture | ALIGNN layers, MACE medium, Optuna ensemble, isotonic calibration |
| Training | 850 structures (post-4B), 5-fold CV, JARVIS/MP pretraining |
| Performance | OOF MAE / score (filled from `predictions_test.json` or `oof_metrics.json` when present) |
| Uncertainty | `uncertainty_available: true`, method documented |

**After Step 4B completes:** update `performance` from 4B OOF report:

```python
"performance": {
    "oof_mae_ef_eV_per_atom": <4B OOF EF MAE>,
    "oof_mae_bg_eV": <4B OOF BG MAE>,
    "oof_score_40pts": <4B OOF total>,
}
```

Auto-loaded via `_load_performance_block()` when submission JSON contains `oof_mae_ef`, `oof_mae_bg`, `oof_score`.

---

## Implementation

### Code added

| Path | Role |
|---|---|
| `src/eumine_databridge/matfed/predictor.py` | `LISTEuMINePredictor` — MatFed wrapper |
| `src/eumine_databridge/matfed/__init__.py` | Public export |
| `tests/test_matfed_compliance.py` | 10-test compliance mirror |
| `scripts/run_matfed_compliance.sh` | Official 5-test runner with env vars |

### Changes to existing code

| Path | Change |
|---|---|
| `src/eumine_databridge/models/alignn_model.py` | `_init_model_only()` for inference-only load; `predict()` padding for `n_train≥1`; unique LMDB `filename` per material-id hash (avoids stale cache); single-structure batch fix |

### Artifact path resolution

| Layout | `model_path` | ALIGNN | MACE / ensemble |
|---|---|---|---|
| Step 4B (target) | `models/full_retrain/` | `alignn_ef_full/`, `alignn_bg_full/` | same dir: `mace_artifacts/`, `ensemble_weights.json`, `calibration/` |
| Step 4 (current tests) | `models/ensemble/` | `../alignn_ef/`, `../alignn_bg/` | `ensemble/*` |

### Run commands

```bash
conda activate ip
cd ~/EuMINe/eumine_databridge

# Our extended suite
MATFED_MODEL_PATH=models/ensemble pytest tests/test_matfed_compliance.py -v

# Official hackathon suite
bash scripts/run_matfed_compliance.sh

# Sample CIF smoke (5 structures)
MATFED_MODEL_PATH=models/ensemble python -c "
# see scripts/run_matfed_compliance.sh for PYTHONPATH setup
"
```

After 4B: `MATFED_MODEL_PATH=models/full_retrain bash scripts/run_matfed_compliance.sh`

---

## Issues resolved during Step 5

| Issue | Resolution |
|---|---|
| Official tests never call `load_model()` | Auto-load from `MATFED_MODEL_PATH` in `__init__` |
| `alignn.pretrained` broken in Step 3 | `_init_model_only()` uses `load_pretrained_alignn()` |
| `predict()` with 1 structure → `ValueError` (n_train+n_val > n) | Always prepend one padded graph |
| Stale LMDB (`n_val: 150` for 5 inputs) | Per-batch MD5 tag in `alignn_infer_<hash>_` filename |
| Example RF tests fail without matminer | Documented; not required for our predictor |

---

## Schema contract (predict output)

Required keys per prediction (validated by `matfed_api.schema`):

- `formation_energy_per_atom` (number)
- `band_gap` (number)
- `model_id` (string)
- `data_sources_used` (array of strings)

Optional (allowed by schema, `additionalProperties: true`):

- `uncertainty_ef`, `uncertainty_bg` (our implementation)
- `uncertainty` (schema example key; we use property-specific names)

---

## Bridge to Step 4B and Step 6

| Step | Action |
|---|---|
| **4B** | Complete — OOF **35.17/40**; see `STEP4B_FULL_RETRAIN_CV.md` |
| **5 (post-4B)** | Re-run `run_matfed_compliance.sh` on `full_retrain`; `describe()` reads `oof_metrics.json` |
| **6 Submission** | PR, registration form, final `predictions_test.json` |

### Parallel writing (no GPU dependency)

While 4B runs:

| Option | Deliverable | Hackathon points |
|---|---|---|
| **A — Data Integration Report** | ~4 pages: 603 MP/JARVIS pairs, MAE 0.144/0.218, R² 0.961/0.861, harmonization | **25** |
| **B — Technical Report** | ~4 pages: architecture, training, validation, federation; plug 4B OOF when ready | **15** (reproducibility + jury) |

Numbers for A are final from Step 2 (`data/processed/harmonizer_params.json`). Numbers for B: performance block uses Step 4 val **35.96/40** until 4B OOF replaces them.

---

## References

- MatFed spec: `hackathon_ref/matfed-api-template/README.md`, `matfed_api/predictor.py`
- Step 4 ensemble: `reports/technical/STEP4_MACE_ENSEMBLE_CALIBRATION.md`
- Step 3 ALIGNN: `reports/technical/STEP3_ALIGNN_FINETUNING.md`
- Compliance logs: local runs 2026-05-15 (10/10 + 5/5)
