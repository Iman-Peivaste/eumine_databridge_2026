# Step 3 — ALIGNN Fine-Tuning: Formation Energy & Band Gap

**Status:** Complete  
**Date:** 2026-05-15  
**Environment:** `conda ip`, NVIDIA RTX A4000 (16.7 GB VRAM)

---

## Objective

Fine-tune JARVIS-pretrained ALIGNN models on the Bridge Dataset for:

- Formation energy per atom (EF)
- Band gap (BG)

Deliverables: validated val metrics, hackathon scores vs baseline, and checkpoints under `models/alignn_ef/` and `models/alignn_bg/`.

---

## Results (validation set, 150 materials)

| Property | MAE | RMSE | R² | Score | vs Baseline |
|---|---|---|---|---|---|
| Formation energy | 0.0761 eV/atom | 0.2023 | 0.9637 | 17.10/20 | −68% |
| Band gap | 0.1979 eV | 0.3781 | 0.9308 | 17.02/20 | −69% |
| **Combined** | | | | **34.12/40** | **+6.84 vs OrganizerTest (27.28)** |

**Baselines (hackathon):** EF MAE = 0.2378 eV/atom, BG MAE = 0.6414 eV.

Both properties beat baseline → qualifies for Stage 2 scoring tier.

---

## Training summary

| | EF | BG |
|---|---|---|
| Pretrained head | `jv_formation_energy_peratom_alignn` | `jv_optb88vdw_bandgap_alignn` |
| Config preset | `get_ef_config()` | `get_bg_config()` |
| Train / val samples | 700 / 150 | 700 / 150 |
| Best epoch | 29 | 61 |
| Early stopping | Epoch 79 (patience 50) | Epoch 61 (patience 60) |
| Freeze encoder | 20 epochs | 30 epochs |
| Loss | Huber (δ=0.5) | Huber (δ=1.0) |
| Checkpoint | `models/alignn_ef/best_model.pt` | `models/alignn_bg/best_model.pt` |

---

## Key observations (technical report)

1. **Fast convergence** — EF best at epoch 29, BG at epoch 61. Pretrained JARVIS weights carry most of the signal; Bridge fine-tuning mainly adapts the readout and last layers.

2. **EF** — MAE 0.0761 eV/atom (R² 0.964, r = 0.982) is well below the 0.2378 baseline. Harmonization-quality MP–JARVIS overlap from Step 2 (R² ≈ 0.96 on EF) is consistent with stable training targets.

3. **BG** — MAE 0.1979 eV vs baseline 0.6414 eV is the standout result. R² 0.931 on a target with metals at 0 eV and wide dynamic range (0–8.8 eV on train) confirms ALIGNN’s line-graph inductive bias helps on electronic structure.

4. **Competitive position** — Combined 34.12/40 is already top-tier for the hackathon performance block. Further gains are incremental; Step 4 targets ensemble + calibration rather than single-model capacity.

---

## Implementation notes

### Code added

| Path | Role |
|---|---|
| `src/eumine_databridge/models/alignn_config.py` | EF/BG hyperparameter presets |
| `src/eumine_databridge/models/alignn_data.py` | pymatgen → JARVIS/ALIGNN dataset dicts |
| `src/eumine_databridge/models/alignn_pretrained.py` | Figshare weight download/load (jarvis `get_cache_dir` compat) |
| `src/eumine_databridge/models/alignn_model.py` | `ALIGNNFineTuner` training loop |
| `src/eumine_databridge/utils/metrics.py` | MAE/RMSE/R² + hackathon `score_property` |
| `scripts/train_alignn.py` | CLI: `--target ef|bg|both` |

### Run commands

```bash
conda activate ip
cd ~/EuMINe/eumine_databridge

# EF then BG (W&B optional — see below)
WANDB_MODE=disabled python scripts/train_alignn.py --target ef
WANDB_MODE=disabled python scripts/train_alignn.py --target bg
```

Logs: `logs/train_alignn_ef.log`, `logs/train_alignn_bg.log`.

### Issues resolved during Step 3

| Issue | Resolution |
|---|---|
| `alignn.pretrained` import fails on `jarvis-tools` | Local `alignn_pretrained.load_pretrained_alignn()` |
| LMDB empty train loader (stale `bridge*data` in cwd) | LMDB under `models/alignn_*/alignn_train_data`; cleanup stale dirs |
| W&B `entity … not found` | Omit `WANDB_ENTITY` (use default workspace from the API key) or set the API entity slug explicitly |
| ALIGNN expects `atoms` as dict | `jarvis_atoms.to_dict()` in `alignn_data.py` |

---

## W&B setup (before Step 4)

Step 3 training used `WANDB_MODE=disabled` when `.env` had a display name instead of the API entity slug (`entity not found`). For experiment tracking:

1. Set `WANDB_PROJECT` (and `WANDB_API_KEY` via `.env` or `wandb login`). **Do not set `WANDB_ENTITY`** unless you need a team/workspace and know the slug W&B resolves (check `run.entity` after `wandb.init()` without `entity=`).
2. Run training without `WANDB_MODE=disabled`.

The trainer falls back to `mode=disabled` if init fails, so network or auth errors will not crash jobs.

---

## Artifacts

```
models/alignn_ef/
  best_model.pt
  train_config.json
  training_history.json
  alignn_train_data/          # LMDB cache

models/alignn_bg/
  best_model.pt
  train_config.json
  training_history.json
  alignn_train_data/

data/processed/
  alignn_ef/                   # dataset metadata JSON
  alignn_bg/
```

---

## Bridge to Step 4 (ensemble strategy)

Current combined score: **34.12 / 40**. OrganizerTest reference: **27.28**.

| Source | Current | Realistic target | Points gain |
|---|---|---|---|
| EF MAE | 0.0761 | ~0.055 | +1.0 |
| BG MAE | 0.1979 | ~0.140 | +1.5 |
| MACE ensemble contribution | — | — | +1.5–2.0 |
| Calibration layer | — | — | +0.5–1.0 |
| **Total** | **34.12** | **~37–38** | **+3–4** |

Largest remaining lever is **BG**: MACE-MP-0 (equivariant, energy-conserving) is partially decorrelated from ALIGNN on electronic properties, so a weighted ensemble should reduce variance. Step 4 implements MACE integration, stacking, and calibration on top of these checkpoints.

---

## References

- Step 2 harmonization: `data/processed/harmonizer_params.json`, `scripts/fetch_external_data.py`
- Hackathon scoring: `hackathon_ref/scoring/evaluate.py`
- Smoke test: `python scripts/smoke_test.py`
