# EuMINe DataBridge Hackathon 2026
## Team: CataLIST

[![MatFed API v1](https://img.shields.io/badge/MatFed_API-v1.0-blue)]()
[![Model v4](https://img.shields.io/badge/Model-v4_combined-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

**Predicting formation energy and band gap of inorganic materials
from heterogeneous multi-database data.**

- **Stage 1 completed:** June 27, 2026 ✅
- **Current model:** v4_combined (latest: 38.38/40 validation)
- **Team:** Iman Peivaste (LIST, Luxembourg) · Saeideh Ghaderi (University of Bologna) · Halliru Ibrahim (LIST, Luxembourg) · Dragos Vovea (Universitatea Babeș-Bolyai, Cluj-Napoca)
- **Performance:** 
  - Validation score: **38.38/40** (v4_combined, 14,287 structures)
  - 5-fold OOF: **35.17/40** (honest cross-validation)
  - EF MAE: **0.052 eV/atom** | BG MAE: **0.150 eV**

---

## Model Architecture

**v4_combined (Latest):**
- ALIGNN EF (4 layers) fine-tuned from JARVIS pretrained
- ALIGNN BG (8 layers) fine-tuned on 14,287 structures
- MACE-MP-0 (medium) pre-trained from Materials Project
- Optuna-optimized weighted ensemble
- Isotonic regression calibration

```
CIF Structure
     │
     ├── ALIGNN EF (4 layers) ───┐
     ├── ALIGNN BG (8 layers) ───┤
     ├── MACE-MP-0 EF ───────────┤ Weighted Ensemble → Calibration → Prediction
     └── MACE-MP-0 BG head ──────┘

Ensemble Weights (Optuna-optimized):
  • EF: ALIGNN=51.3%, MACE=48.7%
  • BG: ALIGNN=10.0%, MACE=90.0% ← MACE stronger for band gap
```

**Training data:** 
- Bridge Dataset (850) + Augmented structures (13,437)
- Semiconductors (12,337), Layered/2D (500), Rare-earth (600)

**Data sources:** Materials Project · JARVIS-DFT · OQMD · AFLOW

---

## Repository Structure

```
eumine_databridge/
├── src/eumine_databridge/
│   ├── data/
│   │   ├── loader.py          # BridgeDataset loader
│   │   ├── combined_loader.py # Merge train+val for retraining
│   │   ├── harmonizer.py      # JARVIS→MP correction
│   │   └── fetchers/          # MP, JARVIS, OQMD, AFLOW APIs
│   ├── models/
│   │   ├── alignn_model.py      # ALIGNN fine-tuner
│   │   ├── alignn_config.py     # Config (EF: 4L, BG: 8L)
│   │   ├── alignn_data.py       # Structure→graph conversion
│   │   ├── mace_model.py        # MACE-MP-0 wrapper
│   │   └── ensemble.py          # WeightedEnsemble + Calibration
│   ├── matfed/
│   │   ├── predictor.py         # MatFed API v1 (LISTEuMINePredictor)
│   │   └── federation.py        # Stage 2: FederatedEnsemble
│   └── utils/
│       ├── metrics.py           # Scoring formula
│       └── scoring.py           # Calibration helpers
├── deploy/                      # ⭐ Stage 2 deployment package
│   ├── install.sh              # GPU installer
│   ├── install_cpu.sh          # CPU fallback
│   ├── environment.yml         # Pinned dependencies
│   ├── environment_cpu.yml     # CPU dependencies
│   ├── verify.py               # Health check (7 tests)
│   ├── CataLIST_Colab_Setup.ipynb  # ⭐ NEW: Colab notebook
│   ├── README_UPDATED.md       # ⭐ NEW: Complete guide
│   ├── DEPLOYMENT_SUMMARY.txt  # ⭐ NEW: Quick reference
│   └── colab_notebook.py       # ⭐ NEW: Pip-based setup
├── scripts/
│   ├── train_alignn.py              # Basic training
│   ├── train_mace.py                # MACE fine-tuning
│   ├── run_ensemble.py              # Weight optimization (Optuna)
│   ├── retrain_full.py              # Step 4: Full retrain on train+val
│   ├── retrain_augmented.py         # v3: Semiconductors (38.32/40)
│   ├── retrain_combined_augmented.py # ⭐ v4: ALL data (38.38/40) LATEST
│   ├── fetch_external_data.py       # Download MP/JARVIS/OQMD/AFLOW
│   ├── augment_semiconductors.py    # Generate semiconductor structures
│   ├── sprint_launcher.py           # ⭐ Stage 2: Interactive CLI
│   ├── federate.py                  # Stage 2: Federation engine
│   ├── verify.py                    # MatFed compliance test
│   └── generate_submission.py       # Format predictions JSON
├── models/
│   ├── combined_retrain/    # ⭐ LATEST v4 (38.38/40 validation)
│   │   ├── alignn_ef_combined/
│   │   ├── alignn_bg_combined/
│   │   ├── ensemble_weights.json
│   │   └── calibration/
│   ├── full_retrain/        # v2 (35.17/40 OOF)
│   └── augmented_retrain/   # v3 intermediate
├── data/
│   ├── raw/                 # Bridge Dataset + CIF structures
│   ├── augmented/           # ⭐ 14,287 total structures
│   │   ├── augmentation_dataset.csv
│   │   ├── semiconductors/  # 12,337 structures
│   │   ├── layered_structures/
│   │   └── rare_earth/
│   └── processed/           # Harmonizer params, weights
├── submissions/
│   └── CataLIST/
│       ├── predictions_test.json      # Final submission (v4)
│       └── predictions_dry_run.json   # Sprint dry-run
├── tests/
│   ├── test_matfed_compliance.py      # 10 API tests
│   ├── test_loader.py
│   └── test_metrics.py
└── logs/
    └── *.log               # Training logs
```

---

## ⚡ Quick Start

### Option 1: Test on Google Colab (30 seconds, no local setup)

```
Open: https://colab.research.google.com/github/Iman-Peivaste/eumine_databridge_2026/blob/main/deploy/CataLIST_Colab_Setup.ipynb

1. Run Cell 1 (dependencies, 60s)
2. Run Cell 3 (load model, 5s)
3. Run Cell 4 (predict, 10s)
→ See predictions ✓
```

### Option 2: Local Setup (15 minutes)

```bash
# Clone and setup
git clone https://github.com/Iman-Peivaste/eumine_databridge_2026.git
cd eumine_databridge_2026
bash deploy/install.sh

# Verify everything works
python deploy/verify.py
# Expected: 7/7 checks passed ✓

# Make predictions
from eumine_databridge.matfed.predictor import LISTEuMINePredictor
predictor = LISTEuMINePredictor()
predictor.load_model("models/combined_retrain")
predictions = predictor.predict([structure])
```

## Full Reproduction Pipeline (Optional)

### 1. Environment

```bash
bash deploy/install.sh
conda activate catallist_stage2
```

### 2. Download data

```bash
python scripts/fetch_external_data.py
cd data/raw && unzip *.zip
```

### 3. Set API keys

```bash
cp .env.example .env
# Add MP_API_KEY and optionally WANDB_API_KEY
```

### 4. Train v4 model (latest, 38.38/40)

```bash
# Uses all available data with per-source weights
python scripts/retrain_combined_augmented.py
# Output: models/combined_retrain/
```

### 5. Run compliance tests

```bash
python deploy/verify.py --model_path models/combined_retrain
# Expected: 7/7 tests pass ✓
```

### 6. Generate predictions

```bash
python scripts/generate_submission.py
# Output: submissions/CataLIST/predictions_test.json
```

---

## MatFed API v1 Usage

### Basic Prediction

```python
from eumine_databridge.matfed.predictor import LISTEuMINePredictor
from pymatgen.core import Structure

# Initialize predictor (auto-loads latest v4_combined model)
predictor = LISTEuMINePredictor()

# Or specify model path
predictor.load_model('models/combined_retrain')

# Predict on structures
structures = [Structure.from_file('my_structure.cif')]
results = predictor.predict(structures)

# Results include:
for pred in results:
    print(f"Formation energy: {pred['formation_energy_per_atom']:.4f} eV/atom")
    print(f"Band gap: {pred['band_gap']:.4f} eV")
    print(f"Uncertainty EF: ±{pred['uncertainty_ef']:.4f}")
    print(f"Uncertainty BG: ±{pred['uncertainty_bg']:.4f}")
```

### Model Information

```python
desc = predictor.describe()

print(f"Team: {desc['team_name']}")              # CataLIST
print(f"Model version: {desc['model_version']}") # v4_combined
print(f"API version: {desc['api_version']}")    # 1.0
print(f"Architecture: {desc['model_type']}")    # ALIGNN + MACE ensemble
print(f"Performance (OOF): {desc['performance']['oof_total_score_40pts']}/40")
```

### Stage 2: Federation

```python
from eumine_databridge.matfed.federation import FederatedEnsemble

# Create federation
fed = FederatedEnsemble()

# Add your model
our_model = LISTEuMINePredictor()
fed.add_predictor(our_model, "CataLIST")

# Add partner models
# fed.add_predictor(their_model, "TakeMe2Romania")

# Fit on calibration data
result = fed.fit(
    cal_structures=cal_structures,
    cal_ef=cal_ef,
    cal_bg=cal_bg,
    n_trials=200,
)

# Predict on test set
test_predictions = fed.predict(test_structures)
```

---

## Performance Results

| Model | Training Data | EF MAE (eV/atom) | BG MAE (eV) | Validation Score |
|---|---|---|---|---|
| Baseline (RF+MAGPIE) | 850 | 0.238 | 0.641 | 20.0/40 |
| v1: ALIGNN only | 850 | 0.073 | 0.210 | 34.1/40 |
| v2: + MACE ensemble | 850 | 0.069 | 0.202 | 34.4/40 |
| v2.1: + Calibration | 850 | 0.052 | 0.150 | **35.96/40** |
| v3: + Semiconductors | 2,187 | N/A | N/A | 38.32/40 |
| **v4: ALL augmented** | **14,287** | **0.052** | **0.150** | **38.38/40** ⭐ |
| **v4 OOF (5-fold CV)** | **14,287** | **0.053** | **0.195** | **35.17/40** |

**v4 Architecture:**
- ALIGNN EF: 4 layers (trained on 14,287 structures)
- ALIGNN BG: 8 layers (expanded capacity for larger dataset)
- MACE-MP-0: Medium model (strong for band gap)
- Ensemble weights: EF (51% ALIGNN / 49% MACE), BG (10% ALIGNN / 90% MACE)
- Calibration: Isotonic regression (per-property)

---

## Performance & Requirements

### Hardware

| Operation | GPU (RTX 4090) | GPU (T4 Colab) | CPU |
|-----------|---|---|---|
| **Setup** | 10 min | 60s | 10 min |
| **Load model** | 5s | 8s | 12s |
| **Predict 1 struct** | 1.2s | 2s | 8s |
| **Predict 150 structs** | 30s | 90s | 4 min |
| **Federation fit (200 trials)** | 3 min | 5 min | 8 min |

### Resources

- **GPU:** NVIDIA RTX 4090 (used for training)
- **RAM:** 8 GB minimum (16 GB recommended)
- **Disk:** 14 GB (model + augmented dataset)
- **CPU-compatible:** Yes (slower, but works)
- **Colab:** Free T4 GPU included, sufficient for inference

---

## 📚 Documentation

- **Deployment:** `deploy/README_UPDATED.md` — Complete setup guide
- **Quick Reference:** `deploy/DEPLOYMENT_SUMMARY.txt` — Visual checklist
- **Colab Setup:** `deploy/CataLIST_Colab_Setup.ipynb` — 30-second cloud testing
- **Detailed Guide:** `DEPLOYMENT_TUTOR.md` — System explanation

## 📝 Submission Status

| Item | Status |
|------|--------|
| Model trained (v4_combined) | ✅ 38.38/40 validation |
| Deployment package | ✅ conda + verify.py |
| Colab support | ✅ Interactive notebook |
| MatFed API v1 | ✅ 10/10 compliance tests pass |
| Documentation | ✅ 4,000+ lines |
| Predictions generated | ✅ predictions_test.json |
| GitHub ready | ✅ Push pending |

## License

MIT License. See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).

## Contact

- **Team:** CataLIST (Luxembourg Institute of Science and Technology)
- **Members:** Iman Peivaste · Saeideh Ghaderi · Halliru Ibrahim · Dragos Vovea
- **Email:** iman.peivaste@list.lu -- euminecost@gmail.com
- **Repository:** [github.com/Iman-Peivaste/eumine_databridge_2026](https://github.com/Iman-Peivaste/eumine_databridge_2026)
- **Hackathon:** [EuMINe DataBridge 2026](https://www.eumine-cost.eu/news/eumine-hackathon-2026/)
