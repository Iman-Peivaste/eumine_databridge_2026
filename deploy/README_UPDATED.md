# CataLIST — Stage 2 Deployment Guide (Updated)

**Latest model:** v4_combined (38.38/40 validation score)  
**Setup time:** 5 minutes (local) | 30 seconds (Colab)  
**Risk mitigation:** Health check + Colab backup

---

## 🚀 Quick Start

### Option 1: Local Machine (Recommended for Cluj)

```bash
# Clone and install
git clone https://github.com/Iman-Peivaste/eumine_databridge_2026.git
cd eumine_databridge_2026
bash deploy/install.sh

# Verify everything works
conda activate catallist_stage2
python deploy/verify.py
```

Expected output:
```
[✓] 7/7 checks passed
STATUS: READY FOR STAGE 2 SPRINT
```

**Timeline:**
- 2 min: Clone repo
- 10 min: Conda environment + pip installs
- 2 min: Health check
- **Total: 15 minutes**

### Option 2: Google Colab (For Testing Before Cluj)

1. Open notebook: [`deploy/CataLIST_Colab_Setup.ipynb`](./CataLIST_Colab_Setup.ipynb)
2. Run cells in order
3. Done in ~30 seconds

Colab link (open in browser):
```
https://colab.research.google.com/github/Iman-Peivaste/eumine_databridge_2026/blob/main/deploy/CataLIST_Colab_Setup.ipynb
```

---

## 📋 What Changed

### Updates to Deploy Package:
- ✅ `verify.py` — Now uses latest `combined_retrain` model (v4_combined)
- ✅ `install.sh` — Health check calls correct model path
- ✅ `colab_notebook.py` — New pip-based setup script
- ✅ `CataLIST_Colab_Setup.ipynb` — New interactive Colab notebook

### Why This Matters:
- **Before:** You might accidentally test with old model (v2_fullretrain, 35.17/40 OOF)
- **After:** You're always testing/using best model (v4_combined, 38.38/40 validation)
- **Colab:** No conda needed — test your model from anywhere

---

## 🔧 Installation Details

### Requirements

**Local (GPU recommended):**
- Linux or macOS
- conda/miniconda installed
- 8 GB RAM
- NVIDIA GPU (optional, CPU works)
- 14.3 GB disk (models)

**Colab:**
- Google account
- Browser
- No installation needed!

### GPU vs CPU

Both `install.sh` and `install_cpu.sh` work. Colab auto-detects.

| Aspect | GPU | CPU |
|--------|-----|-----|
| Install time | 10 min | 10 min |
| Model load | 5s | 12s |
| Predict 1 structure | 1-2s | 5-10s |
| Recommend for Cluj | ✓ YES | OK (slower) |

### Troubleshooting

| Error | Fix |
|-------|-----|
| `conda: command not found` | Install miniconda: https://docs.conda.io |
| `No module named 'alignn'` | `pip install alignn==2026.4.2` |
| CUDA version mismatch | Use `install_cpu.sh` instead |
| `matfed_api not found` | Clone `../hackathon_ref/matfed-api-template` |
| Out of memory | Use CPU mode (`--cpu_only` flag) |

---

## 🧪 Health Check (verify.py)

Run after installation to catch problems:

```bash
conda activate catallist_stage2
python deploy/verify.py
```

What it tests:
1. ✓ Core imports (torch, pymatgen, alignn, mace, matfed_api)
2. ✓ GPU/CPU detection
3. ✓ Model artifacts present
4. ✓ Predictor loads
5. ✓ API compliance (describe() method)
6. ✓ End-to-end prediction (3 structures)
7. ✓ Federation engine ready

**If all pass:** You're good for Cluj! 🎯

**If any fail:** Error message tells you exactly what's wrong.

---

## 📖 Usage Examples

### Basic Prediction

```python
from eumine_databridge.matfed.predictor import LISTEuMINePredictor
from pymatgen.core import Structure

# Load model
predictor = LISTEuMINePredictor()
predictor.load_model("models/combined_retrain")

# Load structures
structures = [
    Structure.from_file("Si.cif"),
    Structure.from_file("GaAs.cif"),
]

# Predict
predictions = predictor.predict(structures)

for pred in predictions:
    print(f"Formation energy: {pred['formation_energy_per_atom']:.4f} eV/atom")
    print(f"Band gap: {pred['band_gap']:.4f} eV")
    print(f"Uncertainty EF: ±{pred['uncertainty_ef']:.4f}")
    print(f"Uncertainty BG: ±{pred['uncertainty_bg']:.4f}")
```

### Federation (For Cluj)

```python
from eumine_databridge.matfed.federation import FederatedEnsemble

# Create federation
fed = FederatedEnsemble()

# Add our model
our = LISTEuMINePredictor()
our.load_model("models/combined_retrain")
fed.add_predictor(our, "CataLIST")

# Add partner models
# fed.add_predictor(takeme2romania_model, "TakeMe2Romania")
# fed.add_predictor(prophx_model, "ProphX")

# Fit on calibration data (provided by organizers)
result = fed.fit(
    cal_structures=cal_structures,
    cal_ef=cal_ef,
    cal_bg=cal_bg,
    n_trials=200,
)

# Predict on test set
predictions = fed.predict(test_structures)
```

### Model Info

```python
desc = predictor.describe()

print(f"Team: {desc['team_name']}")
print(f"Model version: {desc['model_version']}")
print(f"Architecture:")
print(f"  ALIGNN BG layers: {desc['architecture']['alignn_layers_bg']}")
print(f"  ALIGNN EF layers: {desc['architecture']['alignn_layers_ef']}")
print(f"  Ensemble method: {desc['architecture']['ensemble_method']}")
print(f"Performance (OOF): {desc['performance']['oof_total_score_40pts']}/40")
```

---

## ☁️ Google Colab Workflow

### For Testing Before Cluj:

1. **Clone & Setup** (30s)
   ```
   Run "Step 1: Clone Repository & Install Dependencies"
   ```

2. **Mount Google Drive** (optional)
   ```
   If you have model artifacts in Drive
   ```

3. **Load Predictor** 
   ```
   Run "Step 3: Initialize Predictor"
   ```

4. **Test Predictions** (1-2s per structure)
   ```
   Run "Step 4: Test Prediction" or "Step 6: Use Your Own Structures"
   ```

5. **Test Federation Interface**
   ```
   Run "Step 5: Federation Test (For Cluj)"
   ```

### Benefits:
- ✅ No local CUDA setup needed
- ✅ Free GPU (T4, fast for inference)
- ✅ Share link with team for testing
- ✅ Practice before Cluj
- ✅ Works from phone/tablet in a pinch

---

## 📊 Performance (Approximate)

| Operation | GPU (RTX 4090) | CPU (Intel) | Colab GPU (T4) |
|---|---|---|---|
| Load model | 5s | 12s | 8s |
| Predict 1 structure | 1.2s | 8s | 2s |
| Predict 150 structures | 30s | 4 min | 90s |
| Federation fit (200 trials) | 3 min | 8 min | 5 min |
| **Full sprint pipeline** | **10 min** | **20 min** | **15 min** |

In Cluj with 2 hours, you have plenty of time. 🎯

---

## 🎯 Cluj Sprint Checklist (Day Of)

- [ ] **Before arriving:** Test on Colab or local machine
- [ ] **Arrive early:** Run `bash deploy/install.sh`
- [ ] **Verify:** `python deploy/verify.py` — all 7 checks pass?
- [ ] **Connect to organizers:** Get calibration data
- [ ] **Load partners' models:** One by one, test their predictor classes
- [ ] **Federate:** `fed.fit()` then `fed.predict()`
- [ ] **Submit:** Format predictions JSON
- [ ] **Victory:** 🎉

---

## 📦 What's Included

```
deploy/
├── install.sh                  ← GPU installer
├── install_cpu.sh              ← CPU-only installer
├── environment.yml             ← GPU dependencies
├── environment_cpu.yml         ← CPU dependencies
├── verify.py                   ← Health check (7 tests)
├── colab_notebook.py           ← Pip-based setup script
├── CataLIST_Colab_Setup.ipynb  ← Interactive Colab notebook
├── README_UPDATED.md           ← This file
└── README.md                   ← Original README

Key scripts in root:
├── scripts/
│   ├── federate.py             ← Stage 2 federation CLI
│   ├── sprint_launcher.py      ← Interactive federation launcher
│   ├── _sprint_sim.py          ← Simulation mode
│   └── ... (other training scripts)
```

---

## 🆘 Support

| Issue | Solution |
|-------|----------|
| Installation fails | Run `deploy/verify.py` — tells you exactly what's wrong |
| Model won't load | Check disk space (14.3 GB) + run health check |
| Prediction is slow | Use GPU or batch process structures |
| In Cluj, partner model fails | Load it with try/except, continue without it |
| Forgot to test before Cluj | Colab link takes 30 seconds 😅 |

**Contact:** euminecost@gmail.com

---

## ✨ What Makes This Robust

✅ **Reproducible:** Pinned package versions (torch==2.4.0, not torch>=2.0)  
✅ **Testable:** Health check verifies before sprint  
✅ **Flexible:** GPU + CPU + Colab options  
✅ **Isolated:** Conda environment won't break system Python  
✅ **Automatic:** `install.sh` handles all steps  
✅ **Fast:** 5 min local, 30s Colab  
✅ **Safe:** Federation still works if partner model fails  

---

**You're ready for Cluj.** 🚀
