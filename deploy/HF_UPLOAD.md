# Hugging Face Hub — Upload Guide

How to re-upload model artifacts to `Imanpeivaste/catallist_v4_combined` if needed.

## Prerequisites

```bash
conda activate ip
hf auth login --force   # paste a Write-permission token from huggingface.co/settings/tokens
hf auth whoami          # should show: user=Imanpeivaste
```

## What to Upload

Only the `models/catallist_v4_combined/` directory (~33 MB):

```
catallist_v4_combined/
├── alignn_ef_combined/best_model.pt     (16 MB)
├── alignn_bg_combined/best_model.pt     (16 MB)
├── mace_artifacts/
│   ├── ref_energies.json
│   ├── bg_head.joblib
│   └── bg_scaler.joblib
├── calibration/
│   ├── ef_calibrator.joblib
│   └── bg_calibrator.joblib
└── ensemble_weights.json
```

Note: the MACE base model (~43 MB) is NOT uploaded — it downloads automatically
at runtime via `mace_mp(model="medium")`.

## Upload Command

Run from inside `eumine_databridge/`:

```bash
conda run -n ip python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='models/catallist_v4_combined',
    repo_id='Imanpeivaste/catallist_v4_combined',
    repo_type='model',
)
print('Upload complete.')
"
```

## Rebuild the Package (if model was retrained)

If you retrain and get new checkpoints, rebuild `catallist_v4_combined/` first:

```bash
BASE=models
DST=$BASE/catallist_v4_combined

# ALIGNN checkpoints
cp $BASE/combined_retrain/alignn_ef_combined/best_model.pt $DST/alignn_ef_combined/
cp $BASE/combined_retrain/alignn_bg_combined/best_model.pt $DST/alignn_bg_combined/

# Ensemble + calibration
cp $BASE/combined_retrain/ensemble_weights.json $DST/
cp $BASE/combined_retrain/calibration/ef_calibrator.joblib $DST/calibration/
cp $BASE/combined_retrain/calibration/bg_calibrator.joblib $DST/calibration/

# MACE artifacts (from full_retrain, reused)
cp $BASE/full_retrain/mace_artifacts/ref_energies.json $DST/mace_artifacts/
cp $BASE/full_retrain/mace_artifacts/bg_head.joblib    $DST/mace_artifacts/
cp $BASE/full_retrain/mace_artifacts/bg_scaler.joblib  $DST/mace_artifacts/
```

Then run the upload command above.

## Verify Upload

```bash
conda run -n ip python -c "
from huggingface_hub import HfApi
for f in HfApi().list_repo_files('Imanpeivaste/catallist_v4_combined', repo_type='model'):
    print(f)
"
```

Expected output (8 files + .gitattributes):
```
.gitattributes
alignn_bg_combined/best_model.pt
alignn_ef_combined/best_model.pt
calibration/bg_calibrator.joblib
calibration/ef_calibrator.joblib
ensemble_weights.json
mace_artifacts/bg_head.joblib
mace_artifacts/bg_scaler.joblib
mace_artifacts/ref_energies.json
```
