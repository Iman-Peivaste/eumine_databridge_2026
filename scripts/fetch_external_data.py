"""
Fetch external data from MP, JARVIS, AFLOW, OQMD for Bridge Dataset materials.
Run: python scripts/fetch_external_data.py

This script:
1. Loads the Bridge Dataset train+val CSVs
2. Fetches MP and JARVIS data for all material IDs/formulas
3. Fits the harmonizer on the overlap
4. Saves processed data to data/processed/
5. Prints a full harmonization report
"""

import os
import sys
from pathlib import Path

# Make sure src/ is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pandas as pd
from eumine_databridge.data.loader import BridgeDataset
from eumine_databridge.data.fetchers.mp_fetcher import MPFetcher
from eumine_databridge.data.fetchers.jarvis_fetcher import JARVISFetcher
from eumine_databridge.data.fetchers.aflow_fetcher import AFLOWFetcher
from eumine_databridge.data.fetchers.oqmd_fetcher import OQMDFetcher
from eumine_databridge.data.harmonizer import DatabaseHarmonizer

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── 1. Load Bridge Dataset ──────────────────────────────────────────────────
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
test_ds = BridgeDataset(
    csv_path=RAW / "bridge_dataset_test.csv",  # no labels — built from CIFs
    structures_dir=RAW / "test_structures",
    split="test",
)

print(train_ds.summary())
print(val_ds.summary())

train_df = train_ds.to_dataframe()
val_df = val_ds.to_dataframe()

# ── 2. Fetch from Materials Project ────────────────────────────────────────
print("\n[2] Fetching from Materials Project...")
mp_fetcher = MPFetcher()

# Extract mp-XXXX IDs from material_id column
all_ids = pd.concat([train_df, val_df])["material_id"].tolist()
mp_ids = [m for m in all_ids if str(m).startswith("mp-")]
mp_data = mp_fetcher.fetch_by_ids(mp_ids)
mp_data.to_csv(PROCESSED / "mp_fetched.csv", index=False)
print(f"  Saved to data/processed/mp_fetched.csv")

# ── 3. Fetch from JARVIS ────────────────────────────────────────────────────
print("\n[3] Fetching from JARVIS-DFT...")
jarvis_fetcher = JARVISFetcher()
jarvis_fetcher.load_database()

# Use formulas from Bridge Dataset to match JARVIS entries
formulas = (
    pd.concat([train_df, val_df])["formula"]
    .dropna()
    .unique()
    .tolist()
)
jarvis_data = jarvis_fetcher.fetch_by_formula(formulas)
jarvis_data.to_csv(PROCESSED / "jarvis_fetched.csv", index=False)
print(f"  Saved to data/processed/jarvis_fetched.csv")

# ── 4. Merge MP + JARVIS on formula ────────────────────────────────────────
print("\n[4] Merging MP and JARVIS data...")

# Merge train data with fetched MP values (API overwrites CSV columns)
train_merged = train_df.drop(
    columns=[c for c in ("mp_formation_energy", "mp_band_gap") if c in train_df.columns],
    errors="ignore",
)
if not mp_data.empty:
    train_merged = train_merged.merge(
        mp_data[["material_id", "mp_formation_energy", "mp_band_gap"]],
        on="material_id",
        how="left",
    )

# Merge with JARVIS on formula
jarvis_by_formula = (
    jarvis_data.groupby("formula")
    .agg({
        "jarvis_formation_energy": "mean",
        "jarvis_band_gap": "mean",
    })
    .reset_index()
)
train_merged = train_merged.drop(
    columns=[c for c in ("jarvis_formation_energy", "jarvis_band_gap") if c in train_merged.columns],
    errors="ignore",
)
if not jarvis_by_formula.empty:
    train_merged = train_merged.merge(
        jarvis_by_formula,
        on="formula",
        how="left",
    )
train_merged.to_csv(PROCESSED / "train_merged.csv", index=False)
print(f"  Saved to data/processed/train_merged.csv")
print(f"  MP values  : {train_merged['mp_formation_energy'].notna().sum()} / {len(train_merged)}")
print(f"  JARVIS values: {train_merged['jarvis_formation_energy'].notna().sum()} / {len(train_merged)}")

# ── 5. Fit harmonizer ───────────────────────────────────────────────────────
print("\n[5] Fitting harmonizer on MP/JARVIS overlap...")
harmonizer = DatabaseHarmonizer()
harmonizer.fit(train_merged)
print(harmonizer.report())
harmonizer.save(PROCESSED / "harmonizer_params.json")

# ── 6. Apply harmonization ──────────────────────────────────────────────────
print("\n[6] Applying harmonization to training data...")
train_harmonized = harmonizer.harmonize_dataframe(train_merged)
train_harmonized.to_csv(PROCESSED / "train_harmonized.csv", index=False)

print(f"\nHarmonized training data source breakdown:")
print(f"  EF sources: {train_harmonized['ef_source'].value_counts().to_dict()}")
print(f"  BG sources: {train_harmonized['bg_source'].value_counts().to_dict()}")

# ── 7. Optional: AFLOW + OQMD for formulas with missing data ───────────────
missing_ef = train_harmonized[train_harmonized["harmonized_ef"].isna()]["formula"].tolist()
missing_bg = train_harmonized[train_harmonized["harmonized_bg"].isna()]["formula"].tolist()

if missing_ef or missing_bg:
    missing_formulas = list(set(missing_ef + missing_bg))
    print(f"\n[7] Fetching OQMD for {len(missing_formulas)} formulas with missing data...")
    oqmd_fetcher = OQMDFetcher()
    oqmd_data = oqmd_fetcher.fetch_by_formula(missing_formulas)
    oqmd_data.to_csv(PROCESSED / "oqmd_fetched.csv", index=False)
else:
    print("\n[7] No missing data — skipping OQMD fetch")

print("\n" + "="*50)
print("Data pipeline complete.")
print(f"Processed files saved to: {PROCESSED}")
print("="*50)
