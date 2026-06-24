"""
Augment training set with semiconductor structures from JARVIS and MP.

Strategy:
- Semiconductors (0 < BG < 3 eV) are underrepresented and have
  the highest error. We fetch additional structures in this range
  from JARVIS dft_3d and Materials Project.
- Target: add 500-800 semiconductor structures to training.
- These are used as AUXILIARY training data alongside Bridge labels.

Run:
    python scripts/augment_semiconductors.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

import os
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor

DATA = ROOT / "data"
AUG = DATA / "augmented"
AUG.mkdir(parents=True, exist_ok=True)
(AUG / "structures").mkdir(exist_ok=True)

# ── 1. Fetch from JARVIS ──────────────────────────────────────────────────────
print("[1] Loading JARVIS dft_3d...")
from jarvis.db.figshare import data as jdata
db = jdata("dft_3d")
print(f"  Total JARVIS entries: {len(db)}")

# Filter for semiconductors with valid BG and EF
records = []
adaptor = JarvisAtomsAdaptor()

for entry in tqdm(db, desc="Filtering JARVIS semiconductors"):
    ef = entry.get("formation_energy_peratom")
    bg = entry.get("optb88vdw_bandgap")

    # Skip missing or invalid
    if ef in (None, "na", "") or bg in (None, "na", ""):
        continue
    try:
        ef = float(ef)
        bg = float(bg)
    except (ValueError, TypeError):
        continue

    # Target: semiconductors (0.1 < BG < 3.5 eV) with stable EF
    if not (0.1 < bg < 3.5):
        continue
    if not (-5.0 < ef < 2.0):
        continue

    records.append({
        "jid": entry.get("jid", ""),
        "formula": entry.get("formula", ""),
        "formation_energy_per_atom": ef,
        "band_gap": bg,
        "source": "JARVIS",
        "functional": "OptB88vdW",
        "atoms": entry.get("atoms"),
    })

print(f"  JARVIS semiconductors found: {len(records)}")

# Deduplicate by formula — keep lowest EF per formula
df_jarvis = pd.DataFrame([{k: v for k, v in r.items() if k != 'atoms'}
                           for r in records])
df_jarvis_dedup = df_jarvis.loc[
    df_jarvis.groupby('formula')['formation_energy_per_atom'].idxmin()
].reset_index(drop=True)
print(f"  After dedup by formula: {len(df_jarvis_dedup)}")

# Save CIF files for deduplicated entries
print("  Converting to CIF...")
records_by_jid = {r['jid']: r for r in records}
saved = []

for _, row in tqdm(df_jarvis_dedup.iterrows(), total=len(df_jarvis_dedup)):
    jid = row['jid']
    record = records_by_jid.get(jid)
    if record is None or record['atoms'] is None:
        continue
    try:
        from jarvis.core.atoms import Atoms as JAtoms
        atoms = JAtoms.from_dict(record['atoms'])
        pmg_struct = adaptor.get_structure(atoms)
        cif_path = AUG / "structures" / f"{jid}.cif"
        pmg_struct.to(filename=str(cif_path))
        saved.append({
            "material_id": jid,
            "formula": row['formula'],
            "formation_energy_per_atom": row['formation_energy_per_atom'],
            "band_gap": row['band_gap'],
            "source": "JARVIS",
        })
    except Exception as e:
        continue

print(f"  CIFs saved: {len(saved)}")

# ── 2. Fetch from MP ──────────────────────────────────────────────────────────
print("\n[2] Fetching MP semiconductors...")
from mp_api.client import MPRester

mp_records = []
with MPRester(os.getenv("MP_API_KEY")) as mpr:
    docs = mpr.summary.search(
        band_gap=(0.1, 3.5),
        formation_energy_per_atom=(-5.0, 2.0),
        fields=[
            "material_id",
            "formula_pretty",
            "formation_energy_per_atom",
            "band_gap",
            "structure",
        ],
    )
    print(f"  MP semiconductor entries found: {len(docs)}")

    # Sample up to 800 to avoid overlap with Bridge Dataset
    import random
    random.seed(42)
    sample = random.sample(docs, min(800, len(docs)))

    for doc in tqdm(sample, desc="Saving MP CIFs"):
        try:
            mat_id = str(doc.material_id)
            cif_path = AUG / "structures" / f"{mat_id}.cif"
            doc.structure.to(filename=str(cif_path))
            mp_records.append({
                "material_id": mat_id,
                "formula": doc.formula_pretty,
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "band_gap": doc.band_gap,
                "source": "MP",
            })
        except Exception as e:
            continue

print(f"  MP CIFs saved: {len(mp_records)}")

# ── 3. Combine and save augmentation CSV ─────────────────────────────────────
print("\n[3] Saving augmentation dataset...")
all_records = saved + mp_records

# Remove any IDs that overlap with Bridge Dataset train+val
bridge_ids = set()
for csv_path in [
    DATA / "raw" / "bridge_dataset_train.csv",
    DATA / "raw" / "bridge_dataset_val.csv",
]:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        bridge_ids.update(df['material_id'].astype(str).tolist())

all_records = [r for r in all_records
               if r['material_id'] not in bridge_ids]

df_aug = pd.DataFrame(all_records)
df_aug.to_csv(AUG / "augmentation_dataset.csv", index=False)

print(f"\nAugmentation dataset summary:")
print(f"  Total structures  : {len(df_aug)}")
print(f"  From JARVIS       : {(df_aug['source']=='JARVIS').sum()}")
print(f"  From MP           : {(df_aug['source']=='MP').sum()}")
print(f"  BG range          : {df_aug['band_gap'].min():.2f} – {df_aug['band_gap'].max():.2f} eV")
print(f"  EF range          : {df_aug['formation_energy_per_atom'].min():.2f} – {df_aug['formation_energy_per_atom'].max():.2f} eV/atom")
print(f"\nSaved to: {AUG}")
