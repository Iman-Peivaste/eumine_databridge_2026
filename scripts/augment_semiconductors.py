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


# ── Layered / 2D materials fetcher ───────────────────────────────────────────

def fetch_layered_materials(
    db=None,
    max_structures: int = 500,
    out_dir: Path = None,
) -> pd.DataFrame:
    """
    Fetch JARVIS dft_3d entries that are layered or 2D materials.

    Criteria (OR on the first two, AND with the rest):
      - exfol_en < 100 meV/atom  (low exfoliation energy → layered)
      - OR dimensionality == "2D"
      - formation_energy_peratom in [-4, 1] eV/atom
      - optb88vdw_bandgap in [0, 6] eV

    Parameters
    ----------
    db             : pre-loaded JARVIS dft_3d list (reuse if already loaded)
    max_structures : cap on CIFs to save (keeps runtime bounded)
    out_dir        : directory for CIF files; defaults to
                     data/augmented/layered_structures/

    Returns
    -------
    DataFrame with material_id, formula, formation_energy_per_atom,
    band_gap, exfol_en, dimensionality, source columns.
    """
    from collections import Counter
    from jarvis.db.figshare import data as jdata
    from jarvis.core.atoms import Atoms as JAtoms

    if out_dir is None:
        out_dir = DATA / "augmented" / "layered_structures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if db is None:
        print("  Loading JARVIS dft_3d (cached after first run)...")
        db = jdata("dft_3d")
        print(f"  Total JARVIS entries: {len(db)}")

    adaptor = JarvisAtomsAdaptor()

    # ── Filter ────────────────────────────────────────────────────────────────
    records = []
    for entry in tqdm(db, desc="Filtering layered/2D entries"):
        ef  = entry.get("formation_energy_peratom")
        bg  = entry.get("optb88vdw_bandgap")
        exf = entry.get("exfoliation_energy")   # correct field name
        dim = str(entry.get("dimensionality", ""))

        # EF and BG must be valid numbers
        try:
            ef = float(ef)
            bg = float(bg)
        except (TypeError, ValueError):
            continue
        if not (-4.0 <= ef <= 1.0):
            continue
        if not (0.0 <= bg <= 6.0):
            continue

        # Layered criterion: low exfoliation energy OR dimensionality contains "2D"
        # (JARVIS stores dimensionality as e.g. "2D-bulk", "3D-bulk")
        is_layered = False
        if "2D" in dim:
            is_layered = True
        elif exf not in (None, "na", ""):
            try:
                if float(exf) < 100.0:
                    is_layered = True
            except (TypeError, ValueError):
                pass
        if not is_layered:
            continue

        records.append({
            "jid":    entry.get("jid", ""),
            "formula": entry.get("formula", ""),
            "formation_energy_per_atom": ef,
            "band_gap": bg,
            "exfol_en": exf if exf not in (None, "na", "") else None,
            "dimensionality": str(dim),
            "atoms": entry.get("atoms"),
        })

    print(f"\n  Layered/2D entries found: {len(records)}")
    if not records:
        print("  No entries matched. Check JARVIS field names.")
        return pd.DataFrame()

    # Deduplicate by formula — keep entry with lowest EF
    df_all = pd.DataFrame([{k: v for k, v in r.items() if k != "atoms"}
                            for r in records])
    keep_idx = df_all.groupby("formula")["formation_energy_per_atom"].idxmin()
    df_dedup = df_all.loc[keep_idx].reset_index(drop=True)
    records_by_jid = {r["jid"]: r for r in records}
    print(f"  After dedup by formula: {len(df_dedup)}")

    # Cap to max_structures (take lowest-EF entries — most stable)
    df_dedup = df_dedup.nsmallest(max_structures, "formation_energy_per_atom")

    # ── Save CIFs ─────────────────────────────────────────────────────────────
    saved = []
    for _, row in tqdm(df_dedup.iterrows(), total=len(df_dedup),
                       desc="Saving layered CIFs"):
        jid = row["jid"]
        rec = records_by_jid.get(jid)
        if rec is None or rec["atoms"] is None:
            continue
        try:
            atoms = JAtoms.from_dict(rec["atoms"])
            struct = adaptor.get_structure(atoms)
            cif_path = out_dir / f"{jid}.cif"
            struct.to(filename=str(cif_path))
            saved.append({
                "material_id":               jid,
                "formula":                   row["formula"],
                "formation_energy_per_atom": row["formation_energy_per_atom"],
                "band_gap":                  row["band_gap"],
                "exfol_en":                  row["exfol_en"],
                "dimensionality":            row["dimensionality"],
                "source":                    "JARVIS_layered",
            })
        except Exception:
            continue

    df_saved = pd.DataFrame(saved)
    if df_saved.empty:
        print("  No CIFs saved.")
        return df_saved

    csv_path = out_dir / "layered_dataset.csv"
    df_saved.to_csv(csv_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    bg = df_saved["band_gap"].values
    metals   = (bg == 0).sum()
    semis    = ((bg > 0) & (bg < 3)).sum()
    wide_gap = (bg >= 3).sum()

    # Element coverage
    all_elements: list = []
    for formula in df_saved["formula"]:
        try:
            from pymatgen.core import Composition
            comp = Composition(formula)
            all_elements.extend([str(el) for el in comp.elements])
        except Exception:
            pass
    top_elements = Counter(all_elements).most_common(20)

    print(f"\n{'='*55}")
    print(f"LAYERED / 2D MATERIAL FETCH — SUMMARY")
    print(f"{'='*55}")
    print(f"  Matched in JARVIS      : {len(records)}")
    print(f"  After dedup by formula : {len(df_dedup)}")
    print(f"  CIFs saved             : {len(df_saved)}  →  {out_dir}")
    print(f"\n  BG distribution:")
    print(f"    Metals   (BG = 0)    : {metals}")
    print(f"    Semis    (0 < BG < 3): {semis}")
    print(f"    Wide-gap (BG ≥ 3)    : {wide_gap}")
    print(f"    BG range             : {bg.min():.2f} – {bg.max():.2f} eV")
    print(f"\n  Top-20 elements:")
    for el, cnt in top_elements:
        bar = "█" * min(cnt // 2, 30)
        print(f"    {el:3s} {cnt:4d}  {bar}")
    print(f"{'='*55}")

    return df_saved


# ── Rare earth + complex oxide fetcher ───────────────────────────────────────

LANTHANIDES = {"La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}
ACTINIDES   = {"Th", "U"}
RARE_EARTH_ELEMENTS = LANTHANIDES | ACTINIDES


def fetch_rare_earth_materials(
    db=None,
    max_re: int = 300,
    max_oxide: int = 300,
    out_dir: Path = None,
) -> pd.DataFrame:
    """
    Fetch JARVIS dft_3d rare-earth and complex oxide entries.

    Two categories (independent caps):
      - Rare earth : formula contains any lanthanide or actinide (La–Lu, Th, U)
      - Complex oxide: 3+ distinct elements AND O present (and not rare earth)

    Filter conditions (both categories):
      - formation_energy_peratom in [-6, 2] eV/atom
      - optb88vdw_bandgap >= 0 and not "na"
      - Convertible to pymatgen Structure

    Parameters
    ----------
    db        : pre-loaded JARVIS dft_3d list (reused if provided)
    max_re    : cap on rare-earth CIFs saved
    max_oxide : cap on complex-oxide CIFs saved
    out_dir   : defaults to data/augmented/rare_earth_structures/

    Returns
    -------
    DataFrame with all saved entries.
    """
    from collections import Counter
    from jarvis.db.figshare import data as jdata
    from jarvis.core.atoms import Atoms as JAtoms

    if out_dir is None:
        out_dir = DATA / "augmented" / "rare_earth_structures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if db is None:
        print("  Loading JARVIS dft_3d...")
        db = jdata("dft_3d")
        print(f"  Total JARVIS entries: {len(db)}")

    adaptor = JarvisAtomsAdaptor()

    re_records    = []
    oxide_records = []

    for entry in tqdm(db, desc="Scanning rare-earth / complex-oxide entries"):
        ef     = entry.get("formation_energy_peratom")
        bg     = entry.get("optb88vdw_bandgap")
        formula = entry.get("formula", "")
        jid    = entry.get("jid", "")

        # EF must be valid
        try:
            ef = float(ef)
        except (TypeError, ValueError):
            continue
        if not (-6.0 <= ef <= 2.0):
            continue

        # BG must be valid and non-negative
        if bg in (None, "na", ""):
            continue
        try:
            bg = float(bg)
        except (TypeError, ValueError):
            continue
        if bg < 0:
            continue

        # Parse element set from formula
        try:
            from pymatgen.core import Composition
            comp   = Composition(formula)
            elems  = {str(el) for el in comp.elements}
            n_elems = len(elems)
        except Exception:
            continue

        is_re    = bool(elems & RARE_EARTH_ELEMENTS)
        is_oxide = (not is_re) and ("O" in elems) and (n_elems >= 3)

        if not (is_re or is_oxide):
            continue

        rec = {
            "jid":     jid,
            "formula": formula,
            "formation_energy_per_atom": ef,
            "band_gap": bg,
            "atoms":   entry.get("atoms"),
        }
        if is_re:
            re_records.append(rec)
        else:
            oxide_records.append(rec)

    print(f"\n  Rare-earth entries found : {len(re_records)}")
    print(f"  Complex-oxide entries   : {len(oxide_records)}")

    def _dedup_and_cap(records, cap, label):
        if not records:
            return []
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "atoms"}
                           for r in records])
        keep = df.groupby("formula")["formation_energy_per_atom"].idxmin()
        df   = df.loc[keep].reset_index(drop=True)
        df   = df.nsmallest(cap, "formation_energy_per_atom")
        print(f"  {label}: {len(records)} → {len(df)} after dedup+cap")
        return df, {r["jid"]: r for r in records}

    re_df,    re_by_jid    = _dedup_and_cap(re_records,    max_re,    "Rare-earth")
    oxide_df, oxide_by_jid = _dedup_and_cap(oxide_records, max_oxide, "Complex-oxide")

    def _save_cifs(df, by_jid, source_label):
        saved = []
        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"Saving {source_label} CIFs"):
            rec = by_jid.get(row["jid"])
            if rec is None or rec["atoms"] is None:
                continue
            try:
                atoms  = JAtoms.from_dict(rec["atoms"])
                struct = adaptor.get_structure(atoms)
                cif_path = out_dir / f"{row['jid']}.cif"
                struct.to(filename=str(cif_path))
                saved.append({
                    "material_id":               row["jid"],
                    "formula":                   row["formula"],
                    "formation_energy_per_atom": row["formation_energy_per_atom"],
                    "band_gap":                  row["band_gap"],
                    "source":                    source_label,
                })
            except Exception:
                continue
        return saved

    re_saved    = _save_cifs(re_df,    re_by_jid,    "JARVIS_rare_earth")
    oxide_saved = _save_cifs(oxide_df, oxide_by_jid, "JARVIS_complex_oxide")

    all_saved = re_saved + oxide_saved
    if not all_saved:
        print("  No CIFs saved.")
        return pd.DataFrame()

    df_out = pd.DataFrame(all_saved)
    df_out.to_csv(out_dir / "rare_earth_dataset.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    bg_arr   = df_out["band_gap"].values
    metals   = (bg_arr == 0).sum()
    semis    = ((bg_arr > 0) & (bg_arr < 3)).sum()
    wide_gap = (bg_arr >= 3).sum()

    all_elems: list = []
    for formula in df_out["formula"]:
        try:
            from pymatgen.core import Composition
            all_elems.extend(str(el) for el in Composition(formula).elements)
        except Exception:
            pass
    top_elems = Counter(all_elems).most_common(15)

    re_ex    = df_out[df_out["source"]=="JARVIS_rare_earth"]["formula"].head(5).tolist()
    ox_ex    = df_out[df_out["source"]=="JARVIS_complex_oxide"]["formula"].head(5).tolist()

    print(f"\n{'='*55}")
    print(f"RARE EARTH + COMPLEX OXIDE FETCH — SUMMARY")
    print(f"{'='*55}")
    print(f"  Total JARVIS entries scanned : {len(db)}")
    print(f"  Rare-earth structures saved  : {len(re_saved)}")
    print(f"  Complex-oxide structures saved: {len(oxide_saved)}")
    print(f"  Total CIFs saved             : {len(df_out)}  →  {out_dir}")
    print(f"\n  BG distribution:")
    print(f"    Metals   (BG = 0)    : {metals}")
    print(f"    Semis    (0 < BG < 3): {semis}")
    print(f"    Wide-gap (BG ≥ 3)    : {wide_gap}")
    print(f"    BG range             : {bg_arr.min():.2f} – {bg_arr.max():.2f} eV")
    print(f"\n  Top-15 elements:")
    for el, cnt in top_elems:
        bar = "█" * min(cnt // 3, 30)
        print(f"    {el:3s} {cnt:4d}  {bar}")
    print(f"\n  Example rare-earth formulas  : {re_ex}")
    print(f"  Example complex-oxide formulas: {ox_ex}")
    print(f"{'='*55}")

    return df_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layered_only",
        action="store_true",
        help="Run only fetch_layered_materials() (skip semiconductor fetch)",
    )
    parser.add_argument(
        "--rare_earth_only",
        action="store_true",
        help="Run only fetch_rare_earth_materials() (skip semiconductor fetch)",
    )
    args = parser.parse_args()

    if args.layered_only:
        print("\n[Layered fetch only — loading JARVIS db...]")
        from jarvis.db.figshare import data as jdata
        _db = jdata("dft_3d")
        fetch_layered_materials(db=_db)
    elif args.rare_earth_only:
        print("\n[Rare earth fetch only — loading JARVIS db...]")
        from jarvis.db.figshare import data as jdata
        _db = jdata("dft_3d")
        fetch_rare_earth_materials(db=_db)
    else:
        print("\n[Running all fetchers on already-loaded db...]")
        fetch_layered_materials(db=db)
        fetch_rare_earth_materials(db=db)
