"""
Bridge Dataset loader.
Loads CSV labels + CIF structures into a unified BridgeDataset object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm

# Bridge Dataset CSV column aliases → internal field names
_CSV_COLUMN_MAP = {
    "formation_energy_per_atom_label": "formation_energy_per_atom",
    "band_gap_label": "band_gap",
    "formation_energy_per_atom_mp": "mp_formation_energy",
    "band_gap_mp": "mp_band_gap",
    "formation_energy_per_atom_jarvis": "jarvis_formation_energy",
    "band_gap_jarvis": "jarvis_band_gap",
}


@dataclass
class MaterialEntry:
    """Single material with all available data."""

    material_id: str
    structure: Structure
    # Primary targets (may be None for test set)
    formation_energy_per_atom: Optional[float] = None
    band_gap: Optional[float] = None
    # Per-database values (populated by fetchers)
    mp_formation_energy: Optional[float] = None
    mp_band_gap: Optional[float] = None
    jarvis_formation_energy: Optional[float] = None
    jarvis_band_gap: Optional[float] = None
    aflow_formation_energy: Optional[float] = None
    oqmd_formation_energy: Optional[float] = None
    # Metadata
    formula: str = ""
    nsites: int = 0
    spacegroup: str = ""
    crystal_system: str = ""
    data_sources: List[str] = field(default_factory=list)


class BridgeDataset:
    """
    Container for the EuMINe Bridge Dataset split.

    Parameters
    ----------
    csv_path : path to the CSV file with labels
    structures_dir : path to the directory containing CIF files
    split : 'train', 'val', or 'test'
    """

    def __init__(
        self,
        csv_path: Path,
        structures_dir: Path,
        split: str = "train",
    ):
        self.split = split
        self.csv_path = Path(csv_path)
        self.structures_dir = Path(structures_dir)
        self.entries: List[MaterialEntry] = []
        self._load()

    def _load(self):
        # Load CSV
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            df = df.rename(columns={
                k: v for k, v in _CSV_COLUMN_MAP.items() if k in df.columns
            })
        else:
            # Test set has no labels CSV — build from CIF directory
            df = pd.DataFrame({
                "material_id": [
                    f.stem for f in sorted(self.structures_dir.glob("*.cif"))
                ]
            })

        print(f"\nLoading {self.split} split — {len(df)} entries")
        missing_cif = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {self.split}"):
            mat_id = str(row["material_id"])
            cif_path = self.structures_dir / f"{mat_id}.cif"

            if not cif_path.exists():
                missing_cif += 1
                continue

            try:
                structure = Structure.from_file(str(cif_path))
            except Exception as e:
                print(f"  WARNING: could not parse {cif_path.name}: {e}")
                continue

            formula = (
                str(row["formula"])
                if "formula" in row and pd.notna(row.get("formula"))
                else structure.composition.reduced_formula
            )
            spacegroup = (
                str(row["spacegroup_symbol"])
                if "spacegroup_symbol" in row and pd.notna(row.get("spacegroup_symbol"))
                else structure.get_space_group_info()[0]
            )
            entry = MaterialEntry(
                material_id=mat_id,
                structure=structure,
                formula=formula,
                nsites=int(row["nsites"]) if "nsites" in row and pd.notna(row.get("nsites")) else len(structure),
                spacegroup=spacegroup,
                crystal_system=spacegroup,
            )

            # Attach labels if available
            if "formation_energy_per_atom" in row:
                entry.formation_energy_per_atom = (
                    float(row["formation_energy_per_atom"])
                    if pd.notna(row.get("formation_energy_per_atom"))
                    else None
                )
            if "band_gap" in row:
                entry.band_gap = (
                    float(row["band_gap"])
                    if pd.notna(row.get("band_gap"))
                    else None
                )

            # Attach per-database values if columns exist
            for col in [
                "mp_formation_energy", "mp_band_gap",
                "jarvis_formation_energy", "jarvis_band_gap",
            ]:
                if col in row and pd.notna(row.get(col)):
                    setattr(entry, col, float(row[col]))

            self.entries.append(entry)

        print(f"  Loaded   : {len(self.entries)} entries")
        if missing_cif > 0:
            print(f"  Missing CIFs : {missing_cif}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> MaterialEntry:
        return self.entries[idx]

    def to_dataframe(self) -> pd.DataFrame:
        """Export entries to a flat DataFrame for analysis."""
        records = []
        for e in self.entries:
            records.append({
                "material_id": e.material_id,
                "formula": e.formula,
                "nsites": e.nsites,
                "spacegroup": e.spacegroup,
                "formation_energy_per_atom": e.formation_energy_per_atom,
                "band_gap": e.band_gap,
                "mp_formation_energy": e.mp_formation_energy,
                "mp_band_gap": e.mp_band_gap,
                "jarvis_formation_energy": e.jarvis_formation_energy,
                "jarvis_band_gap": e.jarvis_band_gap,
            })
        return pd.DataFrame(records)

    def get_structures(self) -> List[Structure]:
        return [e.structure for e in self.entries]

    def get_targets(
        self, property_name: str
    ) -> Tuple[List[float], List[str]]:
        """
        Returns (values, material_ids) for entries where property is not None.
        property_name: 'formation_energy_per_atom' or 'band_gap'
        """
        values, ids = [], []
        for e in self.entries:
            val = getattr(e, property_name)
            if val is not None:
                values.append(val)
                ids.append(e.material_id)
        return values, ids

    def summary(self) -> str:
        df = self.to_dataframe()
        lines = [
            f"\n{'='*50}",
            f"BridgeDataset — {self.split.upper()} split",
            f"{'='*50}",
            f"Total entries         : {len(self.entries)}",
            f"With EF labels        : {df['formation_energy_per_atom'].notna().sum()}",
            f"With BG labels        : {df['band_gap'].notna().sum()}",
            f"Unique formulas       : {df['formula'].nunique()}",
            f"N sites range         : {df['nsites'].min()} – {df['nsites'].max()}",
        ]
        if df['formation_energy_per_atom'].notna().any():
            ef = df['formation_energy_per_atom'].dropna()
            lines += [
                f"\nFormation energy (eV/atom):",
                f"  mean={ef.mean():.3f}  std={ef.std():.3f}",
                f"  min={ef.min():.3f}   max={ef.max():.3f}",
            ]
        if df['band_gap'].notna().any():
            bg = df['band_gap'].dropna()
            lines += [
                f"\nBand gap (eV):",
                f"  mean={bg.mean():.3f}  std={bg.std():.3f}",
                f"  min={bg.min():.3f}   max={bg.max():.3f}",
                f"  metals (BG=0)     : {(bg == 0).sum()}",
                f"  semiconductors    : {((bg > 0) & (bg < 3)).sum()}",
                f"  wide-gap (BG>3)   : {(bg >= 3).sum()}",
            ]
        lines.append("="*50)
        return "\n".join(lines)
