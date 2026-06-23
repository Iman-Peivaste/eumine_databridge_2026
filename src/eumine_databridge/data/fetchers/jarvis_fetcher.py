"""
JARVIS-DFT fetcher.
Uses jarvis-tools to access JARVIS-DFT database.
Key functional: OptB88vdW (good for vdW systems, systematic offset vs PBE).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


class JARVISFetcher:
    """
    Fetch formation energy and band gap from JARVIS-DFT.

    JARVIS uses OptB88vdW functional — systematically different from
    Materials Project PBE. This offset is a core challenge of the hackathon.

    Usage
    -----
    fetcher = JARVISFetcher()
    fetcher.load_database()            # downloads ~200MB once, cached
    data = fetcher.fetch_by_formula(["TiO2", "ZnO"])
    """

    SOURCE_NAME = "JARVIS-DFT"
    FUNCTIONAL = "OptB88vdW"

    def __init__(self):
        self._db = None

    def load_database(self):
        """Load the full JARVIS-DFT database into memory (cached locally)."""
        from jarvis.db.figshare import data as jdata

        print("Loading JARVIS-DFT database (may download ~200MB on first run)...")
        self._db = jdata("dft_3d")
        print(f"  JARVIS-DFT loaded: {len(self._db)} entries")

    def _ensure_loaded(self):
        if self._db is None:
            self.load_database()

    def fetch_by_jid(self, jids: List[str]) -> pd.DataFrame:
        """Fetch entries by JARVIS ID (JVASP-XXXX format)."""
        self._ensure_loaded()
        jid_set = set(jids)
        results = []
        for entry in tqdm(self._db, desc="Searching JARVIS by JID"):
            if entry.get("jid") in jid_set:
                results.append(self._parse_entry(entry))
        df = pd.DataFrame(results)
        print(f"JARVIS: found {len(df)} / {len(jids)} requested JIDs")
        return df

    def fetch_by_formula(self, formulas: List[str]) -> pd.DataFrame:
        """Fetch entries matching reduced formulas."""
        self._ensure_loaded()
        formula_set = set(formulas)
        results = []
        for entry in tqdm(self._db, desc="Searching JARVIS by formula"):
            formula = entry.get("formula", "")
            if formula in formula_set:
                results.append(self._parse_entry(entry))
        df = pd.DataFrame(results)
        print(f"JARVIS: found {len(df)} entries for {len(formulas)} formulas")
        return df

    def fetch_all(self, max_entries: Optional[int] = None) -> pd.DataFrame:
        """Fetch all JARVIS-DFT entries (for bulk training augmentation)."""
        self._ensure_loaded()
        db = self._db[:max_entries] if max_entries else self._db
        results = [self._parse_entry(e) for e in tqdm(db, desc="Parsing JARVIS")]
        df = pd.DataFrame(results)
        # Drop entries with missing key properties
        df = df.dropna(subset=["jarvis_formation_energy"])
        print(f"JARVIS full fetch: {len(df)} valid entries")
        return df

    def _parse_entry(self, entry: dict) -> dict:
        ef = entry.get("formation_energy_peratom")
        bg = entry.get("optb88vdw_bandgap")
        # JARVIS uses 'na' string for missing values
        return {
            "jid": entry.get("jid", ""),
            "formula": entry.get("formula", ""),
            "jarvis_formation_energy": (
                float(ef) if ef not in (None, "na", "") else None
            ),
            "jarvis_band_gap": (
                float(bg) if bg not in (None, "na", "") else None
            ),
            "functional": self.FUNCTIONAL,
            "source": self.SOURCE_NAME,
            "spacegroup": entry.get("spg_symbol", ""),
        }
