"""
OQMD fetcher via REST API (qmpy_rester).
OQMD uses PBE — directly comparable to Materials Project.
~1M entries, excellent coverage for binary oxides and common compounds.
"""

from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm


class OQMDFetcher:
    """
    Fetch formation energy from OQMD via REST API.

    OQMD uses PBE — the same functional as Materials Project.
    Therefore OQMD ↔ MP discrepancies reflect convergence criteria
    and pseudopotential differences rather than functional differences.
    This makes it useful as a secondary validation source for EF.

    Usage
    -----
    fetcher = OQMDFetcher()
    data = fetcher.fetch_by_formula(["TiO2", "ZnO"])
    """

    SOURCE_NAME = "OQMD"
    FUNCTIONAL = "PBE"
    BASE_URL = "https://oqmd.org/oqmdapi/formationenergy"

    def fetch_by_formula(
        self,
        formulas: List[str],
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """Fetch OQMD entries for a list of reduced formulas."""
        results = []
        for formula in tqdm(formulas, desc="Fetching from OQMD"):
            try:
                params = {
                    "composition": formula,
                    "fields": "name,delta_e,band_gap,entry_id",
                    "format": "json",
                    "limit": 5,
                    "offset": 0,
                }
                resp = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    entries = data.get("data", [])
                    for entry in entries:
                        delta_e = entry.get("delta_e")
                        results.append({
                            "oqmd_id": entry.get("entry_id", ""),
                            "formula": formula,
                            "oqmd_formation_energy": (
                                float(delta_e)
                                if delta_e is not None
                                else None
                            ),
                            "functional": self.FUNCTIONAL,
                            "source": self.SOURCE_NAME,
                        })
                time.sleep(0.1)
            except Exception as e:
                print(f"  WARNING: OQMD query failed for {formula}: {e}")
                continue

        df = pd.DataFrame(results) if results else pd.DataFrame()
        print(f"OQMD: fetched {len(df)} entries for {len(formulas)} formulas")
        return df
