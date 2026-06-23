"""
AFLOW fetcher via REST API.
AFLOW uses multiple functionals — we filter for PBE entries.
No dedicated client needed: uses requests against aflow.org REST endpoint.
"""

from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm


class AFLOWFetcher:
    """
    Fetch formation enthalpy from AFLOW database via REST API.

    Note: AFLOW does not provide band gap as systematically as MP/JARVIS.
    Primary use here is for formation energy augmentation and structural
    diversity (3.5M entries, widest chemical space coverage).

    Usage
    -----
    fetcher = AFLOWFetcher()
    data = fetcher.fetch_by_formula(["TiO2", "ZnO"])
    """

    SOURCE_NAME = "AFLOW"
    FUNCTIONAL = "PBE (various)"
    BASE_URL = "http://aflow.org/API/aflowlib.py"

    def fetch_by_formula(
        self,
        formulas: List[str],
        max_per_formula: int = 5,
    ) -> pd.DataFrame:
        """
        Fetch AFLOW entries for given formulas.
        Returns best-matching entries (lowest enthalpy per formula).
        """
        results = []
        for formula in tqdm(formulas, desc="Fetching from AFLOW"):
            try:
                params = {
                    "species": formula,
                    "format": "json",
                    "paging": 0,
                    "catalog": "icsd",
                }
                resp = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    entries = data if isinstance(data, list) else [data]
                    for entry in entries[:max_per_formula]:
                        ef = entry.get("enthalpy_formation_atom")
                        results.append({
                            "auid": entry.get("auid", ""),
                            "formula": formula,
                            "aflow_formation_energy": (
                                float(ef) if ef is not None else None
                            ),
                            "compound": entry.get("compound", ""),
                            "functional": self.FUNCTIONAL,
                            "source": self.SOURCE_NAME,
                        })
                time.sleep(0.05)
            except Exception as e:
                print(f"  WARNING: AFLOW query failed for {formula}: {e}")
                continue

        df = pd.DataFrame(results) if results else pd.DataFrame()
        print(f"AFLOW: fetched {len(df)} entries for {len(formulas)} formulas")
        return df

    def fetch_bulk_REST(
        self,
        elements: List[str],
        max_entries: int = 10000,
    ) -> pd.DataFrame:
        """
        Bulk fetch from AFLOW for given element list.
        Uses the AFLOWLIB REST interface.
        """
        results = []
        url = (
            f"http://aflow.org/API/aflowlib.py?"
            f"species={'|'.join(elements)}&format=json&paging=0"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                entries = data if isinstance(data, list) else [data]
                for entry in entries[:max_entries]:
                    ef = entry.get("enthalpy_formation_atom")
                    results.append({
                        "auid": entry.get("auid", ""),
                        "formula": entry.get("compound", ""),
                        "aflow_formation_energy": (
                            float(ef) if ef is not None else None
                        ),
                        "functional": self.FUNCTIONAL,
                        "source": self.SOURCE_NAME,
                    })
        except Exception as e:
            print(f"  WARNING: AFLOW bulk fetch failed: {e}")

        df = pd.DataFrame(results) if results else pd.DataFrame()
        print(f"AFLOW bulk: {len(df)} entries fetched")
        return df
