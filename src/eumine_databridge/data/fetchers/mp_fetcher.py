"""
Materials Project fetcher.
Queries mp-api for formation energy and band gap
for a list of material IDs or formulas.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from tqdm import tqdm

load_dotenv()


class MPFetcher:
    """
    Fetch formation energy and band gap from Materials Project.

    Usage
    -----
    fetcher = MPFetcher()
    data = fetcher.fetch_by_ids(["mp-1234", "mp-5678"])
    """

    SOURCE_NAME = "Materials Project"
    FUNCTIONAL = "PBE/PBE+U"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MP_API_KEY not found. Set it in .env or pass explicitly."
            )

    def fetch_by_ids(
        self,
        material_ids: List[str],
        batch_size: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch properties for a list of mp-XXXX IDs.

        Returns
        -------
        DataFrame with columns:
            material_id, formula, formation_energy_per_atom,
            band_gap, functional, source
        """
        results = []
        batches = [
            material_ids[i:i + batch_size]
            for i in range(0, len(material_ids), batch_size)
        ]

        with MPRester(self.api_key) as mpr:
            for batch in tqdm(batches, desc="Fetching from Materials Project"):
                try:
                    docs = mpr.summary.search(
                        material_ids=batch,
                        fields=[
                            "material_id",
                            "formula_pretty",
                            "formation_energy_per_atom",
                            "band_gap",
                        ],
                    )
                    for doc in docs:
                        results.append({
                            "material_id": str(doc.material_id),
                            "formula": doc.formula_pretty,
                            "mp_formation_energy": doc.formation_energy_per_atom,
                            "mp_band_gap": doc.band_gap,
                            "functional": self.FUNCTIONAL,
                            "source": self.SOURCE_NAME,
                        })
                except Exception as e:
                    print(f"  WARNING: batch failed — {e}")
                time.sleep(0.1)  # polite rate limiting

        df = pd.DataFrame(results)
        print(
            f"Materials Project: fetched {len(df)} records "
            f"for {len(material_ids)} requested IDs"
        )
        return df

    def fetch_bulk_for_elements(
        self,
        elements: List[str],
        max_sites: int = 20,
    ) -> pd.DataFrame:
        """
        Fetch all binary/ternary compounds containing given elements.
        Useful for augmenting the Bridge Dataset with additional training data.
        """
        results = []
        with MPRester(self.api_key) as mpr:
            docs = mpr.summary.search(
                elements=elements,
                nsites=(1, max_sites),
                fields=[
                    "material_id",
                    "formula_pretty",
                    "formation_energy_per_atom",
                    "band_gap",
                    "nelements",
                ],
            )
            for doc in tqdm(docs, desc=f"Fetching MP bulk ({elements})"):
                results.append({
                    "material_id": str(doc.material_id),
                    "formula": doc.formula_pretty,
                    "mp_formation_energy": doc.formation_energy_per_atom,
                    "mp_band_gap": doc.band_gap,
                    "nelements": doc.nelements,
                    "functional": self.FUNCTIONAL,
                    "source": self.SOURCE_NAME,
                })

        df = pd.DataFrame(results)
        print(f"Materials Project bulk: {len(df)} records fetched")
        return df
