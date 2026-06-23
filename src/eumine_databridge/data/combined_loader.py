"""
Combined train+val dataset loader for final retraining.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from pymatgen.core import Structure
from sklearn.model_selection import StratifiedKFold

from eumine_databridge.data.loader import BridgeDataset, MaterialEntry


class CombinedDataset:
    """
    Merges train and val BridgeDataset splits into one.

    Usage
    -----
    combined = CombinedDataset(train_ds, val_ds)
    structures = combined.get_structures()
    ef, ids = combined.get_targets("formation_energy_per_atom")
    """

    def __init__(
        self,
        train_ds: BridgeDataset,
        val_ds: BridgeDataset,
    ):
        self.entries: List[MaterialEntry] = train_ds.entries + val_ds.entries
        self.split = "combined"
        print(
            f"CombinedDataset: {len(train_ds)} train + {len(val_ds)} val "
            f"= {len(self.entries)} total"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def get_structures(self) -> List[Structure]:
        return [e.structure for e in self.entries]

    def get_targets(
        self, property_name: str
    ) -> Tuple[List[float], List[str]]:
        values, ids = [], []
        for e in self.entries:
            val = getattr(e, property_name)
            if val is not None:
                values.append(val)
                ids.append(e.material_id)
        return values, ids

    def get_target_array(self, property_name: str) -> np.ndarray:
        """Targets aligned with ``entries`` order (one value per entry)."""
        out = []
        for e in self.entries:
            val = getattr(e, property_name)
            if val is None:
                raise ValueError(
                    f"Missing {property_name} for {e.material_id} in combined set"
                )
            out.append(float(val))
        return np.array(out, dtype=float)

    def get_material_ids(self) -> List[str]:
        return [e.material_id for e in self.entries]

    def get_cv_folds(
        self,
        n_folds: int = 5,
        seed: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Stratified CV by band-gap category (metal / semiconductor / wide-gap).

        Returns list of (train_indices, val_indices) per fold.
        """
        strata = []
        for e in self.entries:
            bg = e.band_gap
            if bg is None or bg == 0.0:
                strata.append(0)
            elif bg < 3.0:
                strata.append(1)
            else:
                strata.append(2)

        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        indices = np.arange(len(self.entries))
        folds = list(skf.split(indices, strata))

        print(f"\n{n_folds}-fold CV split (stratified by BG category):")
        strata_arr = np.array(strata)
        for i, (tr_idx, val_idx) in enumerate(folds):
            tr_s = strata_arr[tr_idx]
            vl_s = strata_arr[val_idx]
            print(
                f"  Fold {i + 1}: train={len(tr_idx)} "
                f"(M:{(tr_s == 0).sum()} S:{(tr_s == 1).sum()} "
                f"W:{(tr_s == 2).sum()}) | "
                f"val={len(val_idx)} "
                f"(M:{(vl_s == 0).sum()} S:{(vl_s == 1).sum()} "
                f"W:{(vl_s == 2).sum()})"
            )
        return folds
