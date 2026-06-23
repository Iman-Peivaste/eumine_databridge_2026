"""
MACE-MP-0 wrapper for property prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import joblib


class MACEPredictor:
    """
    MACE-MP-0 based predictor for formation energy and band gap.

    EF: total energy per atom with per-element reference subtraction (Ridge fit).
    BG: invariant node descriptors (mean-pooled) + gradient boosting head.
    """

    def __init__(
        self,
        model_name: str = "medium",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.calculator = None
        self._ref_energies: Optional[Dict[str, float]] = None
        self._element_idx: Optional[Dict[str, int]] = None
        self._all_elements: Optional[List[str]] = None
        self._bg_head: Optional[GradientBoostingRegressor] = None
        self._bg_scaler: Optional[StandardScaler] = None
        self._bg_head_trained = False
        print(f"MACEPredictor — device: {self.device}, model: {model_name}")

    def load_model(self, cache_dir: Optional[Path] = None):
        from mace.calculators import mace_mp

        print(f"Loading MACE-MP-0 ({self.model_name})...")
        self.calculator = mace_mp(
            model=self.model_name,
            device=self.device,
            default_dtype="float32",
        )
        print("  MACE-MP-0 loaded successfully")

    def _structure_to_ase(self, structure: Structure):
        return AseAtomsAdaptor().get_atoms(structure)

    def _get_energies_per_atom(
        self, structures: List[Structure]
    ) -> np.ndarray:
        assert self.calculator is not None, "Call load_model() first"
        energies = []
        for structure in tqdm(structures, desc="MACE energy inference"):
            try:
                atoms = self._structure_to_ase(structure)
                atoms.calc = self.calculator
                energy = atoms.get_potential_energy()
                energies.append(energy / len(atoms))
            except Exception as e:
                print(f"  WARNING: MACE energy failed: {e}")
                energies.append(np.nan)
        return np.array(energies, dtype=float)

    def fit_references(
        self,
        train_structures: List[Structure],
        train_ef_targets: List[float],
    ) -> "MACEPredictor":
        print("\nFitting per-element reference energies for MACE EF...")
        mace_energies = self._get_energies_per_atom(train_structures)

        all_elements = sorted({
            el.symbol
            for s in train_structures
            for el in s.composition.elements
        })
        element_idx = {el: i for i, el in enumerate(all_elements)}
        X = np.zeros((len(train_structures), len(all_elements)))
        for i, structure in enumerate(train_structures):
            comp = structure.composition.fractional_composition
            for el, frac in comp.items():
                X[i, element_idx[el.symbol]] = float(frac)

        valid_mask = ~np.isnan(mace_energies)
        y = mace_energies[valid_mask] - np.array(train_ef_targets)[valid_mask]
        reg = Ridge(alpha=1e-3, fit_intercept=False)
        reg.fit(X[valid_mask], y)

        self._ref_energies = {
            el: float(reg.coef_[i]) for el, i in element_idx.items()
        }
        self._element_idx = element_idx
        self._all_elements = all_elements

        ef_pred_train = mace_energies[valid_mask] - reg.predict(X[valid_mask])
        train_mae = float(np.mean(np.abs(
            ef_pred_train - np.array(train_ef_targets)[valid_mask]
        )))
        print(f"  Elements fitted: {len(all_elements)}")
        print(f"  Train MAE after reference fitting: {train_mae:.4f} eV/atom")
        return self

    def predict_ef(self, structures: List[Structure]) -> np.ndarray:
        assert self._ref_energies is not None, "Call fit_references() first"
        mace_energies = self._get_energies_per_atom(structures)
        predictions = []
        for i, structure in enumerate(structures):
            if np.isnan(mace_energies[i]):
                predictions.append(np.nan)
                continue
            comp = structure.composition.fractional_composition
            ref_sum = sum(
                float(frac) * self._ref_energies.get(el.symbol, 0.0)
                for el, frac in comp.items()
            )
            predictions.append(mace_energies[i] - ref_sum)
        return np.array(predictions, dtype=float)

    def get_embeddings(
        self, structures: List[Structure]
    ) -> np.ndarray:
        """Mean-pooled MACE invariant descriptors per structure."""
        assert self.calculator is not None, "Call load_model() first"
        all_embeddings = []
        for structure in tqdm(structures, desc="MACE embedding extraction"):
            try:
                atoms = self._structure_to_ase(structure)
                desc = self.calculator.get_descriptors(
                    atoms, invariants_only=True, num_layers=-1
                )
                # desc: (n_atoms, n_features)
                if desc.ndim == 3:
                    desc = desc.reshape(desc.shape[0], -1)
                embedding = np.mean(desc, axis=0)
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"  WARNING: embedding extraction failed: {e}")
                all_embeddings.append(np.full(64, np.nan))
        return np.array(all_embeddings, dtype=float)

    def fit_bg_head(
        self,
        train_structures: List[Structure],
        train_bg_targets: List[float],
    ) -> "MACEPredictor":
        print("\nExtracting MACE embeddings for BG head training...")
        embeddings = self.get_embeddings(train_structures)
        valid_mask = ~np.any(np.isnan(embeddings), axis=1)
        X = embeddings[valid_mask]
        y = np.array(train_bg_targets)[valid_mask]
        print(
            f"  Training BG head on {X.shape[0]} samples, "
            f"embedding dim: {X.shape[1]}"
        )

        self._bg_scaler = StandardScaler()
        X_scaled = self._bg_scaler.fit_transform(X)
        self._bg_head = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self._bg_head.fit(X_scaled, y)
        train_pred = self._bg_head.predict(X_scaled)
        train_mae = float(np.mean(np.abs(train_pred - y)))
        print(f"  BG head train MAE: {train_mae:.4f} eV")
        self._bg_head_trained = True
        return self

    def predict_bg(self, structures: List[Structure]) -> np.ndarray:
        assert self._bg_head_trained, "Call fit_bg_head() first"
        embeddings = self.get_embeddings(structures)
        valid_mask = ~np.any(np.isnan(embeddings), axis=1)
        predictions = np.full(len(structures), np.nan, dtype=float)
        if valid_mask.sum() > 0:
            X_scaled = self._bg_scaler.transform(embeddings[valid_mask])
            predictions[valid_mask] = self._bg_head.predict(X_scaled)
        return predictions

    def save(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._ref_energies:
            with open(output_dir / "ref_energies.json", "w") as f:
                json.dump({
                    "ref_energies": self._ref_energies,
                    "element_idx": self._element_idx,
                    "all_elements": self._all_elements,
                }, f, indent=2)
        if self._bg_head_trained:
            joblib.dump(self._bg_head, output_dir / "bg_head.joblib")
            joblib.dump(self._bg_scaler, output_dir / "bg_scaler.joblib")
        print(f"MACE predictor saved to {output_dir}")

    def load_artifacts(self, output_dir: Path):
        output_dir = Path(output_dir)
        ref_path = output_dir / "ref_energies.json"
        bg_path = output_dir / "bg_head.joblib"
        if ref_path.exists():
            with open(ref_path) as f:
                data = json.load(f)
            self._ref_energies = data["ref_energies"]
            self._element_idx = data["element_idx"]
            self._all_elements = data["all_elements"]
        if bg_path.exists():
            self._bg_head = joblib.load(bg_path)
            self._bg_scaler = joblib.load(output_dir / "bg_scaler.joblib")
            self._bg_head_trained = True
        print(f"MACE artifacts loaded from {output_dir}")
