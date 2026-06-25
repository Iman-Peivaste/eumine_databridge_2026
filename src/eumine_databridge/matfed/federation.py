"""
FederatedEnsemble — combines N MatFed API v1 predictors into an
Optuna-optimized weighted ensemble for the Stage 2 federation sprint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from pymatgen.core import Structure

optuna.logging.set_verbosity(optuna.logging.WARNING)


class FederatedEnsemble:
    """
    N-model federated ensemble optimized for Stage 2 sprint conditions.

    Workflow
    --------
    1. Load N MatFedPredictor instances via add_predictor()
    2. Call fit() with calibration structures + labels
    3. Call predict() on test structures
    4. Optionally call save_weights() to persist the result

    Design principles
    -----------------
    - CPU-compatible: no GPU required at the venue
    - Fast: 200 Optuna trials complete in <2 min on CPU
    - Robust: failed predictors fall back to zeros (zero weight)
    - Transparent: prints per-team weights and individual scores
    """

    def __init__(self):
        self.predictors: List[object] = []
        self.team_names: List[str] = []
        self.weights_ef: Optional[List[float]] = None
        self.weights_bg: Optional[List[float]] = None
        self.best_score: Optional[float] = None
        self._cal_preds_ef: Optional[np.ndarray] = None
        self._cal_preds_bg: Optional[np.ndarray] = None

    def add_predictor(self, predictor, team_name: str):
        """Add a loaded MatFedPredictor to the federation."""
        self.predictors.append(predictor)
        self.team_names.append(team_name)
        print(f"  Added: {team_name} (total: {len(self.predictors)} models)")

    def _run_all(
        self,
        structures: List[Structure],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run all predictors on structures.

        Returns ef_matrix, bg_matrix each of shape (n_models, n_structures).
        A predictor that raises falls back to zeros and will receive zero weight.
        """
        n = len(structures)
        ef_matrix = np.zeros((len(self.predictors), n))
        bg_matrix = np.zeros((len(self.predictors), n))

        for i, (predictor, name) in enumerate(
            zip(self.predictors, self.team_names)
        ):
            print(f"  Running {name}...")
            t0 = time.time()
            try:
                preds = predictor.predict(structures)
                ef_matrix[i] = [p["formation_energy_per_atom"] for p in preds]
                bg_matrix[i] = [max(0.0, p["band_gap"]) for p in preds]
                print(
                    f"  {name}: {time.time()-t0:.1f}s | "
                    f"EF mean={ef_matrix[i].mean():.3f} | "
                    f"BG mean={bg_matrix[i].mean():.3f}"
                )
            except Exception as e:
                print(f"  WARNING: {name} failed ({e}) — using zeros")

        return ef_matrix, bg_matrix

    def fit(
        self,
        cal_structures: List[Structure],
        cal_ef: List[float],
        cal_bg: List[float],
        n_trials: int = 200,
    ) -> Dict:
        """
        Optimize ensemble weights on calibration structures.

        Parameters
        ----------
        cal_structures : structures provided at the sprint
        cal_ef         : true formation energies (eV/atom)
        cal_bg         : true band gaps (eV)
        n_trials       : Optuna trials (200 ≈ 2 min on CPU)

        Returns
        -------
        dict with per-team weights, calibration score, and MAEs
        """
        from eumine_databridge.utils.metrics import compute_full_score

        n_models = len(self.predictors)
        true_ef = np.array(cal_ef)
        true_bg = np.array(cal_bg)

        print(f"\nCalibration inference ({len(cal_structures)} structures)...")
        ef_matrix, bg_matrix = self._run_all(cal_structures)
        self._cal_preds_ef = ef_matrix
        self._cal_preds_bg = bg_matrix

        print(f"\nIndividual model scores on calibration set:")
        for i, name in enumerate(self.team_names):
            mae_ef = float(np.mean(np.abs(ef_matrix[i] - true_ef)))
            mae_bg = float(np.mean(np.abs(bg_matrix[i] - true_bg)))
            s = compute_full_score(mae_ef, mae_bg)
            print(
                f"  {name:20s}: EF={mae_ef:.4f}  BG={mae_bg:.4f}  "
                f"Score={s['total_performance_score']:.2f}/40"
            )

        print(f"\nOptimizing weights ({n_trials} Optuna trials)...")

        def objective(trial):
            raw_ef = [trial.suggest_float(f"ef_{i}", 0.01, 1.0)
                      for i in range(n_models)]
            raw_bg = [trial.suggest_float(f"bg_{i}", 0.01, 1.0)
                      for i in range(n_models)]
            w_ef = np.array(raw_ef) / sum(raw_ef)
            w_bg = np.array(raw_bg) / sum(raw_bg)

            ef_pred = np.sum(ef_matrix * w_ef[:, np.newaxis], axis=0)
            bg_pred = np.clip(
                np.sum(bg_matrix * w_bg[:, np.newaxis], axis=0), 0, None
            )
            mae_ef = float(np.mean(np.abs(ef_pred - true_ef)))
            mae_bg = float(np.mean(np.abs(bg_pred - true_bg)))
            return compute_full_score(mae_ef, mae_bg)["total_performance_score"]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        raw_ef = [best[f"ef_{i}"] for i in range(n_models)]
        raw_bg = [best[f"bg_{i}"] for i in range(n_models)]
        self.weights_ef = list(np.array(raw_ef) / sum(raw_ef))
        self.weights_bg = list(np.array(raw_bg) / sum(raw_bg))
        self.best_score = study.best_value

        print(f"\nOptimal weights:")
        print(f"  {'Team':20s} {'EF weight':>10} {'BG weight':>10}")
        print(f"  {'-'*42}")
        for i, name in enumerate(self.team_names):
            print(
                f"  {name:20s} {self.weights_ef[i]:>10.3f} "
                f"{self.weights_bg[i]:>10.3f}"
            )
        print(f"\n  Best calibration score: {self.best_score:.4f}/40")

        ef_ens = np.sum(
            ef_matrix * np.array(self.weights_ef)[:, np.newaxis], axis=0
        )
        bg_ens = np.clip(
            np.sum(bg_matrix * np.array(self.weights_bg)[:, np.newaxis], axis=0),
            0, None
        )
        mae_ef_final = float(np.mean(np.abs(ef_ens - true_ef)))
        mae_bg_final = float(np.mean(np.abs(bg_ens - true_bg)))

        return {
            "weights_ef": dict(zip(self.team_names, self.weights_ef)),
            "weights_bg": dict(zip(self.team_names, self.weights_bg)),
            "calibration_score": self.best_score,
            "federated_mae_ef": mae_ef_final,
            "federated_mae_bg": mae_bg_final,
        }

    def predict(
        self,
        structures: List[Structure],
        team_name: str = "CataLIST_federation",
    ) -> List[Dict]:
        """
        Generate federated predictions. Must call fit() first.

        Returns a list of dicts with formation_energy_per_atom, band_gap,
        model_id, data_sources_used, uncertainty_ef, uncertainty_bg.
        """
        assert self.weights_ef is not None, "Call fit() before predict()"

        ef_matrix, bg_matrix = self._run_all(structures)

        ef_pred = np.sum(
            ef_matrix * np.array(self.weights_ef)[:, np.newaxis], axis=0
        )
        bg_pred = np.clip(
            np.sum(bg_matrix * np.array(self.weights_bg)[:, np.newaxis], axis=0),
            0, None
        )

        all_sources: List[str] = []
        for predictor in self.predictors:
            try:
                all_sources.extend(predictor.describe().get("data_sources", []))
            except Exception:
                pass

        return [
            {
                "formation_energy_per_atom": float(ef_pred[i]),
                "band_gap": float(bg_pred[i]),
                "model_id": f"{team_name}_federated_v1",
                "data_sources_used": list(set(all_sources)),
                "uncertainty_ef": float(np.std(ef_matrix[:, i])),
                "uncertainty_bg": float(np.std(bg_matrix[:, i])),
            }
            for i in range(len(structures))
        ]

    def save_weights(self, path: Path):
        """Persist federation weights to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "team_names": self.team_names,
                    "weights_ef": self.weights_ef,
                    "weights_bg": self.weights_bg,
                    "best_score": self.best_score,
                },
                f,
                indent=2,
            )
        print(f"Federation weights saved to {path}")
