"""
Weighted ensemble combining ALIGNN and MACE predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import optuna
from sklearn.isotonic import IsotonicRegression

from eumine_databridge.utils.metrics import compute_full_score, compute_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


class WeightedEnsemble:
    """Optuna-optimized weighted ensemble of ALIGNN and MACE predictions."""

    def __init__(self):
        self.weights_ef: Optional[Dict] = None
        self.weights_bg: Optional[Dict] = None
        self.best_score: Optional[float] = None
        self.study: Optional[optuna.Study] = None

    def fit(
        self,
        alignn_ef: np.ndarray,
        mace_ef: np.ndarray,
        true_ef: np.ndarray,
        alignn_bg: np.ndarray,
        mace_bg: np.ndarray,
        true_bg: np.ndarray,
        n_trials: int = 300,
    ) -> "WeightedEnsemble":
        print(f"\nOptimizing ensemble weights ({n_trials} trials)...")

        def objective(trial):
            w_alignn_ef = trial.suggest_float("w_alignn_ef", 0.1, 0.99)
            w_mace_ef = 1.0 - w_alignn_ef
            w_alignn_bg = trial.suggest_float("w_alignn_bg", 0.1, 0.99)
            w_mace_bg = 1.0 - w_alignn_bg

            ef_pred = w_alignn_ef * alignn_ef + w_mace_ef * mace_ef
            bg_pred = w_alignn_bg * alignn_bg + w_mace_bg * mace_bg
            bg_pred = np.clip(bg_pred, 0.0, None)

            mae_ef = float(np.mean(np.abs(ef_pred - true_ef)))
            mae_bg = float(np.mean(np.abs(bg_pred - true_bg)))
            scores = compute_full_score(mae_ef, mae_bg)
            return scores["total_performance_score"]

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = self.study.best_params
        self.weights_ef = {
            "alignn": best["w_alignn_ef"],
            "mace": 1.0 - best["w_alignn_ef"],
        }
        self.weights_bg = {
            "alignn": best["w_alignn_bg"],
            "mace": 1.0 - best["w_alignn_bg"],
        }
        self.best_score = self.study.best_value

        print(f"\nOptimal ensemble weights:")
        print(
            f"  EF: ALIGNN={self.weights_ef['alignn']:.3f}, "
            f"MACE={self.weights_ef['mace']:.3f}"
        )
        print(
            f"  BG: ALIGNN={self.weights_bg['alignn']:.3f}, "
            f"MACE={self.weights_bg['mace']:.3f}"
        )
        print(f"  Best val score: {self.best_score:.4f}/40")
        return self

    def predict(
        self,
        alignn_ef: np.ndarray,
        mace_ef: np.ndarray,
        alignn_bg: np.ndarray,
        mace_bg: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.weights_ef is not None, "Call fit() first"
        ef_pred = (
            self.weights_ef["alignn"] * alignn_ef
            + self.weights_ef["mace"] * mace_ef
        )
        bg_pred = (
            self.weights_bg["alignn"] * alignn_bg
            + self.weights_bg["mace"] * mace_bg
        )
        bg_pred = np.clip(bg_pred, 0.0, None)
        return ef_pred, bg_pred

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump({
                "weights_ef": self.weights_ef,
                "weights_bg": self.weights_bg,
                "best_score": self.best_score,
            }, f, indent=2)
        print(f"Ensemble weights saved to {path}")

    def load(self, path: Path) -> "WeightedEnsemble":
        with open(path) as f:
            data = json.load(f)
        self.weights_ef = data["weights_ef"]
        self.weights_bg = data["weights_bg"]
        self.best_score = data.get("best_score")
        return self


class CalibrationLayer:
    """Post-hoc isotonic regression calibration."""

    def __init__(self):
        self._ef_calibrator: Optional[IsotonicRegression] = None
        self._bg_calibrator: Optional[IsotonicRegression] = None

    def fit(
        self,
        ef_predictions: np.ndarray,
        ef_targets: np.ndarray,
        bg_predictions: np.ndarray,
        bg_targets: np.ndarray,
    ) -> "CalibrationLayer":
        print("\nFitting calibration layer...")

        self._ef_calibrator = IsotonicRegression(
            out_of_bounds="clip",
            increasing=True,
        )
        self._ef_calibrator.fit(ef_predictions, ef_targets)
        ef_cal = self._ef_calibrator.predict(ef_predictions)
        ef_mae_before = float(np.mean(np.abs(ef_predictions - ef_targets)))
        ef_mae_after = float(np.mean(np.abs(ef_cal - ef_targets)))
        print(f"  EF calibration: MAE {ef_mae_before:.4f} → {ef_mae_after:.4f}")

        bg_pred_clipped = np.clip(bg_predictions, 0.0, None)
        self._bg_calibrator = IsotonicRegression(
            out_of_bounds="clip",
            increasing=True,
        )
        self._bg_calibrator.fit(bg_pred_clipped, bg_targets)
        bg_cal = self._bg_calibrator.predict(bg_pred_clipped)
        bg_mae_before = float(np.mean(np.abs(bg_pred_clipped - bg_targets)))
        bg_mae_after = float(np.mean(np.abs(bg_cal - bg_targets)))
        print(f"  BG calibration: MAE {bg_mae_before:.4f} → {bg_mae_after:.4f}")
        return self

    def calibrate(
        self,
        ef_predictions: np.ndarray,
        bg_predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ef_cal = self._ef_calibrator.predict(ef_predictions)
        bg_cal = self._bg_calibrator.predict(
            np.clip(bg_predictions, 0.0, None)
        )
        bg_cal = np.clip(bg_cal, 0.0, None)
        return ef_cal, bg_cal

    def save(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._ef_calibrator, output_dir / "ef_calibrator.joblib")
        joblib.dump(self._bg_calibrator, output_dir / "bg_calibrator.joblib")
        print(f"Calibration layer saved to {output_dir}")

    def load(self, output_dir: Path) -> "CalibrationLayer":
        output_dir = Path(output_dir)
        self._ef_calibrator = joblib.load(output_dir / "ef_calibrator.joblib")
        self._bg_calibrator = joblib.load(output_dir / "bg_calibrator.joblib")
        return self
