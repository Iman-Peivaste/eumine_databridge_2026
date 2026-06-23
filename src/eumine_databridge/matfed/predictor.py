"""
MatFed API v1 implementation for the EuMINe DataBridge Hackathon 2026.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_MATFED_API = _PROJECT_ROOT.parent / "hackathon_ref" / "matfed-api-template"
if _MATFED_API.is_dir() and str(_MATFED_API) not in sys.path:
    sys.path.insert(0, str(_MATFED_API))

from matfed_api.predictor import MatFedPredictor  # noqa: E402


def _resolve_artifact_paths(model_path: Path) -> Dict[str, Path]:
    """
    Resolve ALIGNN / MACE / ensemble paths for full_retrain or Step 4 layout.

    full_retrain layout (Step 4B):
        model_path/alignn_ef_full, alignn_bg_full, mace_artifacts, ...

    Step 4 layout (model_path = models/ensemble):
        ../alignn_ef, ../alignn_bg, mace_artifacts, ensemble_weights.json, calibration/
    """
    base = model_path.resolve()
    if (base / "alignn_ef_full" / "best_model.pt").exists():
        return {
            "ef_dir": base / "alignn_ef_full",
            "bg_dir": base / "alignn_bg_full",
            "mace_dir": base / "mace_artifacts",
            "ensemble_weights": base / "ensemble_weights.json",
            "calibration_dir": base / "calibration",
        }

    root = base.parent if base.name == "ensemble" else base
    ens = root / "ensemble" if base.name != "ensemble" else base
    ef_dir = root / "alignn_ef"
    bg_dir = root / "alignn_bg"
    if not (ef_dir / "best_model.pt").exists():
        raise FileNotFoundError(
            f"No ALIGNN EF checkpoint under {ef_dir} or {base / 'alignn_ef_full'}"
        )
    return {
        "ef_dir": ef_dir,
        "bg_dir": bg_dir,
        "mace_dir": ens / "mace_artifacts",
        "ensemble_weights": ens / "ensemble_weights.json",
        "calibration_dir": ens / "calibration",
    }


def _load_performance_block(base: Path) -> Dict:
    """OOF metrics from full_retrain artifacts or submission JSON."""
    for candidate in (
        base / "oof_metrics.json",
        _PROJECT_ROOT / "submissions" / "LIST_EuMINe" / "predictions_test.json",
    ):
        if candidate.exists():
            data = json.loads(candidate.read_text())
            if "oof_mae_ef" in data:
                return {
                    "oof_mae_ef_eV_per_atom": data.get("oof_mae_ef"),
                    "oof_mae_bg_eV": data.get("oof_mae_bg"),
                    "oof_score_40pts": data.get("oof_score"),
                    "val_score_40pts": data.get("val_score"),
                }
    return {
        "oof_mae_ef_eV_per_atom": 0.0533,
        "oof_mae_bg_eV": 0.1951,
        "oof_score_ef_20pts": 18.10,
        "oof_score_bg_20pts": 17.07,
        "oof_total_score_40pts": 35.17,
        "val_score_40pts": 35.96,
        "val_mae_ef_eV_per_atom": 0.0516,
        "val_mae_bg_eV": 0.1496,
    }


class LISTEuMINePredictor(MatFedPredictor):
    """
    ALIGNN + MACE-MP-0 ensemble with isotonic calibration.
    """

    API_VERSION = "1.0"
    MODEL_VERSION = "v2_fullretrain"
    TEAM_NAME = "LIST_EuMINe"

    def __init__(self) -> None:
        self._ef_trainer = None
        self._bg_trainer = None
        self._mace = None
        self._ensemble = None
        self._calibrator = None
        self._loaded = False
        self._model_base: Optional[Path] = None
        self._performance: Dict = {}

        model_path = os.environ.get("MATFED_MODEL_PATH")
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        from eumine_databridge.models.alignn_config import get_bg_config, get_ef_config
        from eumine_databridge.models.alignn_model import ALIGNNFineTuner
        from eumine_databridge.models.ensemble import CalibrationLayer, WeightedEnsemble
        from eumine_databridge.models.mace_model import MACEPredictor

        base = Path(model_path)
        paths = _resolve_artifact_paths(base)
        self._model_base = base.resolve()
        self._performance = _load_performance_block(self._model_base)

        print(f"Loading LISTEuMINePredictor from {self._model_base}...")

        ef_cfg = get_ef_config()
        ef_cfg.output_dir = paths["ef_dir"]
        self._ef_trainer = ALIGNNFineTuner(ef_cfg)
        self._ef_trainer._init_model_only()
        self._ef_trainer.load_best_model()
        print("  ALIGNN EF loaded")

        bg_cfg = get_bg_config()
        bg_cfg.output_dir = paths["bg_dir"]
        self._bg_trainer = ALIGNNFineTuner(bg_cfg)
        self._bg_trainer._init_model_only()
        self._bg_trainer.load_best_model()
        print("  ALIGNN BG loaded")

        self._mace = MACEPredictor(model_name="medium")
        self._mace.load_model()
        self._mace.load_artifacts(paths["mace_dir"])
        print("  MACE-MP-0 loaded")

        self._ensemble = WeightedEnsemble()
        self._ensemble.load(paths["ensemble_weights"])
        print("  Ensemble weights loaded")

        self._calibrator = CalibrationLayer()
        self._calibrator.load(paths["calibration_dir"])
        print("  Calibration layer loaded")

        if (self._model_base / "alignn_ef_full").exists():
            self.MODEL_VERSION = "v2_fullretrain"
        else:
            self.MODEL_VERSION = "v1_ensemble"

        self._loaded = True
        print("LISTEuMINePredictor ready.")

    def predict(self, structures: List) -> List[Dict]:
        assert self._loaded, (
            "Model not loaded. Call load_model(model_path) or set MATFED_MODEL_PATH."
        )
        if not structures:
            return []

        mat_ids = [f"pred_{i:04d}" for i in range(len(structures))]

        alignn_ef = self._ef_trainer.predict(structures, mat_ids)
        alignn_bg = self._bg_trainer.predict(structures, mat_ids)

        mace_ef = self._mace.predict_ef(structures)
        mace_bg = self._mace.predict_bg(structures)
        mace_ef = np.where(np.isnan(mace_ef), alignn_ef, mace_ef)
        mace_bg = np.where(np.isnan(mace_bg), alignn_bg, mace_bg)

        ef_ensemble, bg_ensemble = self._ensemble.predict(
            alignn_ef, mace_ef, alignn_bg, mace_bg
        )
        ef_final, bg_final = self._calibrator.calibrate(
            ef_ensemble, bg_ensemble
        )
        bg_final = np.clip(bg_final, 0.0, None)

        uncertainty_ef = np.abs(alignn_ef - mace_ef)
        uncertainty_bg = np.abs(alignn_bg - mace_bg)

        model_id = f"{self.TEAM_NAME}_{self.MODEL_VERSION}"
        return [
            {
                "formation_energy_per_atom": float(ef_final[i]),
                "band_gap": float(bg_final[i]),
                "model_id": model_id,
                "data_sources_used": [
                    "Materials Project",
                    "JARVIS-DFT",
                    "OQMD",
                ],
                "uncertainty_ef": float(uncertainty_ef[i]),
                "uncertainty_bg": float(uncertainty_bg[i]),
            }
            for i in range(len(structures))
        ]

    def describe(self) -> Dict:
        perf = dict(self._performance)
        return {
            "team_name": self.TEAM_NAME,
            "institution": (
                "Luxembourg Institute of Science and Technology (LIST)"
            ),
            "model_type": (
                "ALIGNN + MACE-MP-0 weighted ensemble "
                "with isotonic calibration"
            ),
            "api_version": self.API_VERSION,
            "model_version": self.MODEL_VERSION,
            "data_sources": [
                "Materials Project",
                "JARVIS-DFT",
                "OQMD",
            ],
            "properties_predicted": [
                "formation_energy_per_atom",
                "band_gap",
            ],
            "architecture": {
                "alignn_layers_ef": 4,
                "alignn_layers_bg": 6,
                "gcn_layers": 4,
                "hidden_features": 256,
                "mace_model": "medium",
                "ensemble_method": "optuna_weighted",
                "calibration": "isotonic_regression",
            },
            "training": {
                "n_train_structures": 850,
                "cv_folds": 5,
                "pretrained_from": (
                    "JARVIS-DFT (ALIGNN), Materials Project (MACE)"
                ),
                "fine_tuned_on": "EuMINe Bridge Dataset",
            },
            "performance": perf,
            "uncertainty_available": True,
            "uncertainty_method": (
                "Absolute difference between ALIGNN and MACE predictions"
            ),
            "contact": "euminecost@gmail.com",
            "repository": (
                "https://github.com/YourGitHub/eumine_databridge_2026"
            ),
        }
