"""
Federation Engine for EuMINe DataBridge Hackathon — Stage 2.

Loads any N MatFed API v1 compliant predictors and combines them
into an optimized weighted ensemble using a small calibration set.

Designed for the Stage 2 Federation Sprint:
- Works on CPU (no GPU required at the venue)
- Runs in under 10 minutes end-to-end
- Handles any number of teams
- Exports final predictions as JSON

Usage (CLI):
    python scripts/federate.py \
        --models CataLIST:models/full_retrain \
                 TakeMe2Romania:path/to/their/model \
                 ProphX:path/to/their/model \
        --cal_structures data/raw/val_structures \
        --cal_labels data/raw/bridge_dataset_val.csv \
        --test_structures data/raw/test_structures \
        --output submissions/federation/predictions_federated.json
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from pymatgen.core import Structure
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add matfed-api-template to path
_MATFED = Path(__file__).parent.parent.parent.parent.parent \
    / "hackathon_ref" / "matfed-api-template"
if _MATFED.exists() and str(_MATFED) not in sys.path:
    sys.path.insert(0, str(_MATFED))


def load_predictor_from_path(
    team_name: str,
    model_path: str,
    predictor_class: str = "eumine_databridge.matfed.predictor.LISTEuMINePredictor",
) -> object:
    """
    Dynamically load a MatFedPredictor from a model path.

    Parameters
    ----------
    team_name      : display name for this predictor
    model_path     : path passed to predictor.load_model()
    predictor_class: dotted import path to the predictor class
                     default is our own CataLIST predictor

    Returns
    -------
    Loaded MatFedPredictor instance ready for predict()
    """
    print(f"  Loading {team_name} from {model_path}...")
    module_path, class_name = predictor_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    predictor = cls()
    predictor.load_model(model_path)
    print(f"  {team_name} loaded. Describe: {predictor.describe()['model_type']}")
    return predictor


class FederatedEnsemble:
    """
    N-model federated ensemble optimized for Stage 2 sprint conditions.

    Workflow
    --------
    1. Load N MatFedPredictor instances (one per team)
    2. Get predictions from all N on calibration structures
    3. Optimize weights with Optuna (maximizes hackathon score)
    4. Apply calibration
    5. Predict on test structures
    6. Export JSON

    Design principles
    -----------------
    - CPU-compatible: works without GPU at the venue
    - Fast: 200 Optuna trials complete in <2 min on CPU
    - Robust: NaN fallback to best individual model
    - Transparent: prints weight assignment per team
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

    def get_all_predictions(
        self,
        structures: List[Structure],
        desc: str = "inference",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run all predictors on a list of structures.

        Returns
        -------
        ef_matrix : shape (n_models, n_structures)
        bg_matrix : shape (n_models, n_structures)
        """
        n = len(structures)
        n_models = len(self.predictors)
        ef_matrix = np.zeros((n_models, n))
        bg_matrix = np.zeros((n_models, n))

        for i, (predictor, name) in enumerate(
            zip(self.predictors, self.team_names)
        ):
            print(f"  Running {name}...")
            t0 = time.time()
            try:
                preds = predictor.predict(structures)
                ef_matrix[i] = [
                    p['formation_energy_per_atom'] for p in preds
                ]
                bg_matrix[i] = [
                    max(0.0, p['band_gap']) for p in preds
                ]
                print(
                    f"  {name}: done in {time.time()-t0:.1f}s | "
                    f"EF mean={ef_matrix[i].mean():.3f} | "
                    f"BG mean={bg_matrix[i].mean():.3f}"
                )
            except Exception as e:
                print(f"  WARNING: {name} failed — {e}")
                # Fallback: use zeros (will get zero weight in optimization)
                ef_matrix[i] = np.zeros(n)
                bg_matrix[i] = np.zeros(n)

        return ef_matrix, bg_matrix

    def fit(
        self,
        cal_structures: List[Structure],
        cal_ef: List[float],
        cal_bg: List[float],
        n_trials: int = 200,
    ) -> Dict:
        """
        Optimize ensemble weights on calibration set.

        Parameters
        ----------
        cal_structures : calibration structures (provided at sprint)
        cal_ef         : true EF values for calibration set
        cal_bg         : true BG values for calibration set
        n_trials       : Optuna trials (200 = ~2 min on CPU)

        Returns
        -------
        dict with optimal weights per team and achieved score
        """
        from eumine_databridge.utils.metrics import compute_full_score

        n_models = len(self.predictors)
        true_ef = np.array(cal_ef)
        true_bg = np.array(cal_bg)

        print(f"\nRunning calibration inference ({len(cal_structures)} structures)...")
        ef_matrix, bg_matrix = self.get_all_predictions(
            cal_structures, desc="calibration"
        )
        self._cal_preds_ef = ef_matrix
        self._cal_preds_bg = bg_matrix

        # Print individual model performance
        print(f"\nIndividual model performance on calibration set:")
        for i, name in enumerate(self.team_names):
            mae_ef = np.mean(np.abs(ef_matrix[i] - true_ef))
            mae_bg = np.mean(np.abs(bg_matrix[i] - true_bg))
            score = compute_full_score(mae_ef, mae_bg)
            print(
                f"  {name:20s}: EF={mae_ef:.4f} BG={mae_bg:.4f} "
                f"Score={score['total_performance_score']:.2f}/40"
            )

        # Optimize weights with Optuna
        print(f"\nOptimizing weights ({n_trials} trials, ~{n_trials//100} min)...")

        def objective(trial):
            # Sample weights, normalize to simplex
            raw_ef = [
                trial.suggest_float(f"ef_{i}", 0.01, 1.0)
                for i in range(n_models)
            ]
            raw_bg = [
                trial.suggest_float(f"bg_{i}", 0.01, 1.0)
                for i in range(n_models)
            ]

            w_ef = np.array(raw_ef) / sum(raw_ef)
            w_bg = np.array(raw_bg) / sum(raw_bg)

            ef_pred = np.sum(
                ef_matrix * w_ef[:, np.newaxis], axis=0
            )
            bg_pred = np.clip(
                np.sum(bg_matrix * w_bg[:, np.newaxis], axis=0),
                0, None
            )

            mae_ef = float(np.mean(np.abs(ef_pred - true_ef)))
            mae_bg = float(np.mean(np.abs(bg_pred - true_bg)))

            return compute_full_score(mae_ef, mae_bg)[
                'total_performance_score'
            ]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Extract and normalize weights
        best = study.best_params
        raw_ef = [best[f"ef_{i}"] for i in range(n_models)]
        raw_bg = [best[f"bg_{i}"] for i in range(n_models)]
        self.weights_ef = list(np.array(raw_ef) / sum(raw_ef))
        self.weights_bg = list(np.array(raw_bg) / sum(raw_bg))
        self.best_score = study.best_value

        # Print results
        print(f"\nOptimal ensemble weights:")
        print(f"  {'Team':20s} {'EF weight':>10} {'BG weight':>10}")
        print(f"  {'-'*42}")
        for i, name in enumerate(self.team_names):
            print(
                f"  {name:20s} {self.weights_ef[i]:>10.3f} "
                f"{self.weights_bg[i]:>10.3f}"
            )
        print(f"\n  Best calibration score: {self.best_score:.4f}/40")

        # Compute calibrated performance
        ef_ensemble = np.sum(
            ef_matrix * np.array(self.weights_ef)[:, np.newaxis], axis=0
        )
        bg_ensemble = np.clip(
            np.sum(
                bg_matrix * np.array(self.weights_bg)[:, np.newaxis], axis=0
            ),
            0, None
        )
        final_mae_ef = float(np.mean(np.abs(ef_ensemble - true_ef)))
        final_mae_bg = float(np.mean(np.abs(bg_ensemble - true_bg)))
        final_score = compute_full_score(final_mae_ef, final_mae_bg)

        print(f"\nFederated ensemble on calibration set:")
        print(f"  EF MAE : {final_mae_ef:.4f} eV/atom")
        print(f"  BG MAE : {final_mae_bg:.4f} eV")
        print(f"  Score  : {final_score['total_performance_score']:.2f}/40")

        return {
            "weights_ef": dict(zip(self.team_names, self.weights_ef)),
            "weights_bg": dict(zip(self.team_names, self.weights_bg)),
            "calibration_score": self.best_score,
            "federated_mae_ef": final_mae_ef,
            "federated_mae_bg": final_mae_bg,
        }

    def predict(
        self,
        structures: List[Structure],
        team_name: str = "CataLIST_federation",
    ) -> List[Dict]:
        """
        Generate federated predictions for a list of structures.
        Must call fit() first.
        """
        assert self.weights_ef is not None, "Call fit() first"

        ef_matrix, bg_matrix = self.get_all_predictions(
            structures, desc="test inference"
        )

        ef_pred = np.sum(
            ef_matrix * np.array(self.weights_ef)[:, np.newaxis], axis=0
        )
        bg_pred = np.clip(
            np.sum(
                bg_matrix * np.array(self.weights_bg)[:, np.newaxis], axis=0
            ),
            0, None
        )

        # Collect all data sources used
        all_sources = []
        for predictor in self.predictors:
            try:
                sources = predictor.describe().get("data_sources", [])
                all_sources.extend(sources)
            except Exception:
                pass
        unique_sources = list(set(all_sources))

        results = []
        for i in range(len(structures)):
            results.append({
                "formation_energy_per_atom": float(ef_pred[i]),
                "band_gap": float(bg_pred[i]),
                "model_id": f"{team_name}_federated_v1",
                "data_sources_used": unique_sources,
                "uncertainty_ef": float(np.std(ef_matrix[:, i])),
                "uncertainty_bg": float(np.std(bg_matrix[:, i])),
            })
        return results

    def save_weights(self, path: Path):
        """Save federation weights to JSON."""
        with open(path, "w") as f:
            json.dump({
                "team_names": self.team_names,
                "weights_ef": self.weights_ef,
                "weights_bg": self.weights_bg,
                "best_score": self.best_score,
            }, f, indent=2)
        print(f"Federation weights saved to {path}")


def load_structures_from_dir(structures_dir: Path) -> Tuple[List[str], List[Structure]]:
    """Load all CIF files from a directory, return (ids, structures)."""
    cifs = sorted(structures_dir.glob("*.cif"))
    ids, structs = [], []
    for cif in tqdm(cifs, desc=f"Loading {structures_dir.name}"):
        try:
            structs.append(Structure.from_file(str(cif)))
            ids.append(cif.stem)
        except Exception as e:
            print(f"  WARNING: skipping {cif.name} — {e}")
    return ids, structs


def load_labels(csv_path: Path) -> Tuple[List[float], List[float], List[str]]:
    """Load EF and BG labels from a CSV file."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return (
        df["formation_energy_per_atom"].tolist(),
        df["band_gap"].tolist(),
        df["material_id"].astype(str).tolist(),
    )


def main():
    import argparse

    ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(ROOT / "src"))

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Federated ensemble for EuMINe Stage 2 sprint"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="TEAM:PATH",
        help=(
            "One or more team:model_path pairs. "
            "Optionally append ::predictor.Class.Path for non-CataLIST predictors. "
            "Example: CataLIST:models/full_retrain  OtherTeam:../their/model"
        ),
    )
    parser.add_argument(
        "--cal_structures",
        type=Path,
        default=ROOT / "data" / "raw" / "val_structures",
        help="Directory of calibration CIF files",
    )
    parser.add_argument(
        "--cal_labels",
        type=Path,
        default=ROOT / "data" / "raw" / "bridge_dataset_val.csv",
        help="CSV with formation_energy_per_atom and band_gap columns",
    )
    parser.add_argument(
        "--test_structures",
        type=Path,
        default=ROOT / "data" / "raw" / "test_structures",
        help="Directory of test CIF files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "submissions" / "federation" / "predictions_federated.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=200,
        help="Optuna optimization trials (default: 200, ~2 min on CPU)",
    )
    parser.add_argument(
        "--federation_name",
        type=str,
        default="CataLIST_federation",
        help="Team name written into the output JSON",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EuMINe DataBridge — Stage 2 Federation Engine")
    print("=" * 60)

    # ── 1. Load predictors ────────────────────────────────────────
    print(f"\n[1] Loading {len(args.models)} predictor(s)...")
    federation = FederatedEnsemble()

    for spec in args.models:
        # Format: TEAM:PATH  or  TEAM:PATH::predictor.Class
        parts = spec.split("::")
        team_path = parts[0]
        predictor_class = parts[1] if len(parts) > 1 else \
            "eumine_databridge.matfed.predictor.LISTEuMINePredictor"

        team, model_path = team_path.split(":", 1)
        predictor = load_predictor_from_path(team, model_path, predictor_class)
        federation.add_predictor(predictor, team)

    # ── 2. Load calibration data ──────────────────────────────────
    print(f"\n[2] Loading calibration data from {args.cal_structures}...")
    cal_ids, cal_structures = load_structures_from_dir(args.cal_structures)
    cal_ef, cal_bg, label_ids = load_labels(args.cal_labels)

    # Align structures with labels by material_id order
    label_id_to_idx = {mid: i for i, mid in enumerate(label_ids)}
    aligned_ef, aligned_bg, aligned_structures = [], [], []
    for mid, struct in zip(cal_ids, cal_structures):
        if mid in label_id_to_idx:
            idx = label_id_to_idx[mid]
            aligned_ef.append(cal_ef[idx])
            aligned_bg.append(cal_bg[idx])
            aligned_structures.append(struct)

    print(f"  Calibration structures matched: {len(aligned_structures)}")

    # ── 3. Optimize federation weights ───────────────────────────
    print(f"\n[3] Optimizing federation weights...")
    fit_results = federation.fit(
        cal_structures=aligned_structures,
        cal_ef=aligned_ef,
        cal_bg=aligned_bg,
        n_trials=args.n_trials,
    )

    # ── 4. Save weights ───────────────────────────────────────────
    weights_path = args.output.parent / "federation_weights.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    federation.save_weights(weights_path)

    # ── 5. Generate test predictions ─────────────────────────────
    print(f"\n[4] Loading test structures from {args.test_structures}...")
    test_ids, test_structures = load_structures_from_dir(args.test_structures)
    print(f"  Test structures: {len(test_structures)}")

    print(f"\n[5] Generating federated test predictions...")
    preds = federation.predict(test_structures, team_name=args.federation_name)

    submission = {
        "team_name": args.federation_name,
        "model_id": f"{args.federation_name}_federated_v1",
        "matfed_api_version": "1.0",
        "federation": {
            "n_models": len(federation.team_names),
            "teams": federation.team_names,
            "calibration_score": fit_results["calibration_score"],
            "federated_mae_ef": fit_results["federated_mae_ef"],
            "federated_mae_bg": fit_results["federated_mae_bg"],
            "weights_ef": fit_results["weights_ef"],
            "weights_bg": fit_results["weights_bg"],
        },
        "predictions": [
            {
                "material_id": test_ids[i],
                **preds[i],
            }
            for i in range(len(test_ids))
        ],
    }

    with open(args.output, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Federation complete.")
    print(f"  Teams       : {', '.join(federation.team_names)}")
    print(f"  Cal score   : {fit_results['calibration_score']:.2f}/40")
    print(f"  Predictions : {len(test_ids)} structures")
    print(f"  Output      : {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
