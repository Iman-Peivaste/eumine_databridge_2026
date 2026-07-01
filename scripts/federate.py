"""
Federation Sprint CLI Tool — Stage 2 EuMINe DataBridge 2026.

This script is the single entry point for the Stage 2 federation sprint.
Run it once the calibration data is provided by the organizers.

Usage:
    # Dry run with our own models (to test before the sprint)
    python scripts/federate.py --dry_run

    # Real sprint usage
    python scripts/federate.py \
        --cal_structures /path/to/cal_structures \
        --cal_labels /path/to/cal_labels.csv \
        --test_structures data/raw/test_structures \
        --output submissions/CataLIST/predictions_federated.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "hackathon_ref" / "matfed-api-template"))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from eumine_databridge.matfed.federation import FederatedEnsemble
from eumine_databridge.matfed.predictor import LISTEuMINePredictor
from eumine_databridge.utils.metrics import compute_full_score


def load_structures_from_dir(structures_dir: str) -> tuple:
    """Load all CIF files from a directory."""
    path = Path(structures_dir)
    cif_files = sorted(path.glob("*.cif"))
    structures = []
    ids = []
    for cif in cif_files:
        try:
            s = Structure.from_file(str(cif))
            structures.append(s)
            ids.append(cif.stem)
        except Exception as e:
            print(f"  WARNING: could not load {cif.name}: {e}")
    print(f"Loaded {len(structures)} structures from {path}")
    return structures, ids


def dry_run():
    """
    Test federation using our own EF and BG models as mock teams.
    Validates the entire pipeline before the real sprint.
    """
    print("\n" + "="*60)
    print("FEDERATION DRY RUN")
    print("Using CataLIST EF and BG as mock separate teams")
    print("="*60)

    # Load val set as calibration proxy
    from eumine_databridge.data.loader import BridgeDataset
    val_ds = BridgeDataset(
        csv_path=ROOT / "data" / "raw" / "bridge_dataset_val.csv",
        structures_dir=ROOT / "data" / "raw" / "val_structures",
        split="val",
    )
    cal_structures = val_ds.get_structures()[:50]  # use 50 as calibration
    cal_ef, _ = val_ds.get_targets("formation_energy_per_atom")
    cal_bg, _ = val_ds.get_targets("band_gap")
    cal_ef = cal_ef[:50]
    cal_bg = cal_bg[:50]

    test_structures = val_ds.get_structures()[50:]  # use rest as test proxy
    test_ef_true = np.array(cal_ef[50:] if len(cal_ef) > 50
                            else cal_ef)

    # Load our predictor twice as mock teams
    print("\nLoading mock team predictors...")
    model_path = str(ROOT / "models" / "combined_retrain")

    p1 = LISTEuMINePredictor()
    p1.load_model(model_path)

    p2 = LISTEuMINePredictor()
    p2.load_model(model_path)

    # Build federation
    fed = FederatedEnsemble()
    fed.add_predictor(p1, "CataLIST_v1")
    fed.add_predictor(p2, "CataLIST_v2_mock")

    # Fit on calibration set
    result = fed.fit(
        cal_structures=cal_structures,
        cal_ef=cal_ef,
        cal_bg=cal_bg,
        n_trials=100,  # fewer for dry run
    )

    # Generate test predictions
    print("\nGenerating federated test predictions...")
    preds = fed.predict(test_structures, team_name="DryRun")

    print(f"\nDry run complete.")
    print(f"Pipeline works end-to-end.")
    print(f"Calibration score: {result['calibration_score']:.2f}/40")
    print(f"Ready for Stage 2 sprint.")

    # Save weights
    (ROOT / "models" / "federation").mkdir(parents=True, exist_ok=True)
    fed.save_weights(ROOT / "models" / "federation" / "dry_run_weights.json")


def main():
    parser = argparse.ArgumentParser(
        description="Federation sprint tool — Stage 2"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run dry run with our own models as mock teams",
    )
    parser.add_argument(
        "--cal_structures",
        type=str,
        default=None,
        help="Path to calibration CIF directory",
    )
    parser.add_argument(
        "--cal_labels",
        type=str,
        default=None,
        help="Path to calibration labels CSV",
    )
    parser.add_argument(
        "--test_structures",
        type=str,
        default=str(ROOT / "data" / "raw" / "test_structures"),
        help="Path to test CIF directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "submissions" / "CataLIST" /
                    "predictions_federated.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=200,
        help="Optuna trials for weight optimization",
    )

    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    # Real sprint mode
    print("\n" + "="*60)
    print("FEDERATION SPRINT — REAL MODE")
    print("="*60)

    assert args.cal_structures, "--cal_structures required"
    assert args.cal_labels, "--cal_labels required"

    # Load calibration data
    cal_structures, cal_ids = load_structures_from_dir(args.cal_structures)
    cal_df = pd.read_csv(args.cal_labels)
    cal_ef = cal_df['formation_energy_per_atom'].tolist()
    cal_bg = cal_df['band_gap'].tolist()

    # Load test structures
    test_structures, test_ids = load_structures_from_dir(args.test_structures)

    # Load our predictor
    print("\nLoading CataLIST predictor...")
    our_predictor = LISTEuMINePredictor()
    our_predictor.load_model(str(ROOT / "models" / "combined_retrain"))

    # Build federation
    fed = FederatedEnsemble()
    fed.add_predictor(our_predictor, "CataLIST")

    # NOTE: At the sprint, add other teams' predictors here:
    # from their_package.predictor import TheirPredictor
    # p2 = TheirPredictor()
    # p2.load_model("path/to/their/model")
    # fed.add_predictor(p2, "TakeMe2Romania")

    # Fit on calibration set
    result = fed.fit(
        cal_structures=cal_structures,
        cal_ef=cal_ef,
        cal_bg=cal_bg,
        n_trials=args.n_trials,
    )

    # Generate test predictions
    print("\nGenerating federated test predictions...")
    preds = fed.predict(test_structures, team_name="CataLIST_federation")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = {
        "team_name": "CataLIST",
        "model_id": "CataLIST_federation_v1",
        "matfed_api_version": "1.0",
        "federation_weights": result,
        "predictions": [
            {
                "material_id": test_ids[i],
                "formation_energy_per_atom": preds[i][
                    'formation_energy_per_atom'
                ],
                "band_gap": preds[i]['band_gap'],
            }
            for i in range(len(test_ids))
        ],
    }

    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"\nFederated predictions saved to {output_path}")
    print(f"Federation score estimate: {result['calibration_score']:.2f}/40")

    # Save weights
    (ROOT / "models" / "federation").mkdir(parents=True, exist_ok=True)
    fed.save_weights(
        ROOT / "models" / "federation" / "sprint_weights.json"
    )


if __name__ == "__main__":
    main()
