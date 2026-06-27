"""
CataLIST Federation Sprint Launcher — Stage 2 EuMINe DataBridge 2026

Single interactive script for the federation sprint at Cluj.

Usage:
    conda activate catallist_stage2
    python scripts/sprint_launcher.py
"""

import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

MATFED = ROOT.parent / "hackathon_ref" / "matfed-api-template"
if MATFED.exists():
    sys.path.insert(0, str(MATFED))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def ask(prompt: str, default: str = None) -> str:
    if default:
        display = f"{prompt} [{default}]: "
    else:
        display = f"{prompt}: "
    val = input(display).strip()
    return val if val else (default or "")


def ask_int(prompt: str, default: int = None) -> int:
    while True:
        raw = ask(prompt, str(default) if default is not None else None)
        try:
            return int(raw)
        except ValueError:
            print(f"  Please enter a number.")


def ask_yn(prompt: str) -> bool:
    while True:
        val = ask(f"{prompt} (y/n)").lower()
        if val in ("y", "yes"):
            return True
        if val in ("n", "no"):
            return False
        print("  Please enter y or n.")


def banner():
    print()
    print("=" * 60)
    print("  CataLIST Federation Sprint Launcher")
    print("  EuMINe DataBridge 2026 — Stage 2")
    print("=" * 60)
    print()


def load_our_predictor() -> object:
    from eumine_databridge.matfed.predictor import LISTEuMINePredictor
    model_path = ROOT / "models" / "full_retrain"
    print(f"  Loading CataLIST from {model_path} ...")
    p = LISTEuMINePredictor()
    p.load_model(str(model_path))
    print("  CataLIST loaded.")
    return p


def try_load_partner(team_name: str, model_path: str, import_path: str):
    """
    Attempt to load a partner predictor.
    import_path: e.g. "their_pkg.predictor.TheirPredictor"
    Returns predictor or None.
    """
    if not import_path:
        return try_generic_load(team_name, model_path)

    try:
        parts = import_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ImportError(f"Expected 'module.ClassName', got '{import_path}'")
        module_name, class_name = parts
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        predictor = cls()
        predictor.load_model(model_path)
        print(f"  {team_name} loaded via {import_path}")
        return predictor
    except Exception as e:
        print(f"\n  ERROR loading {team_name}: {e}")
        return None


def try_generic_load(team_name: str, model_path: str):
    """
    Try common predictor class names as a fallback.
    """
    candidates = [
        ("matfed_api.predictor", "MatFedPredictor"),
        ("predictor", "Predictor"),
        ("model", "Model"),
    ]
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            predictor = cls()
            predictor.load_model(model_path)
            print(f"  {team_name} loaded via {module_name}.{class_name} (generic)")
            return predictor
        except Exception:
            continue
    print(f"\n  Could not auto-detect predictor class for {team_name}.")
    print(f"  You will need to provide the import path manually.")
    return None


def load_calibration(cal_structures_dir: str, cal_labels_csv: str):
    import pandas as pd
    from pymatgen.core import Structure

    cal_path = Path(cal_structures_dir)
    cif_files = sorted(cal_path.glob("*.cif"))
    print(f"  Loading {len(cif_files)} calibration CIFs...")

    structures, ids = [], []
    for cif in cif_files:
        try:
            structures.append(Structure.from_file(str(cif)))
            ids.append(cif.stem)
        except Exception as e:
            print(f"  WARNING: skipped {cif.name}: {e}")

    df = pd.read_csv(cal_labels_csv)
    df = df.set_index("material_id")
    ef, bg = [], []
    used_ids, used_structures = [], []
    for mat_id, s in zip(ids, structures):
        if mat_id in df.index:
            ef.append(float(df.loc[mat_id, "formation_energy_per_atom"]))
            bg.append(float(df.loc[mat_id, "band_gap"]))
            used_ids.append(mat_id)
            used_structures.append(s)
        else:
            print(f"  WARNING: {mat_id} not in labels CSV — skipped")

    print(f"  Calibration set: {len(used_structures)} structures matched")
    return used_structures, ef, bg


def load_test_structures(test_dir: str):
    from pymatgen.core import Structure

    path = Path(test_dir)
    cif_files = sorted(path.glob("*.cif"))
    print(f"  Loading {len(cif_files)} test CIFs...")
    structures, ids = [], []
    for cif in cif_files:
        try:
            structures.append(Structure.from_file(str(cif)))
            ids.append(cif.stem)
        except Exception as e:
            print(f"  WARNING: skipped {cif.name}: {e}")
    print(f"  Test set: {len(structures)} structures loaded")
    return structures, ids


def print_weight_table(team_names, weights_ef, weights_bg):
    print()
    print(f"  {'Team':<25} {'EF weight':>10} {'BG weight':>10}")
    print(f"  {'-'*47}")
    for name, wef, wbg in zip(team_names, weights_ef, weights_bg):
        print(f"  {name:<25} {wef:>10.3f} {wbg:>10.3f}")
    print()


def main():
    t_start = time.time()
    banner()

    # ── Step 1: Load our own model ────────────────────────────────
    print("[1] Loading CataLIST model...")
    our_predictor = load_our_predictor()
    print()

    # ── Step 2: How many teams total? ─────────────────────────────
    n_teams = ask_int("[2] How many teams in your group? (including CataLIST)", default=2)
    n_partners = n_teams - 1

    # ── Step 3: Load partner predictors ───────────────────────────
    partner_predictors = []
    partner_names = []

    for i in range(n_partners):
        print(f"\n[3.{i+1}] Partner team {i+1} of {n_partners}")
        team_name = ask("  Team name")

        while True:
            model_path = ask("  Path to their model directory")
            if not Path(model_path).exists():
                print(f"  WARNING: path does not exist: {model_path}")
                if not ask_yn("  Continue anyway?"):
                    continue

            import_path = ask(
                "  Python import path to their predictor class\n"
                "  (e.g. their_pkg.predictor.TheirPredictor)\n"
                "  Press Enter to skip (generic loader)",
                default="",
            )

            predictor = try_load_partner(team_name, model_path, import_path)

            if predictor is not None:
                partner_predictors.append(predictor)
                partner_names.append(team_name)
                break
            else:
                if ask_yn("  Try a different import path?"):
                    continue
                else:
                    print(f"  Skipping {team_name} — will not be included in federation.")
                    break

    # ── Step 4: Build federation ──────────────────────────────────
    from eumine_databridge.matfed.federation import FederatedEnsemble

    fed = FederatedEnsemble()
    print(f"\n[4] Building federation with {1 + len(partner_predictors)} models...")
    fed.add_predictor(our_predictor, "CataLIST")
    for name, pred in zip(partner_names, partner_predictors):
        fed.add_predictor(pred, name)
    print()

    # ── Step 5: Calibration data ──────────────────────────────────
    print("[5] Calibration data")
    cal_structures_dir = ask("  Path to calibration structures directory")
    cal_labels_csv = ask("  Path to calibration labels CSV")

    cal_structures, cal_ef, cal_bg = load_calibration(
        cal_structures_dir, cal_labels_csv
    )
    print()

    # ── Step 6: Test structures ───────────────────────────────────
    print("[6] Test structures")
    test_dir = ask(
        "  Path to test structures directory",
        default=str(ROOT / "data" / "raw" / "test_structures"),
    )
    test_structures, test_ids = load_test_structures(test_dir)
    print()

    # ── Step 7: Optuna trials ─────────────────────────────────────
    n_trials = ask_int("[7] Number of Optuna trials", default=200)
    n_trials = max(50, n_trials)
    print()

    # ── Step 8: Optimize weights ──────────────────────────────────
    print("[8] Optimizing federation weights...")
    result = fed.fit(
        cal_structures=cal_structures,
        cal_ef=cal_ef,
        cal_bg=cal_bg,
        n_trials=n_trials,
    )

    print("\n  Optimal weights:")
    print_weight_table(
        fed.team_names,
        fed.weights_ef,
        fed.weights_bg,
    )

    # ── Step 9: Generate predictions ─────────────────────────────
    predictions = None
    if ask_yn("[9] Generate test predictions?"):
        print("  Running federated inference...")
        predictions = fed.predict(test_structures, team_name="CataLIST_federation")
        print(f"  {len(predictions)} predictions generated.")
        print()

        # ── Step 10: Output path ──────────────────────────────────
        default_out = str(
            ROOT / "submissions" / "CataLIST" / "predictions_federated.json"
        )
        out_path = ask("[10] Output path for predictions JSON", default=default_out)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        submission = {
            "team_name": "CataLIST",
            "model_id": "CataLIST_federation_v1",
            "matfed_api_version": "1.0",
            "federation_weights": {
                "weights_ef": dict(zip(fed.team_names, fed.weights_ef)),
                "weights_bg": dict(zip(fed.team_names, fed.weights_bg)),
            },
            "predictions": [
                {
                    "material_id": test_ids[i],
                    "formation_energy_per_atom": predictions[i]["formation_energy_per_atom"],
                    "band_gap": predictions[i]["band_gap"],
                }
                for i in range(len(test_ids))
            ],
        }
        with open(out_path, "w") as f:
            json.dump(submission, f, indent=2)
        print(f"  Saved to {out_path}")

        # Save weights separately too
        weights_path = ROOT / "models" / "federation" / "sprint_weights.json"
        fed.save_weights(weights_path)

    # ── Step 11: Final summary ────────────────────────────────────
    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print("  FEDERATION SPRINT — COMPLETE")
    print("=" * 60)
    print(f"  Calibration score : {result['calibration_score']:.2f} / 40")
    print(f"  EF MAE            : {result['federated_mae_ef']:.4f} eV/atom")
    print(f"  BG MAE            : {result['federated_mae_bg']:.4f} eV")
    print(f"  Teams federated   : {len(fed.team_names)}")
    print()
    print("  Weight table:")
    print_weight_table(fed.team_names, fed.weights_ef, fed.weights_bg)
    if predictions is not None:
        print(f"  Predictions saved : {out_path}")
    print(f"  Time elapsed      : {elapsed/60:.1f} min")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
