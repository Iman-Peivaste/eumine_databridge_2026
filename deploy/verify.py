"""
CataLIST Stage 2 — Health Check Script

Tests every component of the pipeline end-to-end.
Run before the federation sprint to confirm everything works.

Usage:
    python deploy/verify.py --model_path models/full_retrain
    python deploy/verify.py --model_path models/full_retrain --cpu_only
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

MATFED = ROOT.parent / "hackathon_ref" / "matfed-api-template"
if MATFED.exists():
    sys.path.insert(0, str(MATFED))


def check(label: str, fn, *args, **kwargs):
    """Run a check and print pass/fail."""
    print(f"  {'Checking':<30}", end="", flush=True)
    print(f" {label:<40}", end="", flush=True)
    try:
        t0 = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f" PASS  ({elapsed:.1f}s)")
        return result
    except Exception as e:
        print(f" FAIL")
        print(f"    Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(ROOT / "models" / "full_retrain"),
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force CPU inference",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    cpu_only = args.cpu_only

    print("\n" + "="*65)
    print("  CataLIST Stage 2 — Health Check")
    print(f"  Model path : {model_path}")
    print(f"  CPU only   : {cpu_only}")
    print("="*65)

    results = {}

    # ── 1. Core imports ───────────────────────────────────────────
    print("\n[1] Core imports")

    def import_torch():
        import torch
        return torch.__version__

    def import_pymatgen():
        from pymatgen.core import Structure
        return "ok"

    def import_alignn():
        import alignn
        return alignn.__version__

    def import_mace():
        import mace
        return "ok"

    def import_matfed():
        from matfed_api.predictor import MatFedPredictor
        return "ok"

    results["torch"] = check("torch", import_torch)
    results["pymatgen"] = check("pymatgen", import_pymatgen)
    results["alignn"] = check("alignn", import_alignn)
    results["mace"] = check("mace", import_mace)
    results["matfed_api"] = check("matfed_api", import_matfed)

    # ── 2. GPU / CPU ──────────────────────────────────────────────
    print("\n[2] Hardware")

    def check_gpu():
        import torch
        if cpu_only:
            return "CPU mode (forced)"
        if torch.cuda.is_available():
            return f"CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}"
        return "No GPU — CPU fallback active"

    results["hardware"] = check("GPU/CPU", check_gpu)

    # ── 3. Model artifacts ────────────────────────────────────────
    print("\n[3] Model artifacts")

    def check_artifacts():
        required = [
            "alignn_ef_full/best_model.pt",
            "alignn_bg_full/best_model.pt",
            "ensemble_weights.json",
            "calibration/",
            "mace_artifacts/",
        ]
        missing = []
        for rel in required:
            if not (model_path / rel).exists():
                missing.append(rel)
        if missing:
            raise FileNotFoundError(f"Missing: {missing}")
        return f"{len(required)} artifacts found"

    results["artifacts"] = check("artifacts present", check_artifacts)

    # ── 4. Load predictor ─────────────────────────────────────────
    print("\n[4] Predictor loading")

    predictor = None

    def load_predictor():
        from eumine_databridge.matfed.predictor import LISTEuMINePredictor
        p = LISTEuMINePredictor()
        p.load_model(str(model_path))
        return p

    predictor = check("load_model()", load_predictor)
    results["load"] = predictor is not None

    # ── 5. describe() ─────────────────────────────────────────────
    print("\n[5] Interface compliance")

    def check_describe():
        d = predictor.describe()
        required = ["team_name", "model_type", "api_version", "data_sources"]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"describe() missing keys: {missing}")
        return d["team_name"]

    if predictor:
        results["describe"] = check("describe()", check_describe)

    # ── 6. predict() on sample CIFs ──────────────────────────────
    print("\n[6] End-to-end prediction")

    sample_dir = MATFED / "tests" / "sample_structures"
    if not sample_dir.exists():
        sample_dir = ROOT / "hackathon_ref" / "matfed-api-template" \
                     / "tests" / "sample_structures"

    def run_prediction():
        from pymatgen.core import Structure
        import numpy as np

        cif_files = sorted(sample_dir.glob("*.cif"))[:3]
        if not cif_files:
            raise FileNotFoundError(f"No CIFs in {sample_dir}")

        structures = [Structure.from_file(str(c)) for c in cif_files]
        preds = predictor.predict(structures)

        assert len(preds) == len(structures)
        for p in preds:
            assert "formation_energy_per_atom" in p
            assert "band_gap" in p
            assert isinstance(p["formation_energy_per_atom"], float)
            assert isinstance(p["band_gap"], float)
            assert p["band_gap"] >= 0.0

        ef_vals = [p["formation_energy_per_atom"] for p in preds]
        bg_vals = [p["band_gap"] for p in preds]
        return (
            f"{len(preds)} structures | "
            f"EF range [{min(ef_vals):.3f}, {max(ef_vals):.3f}] | "
            f"BG range [{min(bg_vals):.3f}, {max(bg_vals):.3f}]"
        )

    if predictor:
        results["predict"] = check("predict() 3 structures", run_prediction)

    # ── 7. Federation engine ──────────────────────────────────────
    print("\n[7] Federation engine")

    def check_federation():
        from eumine_databridge.matfed.federation import FederatedEnsemble
        fed = FederatedEnsemble()
        fed.add_predictor(predictor, "CataLIST_test")
        assert len(fed.predictors) == 1
        return "FederatedEnsemble importable"

    if predictor:
        results["federation"] = check("FederatedEnsemble", check_federation)

    # ── 8. Summary ────────────────────────────────────────────────
    print("\n" + "="*65)
    passed = sum(1 for v in results.values() if v is not None and v is not False)
    total = len(results)
    print(f"  RESULT: {passed}/{total} checks passed")

    if passed == total:
        print("  STATUS: READY FOR STAGE 2 SPRINT")
    else:
        print("  STATUS: ISSUES FOUND — fix before Cluj")
        failed = [k for k, v in results.items()
                  if v is None or v is False]
        print(f"  Failed: {failed}")

    print("="*65 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
