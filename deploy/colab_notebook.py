"""
CataLIST Stage 2 — Google Colab Setup & Inference Notebook

This script sets up the CataLIST model for inference in Google Colab.
No conda needed — uses pip only.

Usage in Colab:
1. Copy this entire cell into Colab
2. Run it (20-30 seconds setup)
3. Use the predictor in subsequent cells

Example usage:
    structures = [Structure.from_file("my_structure.cif")]
    predictions = predictor.predict(structures)
"""

import sys
import subprocess
from pathlib import Path

def run_cmd(cmd: str, show_output: bool = False) -> str:
    """Run shell command and return output."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
    )
    if show_output or result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout

def setup_colab():
    """Complete setup for Colab."""
    print("=" * 70)
    print("CataLIST Stage 2 — Google Colab Setup")
    print("=" * 70)

    # Step 1: Check if in Colab
    print("\n[1/5] Checking environment...")
    try:
        import google.colab
        print("  ✓ Running in Google Colab")
        in_colab = True
    except ImportError:
        print("  ⚠ Not in Google Colab (will still work locally)")
        in_colab = False

    # Step 2: Clone repo if not present
    print("\n[2/5] Setting up repository...")
    repo_path = Path("/content/eumine_databridge_2026")
    if repo_path.exists():
        print(f"  ✓ Repo exists at {repo_path}")
    else:
        print("  Cloning repository...")
        run_cmd(
            "git clone https://github.com/Iman-Peivaste/eumine_databridge_2026.git /content/eumine_databridge_2026",
            show_output=False,
        )
        print("  ✓ Repository cloned")

    # Step 3: Add to path
    print("\n[3/5] Installing dependencies...")
    sys.path.insert(0, str(repo_path / "src"))
    matfed = repo_path.parent / "hackathon_ref" / "matfed-api-template"
    if matfed.exists():
        sys.path.insert(0, str(matfed))

    # Install key packages (skip if already present)
    packages = [
        "pymatgen==2025.6.14",
        "torch==2.4.0",
        "torch-geometric",
        "alignn==2026.4.2",
        "mace-torch==0.3.15",
        "optuna>=3.6",
        "mp-api>=0.45.9",
    ]

    print("  Installing via pip...")
    for pkg in packages:
        try:
            # Check if already installed
            pkg_name = pkg.split("==")[0].split(">=")[0]
            __import__(pkg_name.replace("-", "_"))
            print(f"    ✓ {pkg_name} (already installed)")
        except ImportError:
            print(f"    Installing {pkg}...")
            run_cmd(f"pip install -q {pkg}", show_output=False)
            print(f"      ✓ {pkg_name} installed")

    # Step 4: Download model artifacts if needed
    print("\n[4/5] Checking model artifacts...")
    model_dir = repo_path / "models" / "combined_retrain"
    required_files = [
        model_dir / "alignn_ef_combined" / "best_model.pt",
        model_dir / "alignn_bg_combined" / "best_model.pt",
        model_dir / "ensemble_weights.json",
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print(f"  ⚠ Missing model files: {len(missing)} files")
        print("  Note: Model artifacts (14.3 GB) need to be downloaded.")
        print("  For Colab testing, use a lightweight model or dummy structures.")
        print("  Contact: euminecost@gmail.com for model artifact links.")
    else:
        print(f"  ✓ All model artifacts present ({sum(f.stat().st_size for f in required_files if f.exists()) / 1e9:.1f} GB)")

    # Step 5: Load predictor
    print("\n[5/5] Loading predictor...")
    try:
        from eumine_databridge.matfed.predictor import LISTEuMINePredictor
        predictor = LISTEuMINePredictor()

        # Check if model exists
        if model_dir.exists() and (model_dir / "alignn_ef_combined" / "best_model.pt").exists():
            predictor.load_model(str(model_dir))
            print("  ✓ Predictor loaded (with trained weights)")
        else:
            print("  ⚠ Model artifacts not available")
            print("  Predictor class loaded, but weights not available.")
            print("  (This is OK for testing the interface)")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        predictor = None

    print("\n" + "=" * 70)
    if predictor:
        print("✓ SETUP COMPLETE — Ready to predict!")
    else:
        print("⚠ Setup partial — predictor interface available")
    print("=" * 70)

    return {
        "repo_path": repo_path,
        "in_colab": in_colab,
        "predictor": predictor,
        "model_dir": model_dir,
    }


def example_predict(predictor):
    """Example: predict on dummy structures."""
    if predictor is None or not predictor._loaded:
        print("Predictor not loaded. Skipping example.")
        return

    print("\n" + "=" * 70)
    print("Example: Predict on 2 structures")
    print("=" * 70)

    from pymatgen.core import Structure, Lattice
    import numpy as np

    # Create two dummy structures
    structures = []

    # Si (diamond cubic)
    lat = Lattice.cubic(5.4)
    si = Structure(lat, ["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    structures.append(si)

    # NaCl (rock salt)
    lat = Lattice.cubic(5.64)
    nacl = Structure(
        lat,
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    structures.append(nacl)

    print("\nInput structures:")
    for i, s in enumerate(structures):
        print(f"  {i+1}. {s.composition.reduced_formula} ({len(s)} atoms)")

    print("\nPredicting...")
    predictions = predictor.predict(structures)

    print("\nResults:")
    for i, pred in enumerate(predictions):
        print(f"\n  Structure {i+1}:")
        print(f"    Formation energy: {pred['formation_energy_per_atom']:.4f} eV/atom")
        print(f"    Band gap:         {pred['band_gap']:.4f} eV")
        print(f"    Uncertainty EF:   ±{pred['uncertainty_ef']:.4f}")
        print(f"    Uncertainty BG:   ±{pred['uncertainty_bg']:.4f}")
        print(f"    Model ID:         {pred['model_id']}")


# Run setup
if __name__ == "__main__":
    config = setup_colab()

    # Show how to use
    print("\n" + "=" * 70)
    print("HOW TO USE IN YOUR NOTEBOOK")
    print("=" * 70)
    print("""
# In a new Colab cell:

from pymatgen.core import Structure

# Load your structure
structure = Structure.from_file("my_structure.cif")

# Predict
predictions = predictor.predict([structure])
pred = predictions[0]

print(f"Formation energy: {pred['formation_energy_per_atom']:.4f} eV/atom")
print(f"Band gap: {pred['band_gap']:.4f} eV")

# For federation (in Cluj):
from eumine_databridge.matfed.federation import FederatedEnsemble

fed = FederatedEnsemble()
fed.add_predictor(predictor, "CataLIST")
# ... add partner predictors ...
""")

    # Run example if predictor loaded
    if config["predictor"] and config["predictor"]._loaded:
        example_predict(config["predictor"])
