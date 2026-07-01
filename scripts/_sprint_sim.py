"""
Runs sprint_launcher.py logic directly with pre-filled answers,
so we don't need an interactive terminal.
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

MATFED = ROOT.parent / "hackathon_ref" / "matfed-api-template"
if MATFED.exists():
    sys.path.insert(0, str(MATFED))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import json
import importlib
import numpy as np

# ── partner sys.path injections ───────────────────────────────────
for extra in [
    "/home/su_peivaste/EuMINe/takeme2romania_repo/submissions/TakeMe2Romania",
    "/home/su_peivaste/EuMINe/prophx_repo/src",
    "/home/su_peivaste/EuMINe/prophx_repo/submissions",
]:
    if extra not in sys.path:
        sys.path.insert(0, extra)
        print(f"  sys.path += {extra}")

t_start = time.time()

# ── load CataLIST ─────────────────────────────────────────────────
print("\n[1] Loading CataLIST...")
from eumine_databridge.matfed.predictor import LISTEuMINePredictor
our = LISTEuMINePredictor()
our.load_model(str(ROOT / "models" / "combined_retrain"))
print("  CataLIST ready.")

# ── load TakeMe2Romania ───────────────────────────────────────────
print("\n[2] Loading TakeMe2Romania...")
import sklearn
print(f"  sklearn version: {sklearn.__version__}")
from matfed_predictor import TakeMe2RomaniaPredictor
t2r = TakeMe2RomaniaPredictor()
t2r.load_model("/home/su_peivaste/EuMINe/takeme2romania_repo/submissions/TakeMe2Romania")
print("  TakeMe2Romania ready.")
print(f"  describe: {t2r.describe()}")

# dummy-prediction check
from pymatgen.core import Structure, Lattice
dummy = Structure(Lattice.cubic(4.0), ["Si","Si"], [[0,0,0],[0.5,0.5,0.5]])
p = t2r.predict([dummy])[0]
print(f"  Dummy check: EF={p['formation_energy_per_atom']:.4f}, BG={p['band_gap']:.4f}")
is_dummy = abs(p["formation_energy_per_atom"] - (-1.0)) < 1e-9 and abs(p["band_gap"] - 1.0) < 1e-9
if is_dummy:
    print("  WARNING: TakeMe2Romania returning dummy predictions!")

# ── load ProphX ───────────────────────────────────────────────────
print("\n[3] Loading ProphX...")
from prophx_v2_scaled.ProphX_Predictor import ProphXPredictor
phx = ProphXPredictor()
phx.load_model("/home/su_peivaste/EuMINe/prophx_repo/submissions/models")
print(f"  ProphX ready. model_ef loaded: {phx.model_ef is not None}")
print(f"  describe: {phx.describe()['team_name']}, model_id: {phx.describe()['model_id']}")

p2 = phx.predict([dummy])[0]
print(f"  Dummy check: EF={p2['formation_energy_per_atom']:.4f}, BG={p2['band_gap']:.4f}")
is_dummy2 = abs(p2["formation_energy_per_atom"] - (-1.0)) < 1e-9 and abs(p2["band_gap"] - 1.0) < 1e-9
if is_dummy2:
    print("  WARNING: ProphX returning dummy predictions — model weights missing!")

# ── build federation ──────────────────────────────────────────────
from eumine_databridge.matfed.federation import FederatedEnsemble
fed = FederatedEnsemble()
fed.add_predictor(our, "CataLIST")
fed.add_predictor(t2r, "TakeMe2Romania")
if not is_dummy2:
    fed.add_predictor(phx, "ProphX")
else:
    print("  ProphX excluded from federation (dummy predictions).")

# ── calibration set: first 80 val structures ──────────────────────
print("\n[4] Loading calibration set (first 80 val structures)...")
import pandas as pd
from pymatgen.core import Structure as S

val_csv = ROOT / "data" / "raw" / "bridge_dataset_val.csv"
val_dir = ROOT / "data" / "raw" / "val_structures"
df = pd.read_csv(val_csv).set_index("material_id")

cif_files = sorted(val_dir.glob("*.cif"))
all_structs, all_ids = [], []
for cif in cif_files:
    try:
        all_structs.append(S.from_file(str(cif)))
        all_ids.append(cif.stem)
    except Exception:
        pass

# Match to labels
matched = [(s, mid) for s, mid in zip(all_structs, all_ids) if mid in df.index]
cal_structs = [x[0] for x in matched[:80]]
cal_ids     = [x[1] for x in matched[:80]]
cal_ef = [float(df.loc[mid, "formation_energy_per_atom_label"]) for mid in cal_ids]
cal_bg = [float(df.loc[mid, "band_gap_label"]) for mid in cal_ids]
print(f"  Calibration: {len(cal_structs)} structures")

# ── test set: remaining val structures (rows 80+) ────────────────
test_structs = [x[0] for x in matched[80:]]
test_ids_sim = [x[1] for x in matched[80:]]
test_ef_true = np.array([float(df.loc[mid, "formation_energy_per_atom_label"]) for mid in test_ids_sim])
test_bg_true = np.array([float(df.loc[mid, "band_gap_label"]) for mid in test_ids_sim])
print(f"  Test proxy:  {len(test_structs)} structures")

# ── fit ───────────────────────────────────────────────────────────
print("\n[5] Running federation fit (150 Optuna trials)...")
result = fed.fit(
    cal_structures=cal_structs,
    cal_ef=cal_ef,
    cal_bg=cal_bg,
    n_trials=150,
)

print("\n  Weight table:")
print(f"  {'Team':<25} {'EF weight':>10} {'BG weight':>10}")
print(f"  {'-'*47}")
for name, wef, wbg in zip(fed.team_names, fed.weights_ef, fed.weights_bg):
    print(f"  {name:<25} {wef:>10.3f} {wbg:>10.3f}")

# ── test predictions ──────────────────────────────────────────────
print("\n[6] Generating test predictions...")
preds = fed.predict(test_structs, team_name="CataLIST_federation")
ef_fed = np.array([p["formation_energy_per_atom"] for p in preds])
bg_fed = np.array([p["band_gap"] for p in preds])

# ── score ─────────────────────────────────────────────────────────
from eumine_databridge.utils.metrics import compute_full_score

mae_ef_fed = float(np.mean(np.abs(ef_fed - test_ef_true)))
mae_bg_fed = float(np.mean(np.abs(bg_fed - test_bg_true)))
score_fed = compute_full_score(mae_ef_fed, mae_bg_fed)

# CataLIST solo score for comparison
solo_preds = our.predict(test_structs)
ef_solo = np.array([p["formation_energy_per_atom"] for p in solo_preds])
bg_solo = np.array([p["band_gap"] for p in solo_preds])
mae_ef_solo = float(np.mean(np.abs(ef_solo - test_ef_true)))
mae_bg_solo = float(np.mean(np.abs(bg_solo - test_bg_true)))
score_solo = compute_full_score(mae_ef_solo, mae_bg_solo)

# ── save predictions JSON ─────────────────────────────────────────
out_path = ROOT / "submissions" / "CataLIST" / "predictions_dry_run.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
submission = {
    "team_name": "CataLIST",
    "model_id": "CataLIST_federation_dry_run_v1",
    "matfed_api_version": "1.0",
    "federation_weights": {
        "weights_ef": dict(zip(fed.team_names, fed.weights_ef)),
        "weights_bg": dict(zip(fed.team_names, fed.weights_bg)),
    },
    "predictions": [
        {
            "material_id": test_ids_sim[i],
            "formation_energy_per_atom": float(ef_fed[i]),
            "band_gap": float(bg_fed[i]),
        }
        for i in range(len(test_ids_sim))
    ],
}
with open(out_path, "w") as f:
    json.dump(submission, f, indent=2)

fed.save_weights(ROOT / "models" / "federation" / "dry_run_weights.json")

# ── final report ──────────────────────────────────────────────────
elapsed = time.time() - t_start
print("\n" + "="*60)
print("  S2-6 FULL DRY RUN — RESULTS")
print("="*60)
print(f"\n  Test proxy: {len(test_structs)} structures (val rows 80–{len(matched)})")
print()
print(f"  {'':25} {'EF MAE':>10} {'BG MAE':>10} {'Score':>8}")
print(f"  {'-'*55}")
print(f"  {'CataLIST solo':25} {mae_ef_solo:>10.4f} {mae_bg_solo:>10.4f} {score_solo['total_performance_score']:>8.2f}/40")
print(f"  {'Federation':25} {mae_ef_fed:>10.4f} {mae_bg_fed:>10.4f} {score_fed['total_performance_score']:>8.2f}/40")
print()
print(f"  Teams in federation : {', '.join(fed.team_names)}")
print(f"  Calibration score   : {result['calibration_score']:.2f}/40")
print(f"  Predictions saved   : {out_path}")
print(f"  Time elapsed        : {elapsed/60:.1f} min")
print("="*60)
