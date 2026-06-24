"""
Generate test predictions from saved augmented model artifacts.
Run after retrain_augmented.py has completed successfully.
"""
import sys, json
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from pymatgen.core import Structure
from eumine_databridge.models.alignn_config import get_ef_config, get_bg_config
from eumine_databridge.models.alignn_model import ALIGNNFineTuner
from eumine_databridge.models.mace_model import MACEPredictor
from eumine_databridge.models.ensemble import WeightedEnsemble, CalibrationLayer

AUG_MODELS = ROOT / "models" / "augmented_retrain"
TEST_DIR   = ROOT / "data" / "raw" / "test_structures"

print("Loading test CIFs...")
test_cifs = sorted(TEST_DIR.glob("*.cif"))
test_ids, test_structures = [], []
for cif in tqdm(test_cifs, desc="Test CIFs"):
    try:
        test_structures.append(Structure.from_file(str(cif)))
        test_ids.append(cif.stem)
    except Exception:
        pass
print(f"  Loaded {len(test_structures)} test structures")

print("Loading ALIGNN EF...")
ef_cfg = get_ef_config()
ef_cfg.output_dir = AUG_MODELS / "alignn_ef_aug"
ef_trainer = ALIGNNFineTuner(ef_cfg)
ef_trainer._init_model_only()
ef_trainer.load_best_model()

print("Loading ALIGNN BG...")
bg_cfg = get_bg_config()
bg_cfg.output_dir = AUG_MODELS / "alignn_bg_aug"
bg_cfg.alignn_layers = 6
bg_cfg.hidden_features = 256
bg_trainer = ALIGNNFineTuner(bg_cfg)
bg_trainer._init_model_only()
bg_trainer.load_best_model()

print("ALIGNN predictions...")
alignn_ef_test = np.array(ef_trainer.predict(test_structures, test_ids))
alignn_bg_test = np.array(bg_trainer.predict(test_structures, test_ids))

print("Loading MACE...")
mace = MACEPredictor(model_name="medium")
mace.load_model()
mace.load_artifacts(ROOT / "models" / "full_retrain" / "mace_artifacts")
mace_ef_test = mace.predict_ef(test_structures)
mace_bg_test = mace.predict_bg(test_structures)
mace_ef_test = np.where(np.isnan(mace_ef_test), alignn_ef_test, mace_ef_test)
mace_bg_test = np.where(np.isnan(mace_bg_test), alignn_bg_test, mace_bg_test)

print("Ensemble + calibration...")
ensemble = WeightedEnsemble()
ensemble.load(AUG_MODELS / "ensemble_weights.json")
calibrator = CalibrationLayer()
calibrator.load(AUG_MODELS / "calibration")

ef_ens, bg_ens = ensemble.predict(
    alignn_ef_test, mace_ef_test,
    alignn_bg_test, mace_bg_test,
)
ef_final, bg_final = calibrator.calibrate(ef_ens, bg_ens)
bg_final = np.clip(bg_final, 0.0, None)

pred_path = ROOT / "submissions" / "CataLIST" / "predictions_test.json"
submission = {
    "team_name": "CataLIST",
    "model_id": "ALIGNN_MACE_ensemble_v3_augmented",
    "matfed_api_version": "1.0",
    "predictions": [
        {
            "material_id": test_ids[i],
            "formation_energy_per_atom": float(ef_final[i]),
            "band_gap": float(bg_final[i]),
        }
        for i in range(len(test_ids))
    ],
}
with open(pred_path, "w") as f:
    json.dump(submission, f, indent=2)
print(f"Saved {len(test_ids)} predictions to {pred_path}")
print("DONE")
