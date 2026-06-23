"""
ALIGNN fine-tuning script — EF and BG.

Run:
    python scripts/train_alignn.py --target ef
    python scripts/train_alignn.py --target bg
    python scripts/train_alignn.py --target both
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np

from eumine_databridge.data.loader import BridgeDataset
from eumine_databridge.models.alignn_config import get_ef_config, get_bg_config
from eumine_databridge.models.alignn_data import (
    prepare_alignn_data_splits,
    compute_target_statistics,
)
from eumine_databridge.models.alignn_model import ALIGNNFineTuner
from eumine_databridge.utils.metrics import compute_full_score, compute_metrics

DATA = ROOT / "data"
RAW = DATA / "raw"


def load_splits():
    train_ds = BridgeDataset(
        csv_path=RAW / "bridge_dataset_train.csv",
        structures_dir=RAW / "train_structures",
        split="train",
    )
    val_ds = BridgeDataset(
        csv_path=RAW / "bridge_dataset_val.csv",
        structures_dir=RAW / "val_structures",
        split="val",
    )
    return train_ds, val_ds


def _structures_and_targets(dataset, prop: str):
    targets, ids = dataset.get_targets(prop)
    id_set = set(ids)
    structures = [
        e.structure for e in dataset.entries if e.material_id in id_set
    ]
    # Preserve order matching targets/ids
    id_to_struct = {
        e.material_id: e.structure for e in dataset.entries if e.material_id in id_set
    }
    structures = [id_to_struct[i] for i in ids]
    return structures, targets, ids


def train_single(target: str):
    print(f"\n{'#'*60}")
    print(f"# ALIGNN Fine-Tuning: {target.upper()}")
    print(f"{'#'*60}")

    train_ds, val_ds = load_splits()
    prop = (
        "formation_energy_per_atom"
        if target == "ef"
        else "band_gap"
    )

    train_structures, train_targets, train_ids = _structures_and_targets(
        train_ds, prop
    )
    val_structures, val_targets, val_ids = _structures_and_targets(val_ds, prop)

    print(f"\nTarget: {prop}")
    print(f"Train samples: {len(train_targets)}")
    print(f"Val   samples: {len(val_targets)}")

    config = get_ef_config() if target == "ef" else get_bg_config()

    train_dataset, val_dataset = prepare_alignn_data_splits(
        train_structures=train_structures,
        train_targets=train_targets,
        train_ids=train_ids,
        val_structures=val_structures,
        val_targets=val_targets,
        val_ids=val_ids,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        output_dir=ROOT / "data" / "processed" / f"alignn_{target}",
    )

    stats = compute_target_statistics(train_targets)
    print(f"\nTarget statistics:")
    print(f"  Mean : {stats['mean']:.4f}")
    print(f"  Std  : {stats['std']:.4f}")
    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    trainer = ALIGNNFineTuner(config)
    trainer.setup(train_dataset, val_dataset, target_stats=stats)
    trainer.save_config()
    trainer.train()

    trainer.load_best_model()
    val_mae, val_preds, val_tgts = trainer._validate()

    baseline = 0.2378 if target == "ef" else 0.6414
    metrics = compute_metrics(
        np.array(val_preds),
        np.array(val_tgts),
        property_name=prop,
        baseline_mae=baseline,
    )

    print(f"\n{'='*50}")
    print(f"FINAL VALIDATION RESULTS — {prop.upper()}")
    print(f"{'='*50}")
    print(f"  MAE              : {metrics['mae']:.4f}")
    print(f"  RMSE             : {metrics['rmse']:.4f}")
    print(f"  R²               : {metrics['r2']:.4f}")
    print(f"  Pearson r        : {metrics['pearson_r']:.4f}")
    print(f"  Hackathon score  : {metrics['hackathon_score']:.2f} / 20")
    print(f"  Beats baseline   : {metrics['beats_baseline']}")
    print(f"  Baseline MAE     : {baseline}")
    print(f"{'='*50}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train ALIGNN for EuMINe hackathon"
    )
    parser.add_argument(
        "--target",
        choices=["ef", "bg", "both"],
        default="both",
        help="Which property to train (ef, bg, or both)",
    )
    args = parser.parse_args()

    results = {}
    if args.target in ("ef", "both"):
        results["ef"] = train_single("ef")
    if args.target in ("bg", "both"):
        results["bg"] = train_single("bg")

    if len(results) == 2:
        final = compute_full_score(
            mae_ef=results["ef"]["mae"],
            mae_bg=results["bg"]["mae"],
        )
        print(f"\n{'#'*60}")
        print("# COMBINED HACKATHON SCORE")
        print(f"{'#'*60}")
        print(f"  EF score         : {final['score_ef']:.2f} / 20")
        print(f"  BG score         : {final['score_bg']:.2f} / 20")
        print(f"  TOTAL PERFORMANCE: {final['total_performance_score']:.2f} / 40")
        print(f"  Beats baseline EF: {final['beats_baseline_ef']}")
        print(f"  Beats baseline BG: {final['beats_baseline_bg']}")
        print(f"  Qualifies Stage 2: {final['qualifies_for_stage2']}")
        print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
