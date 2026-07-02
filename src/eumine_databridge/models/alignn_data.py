"""
Convert pymatgen Structures to ALIGNN graph dataset format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from tqdm import tqdm

ALIGNN_TARGET_KEY = "target"
ALIGNN_ID_KEY = "id"


def structures_to_alignn_dataset(
    structures: List[Structure],
    targets: Optional[List[float]],
    material_ids: List[str],
    cutoff: float = 8.0,
    max_neighbors: int = 12,
) -> List[dict]:
    """
    Convert pymatgen Structures to ALIGNN dataset dicts.

    Each entry: {"atoms": jarvis.Atoms, "id": str, "target": float}
    """
    adaptor = JarvisAtomsAdaptor()
    dataset = []
    failed = []

    for i, structure in enumerate(tqdm(structures, desc="Building ALIGNN graphs")):
        mat_id = material_ids[i]
        target = targets[i] if targets is not None else 0.0
        try:
            jarvis_atoms = adaptor.get_atoms(structure)
            dataset.append({
                "atoms": jarvis_atoms.to_dict(),
                ALIGNN_ID_KEY: mat_id,
                ALIGNN_TARGET_KEY: float(target),
            })
        except Exception as e:
            print(f"  WARNING: failed to convert {mat_id}: {e}")
            failed.append(mat_id)

    if failed:
        print(f"  {len(failed)} structures failed conversion: {failed[:5]}...")
    print(
        f"ALIGNN dataset built: {len(dataset)} structures "
        f"({len(failed)} failed)"
    )
    return dataset


def prepare_alignn_data_splits(
    train_structures: List[Structure],
    train_targets: List[float],
    train_ids: List[str],
    val_structures: List[Structure],
    val_targets: List[float],
    val_ids: List[str],
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    output_dir: Optional[Path] = None,
) -> Tuple[List[dict], List[dict]]:
    """Build ALIGNN-compatible train and val dataset lists."""
    print("\nPreparing ALIGNN data")
    print(f"  Train: {len(train_structures)} structures")
    print(f"  Val  : {len(val_structures)} structures")

    train_dataset = structures_to_alignn_dataset(
        train_structures, train_targets, train_ids,
        cutoff=cutoff, max_neighbors=max_neighbors,
    )
    val_dataset = structures_to_alignn_dataset(
        val_structures, val_targets, val_ids,
        cutoff=cutoff, max_neighbors=max_neighbors,
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_alignn_dataset(train_dataset, output_dir / "train_data.json")
        _save_alignn_dataset(val_dataset, output_dir / "val_data.json")
        print(f"  ALIGNN dataset metadata saved to {output_dir}")

    return train_dataset, val_dataset


def _save_alignn_dataset(dataset: List[dict], path: Path):
    """Save dataset IDs and targets (not full Atoms) for inspection."""
    records = [
        {"id": item[ALIGNN_ID_KEY], ALIGNN_TARGET_KEY: item[ALIGNN_TARGET_KEY]}
        for item in dataset
    ]
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def compute_target_statistics(targets: List[float]) -> dict:
    """Compute mean and std for target normalization."""
    arr = np.array(targets)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
