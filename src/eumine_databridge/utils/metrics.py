"""
Evaluation metrics and hackathon scoring formula.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

BASELINE_MAE_EF = 0.2378   # eV/atom
BASELINE_MAE_BG = 0.6414   # eV


def score_property(mae: float, baseline_mae: float) -> float:
    """
    Hackathon scoring formula for a single property.
    Maximum 20 points per property, total 40 points for performance.
    """
    if mae < baseline_mae:
        return 10.0 + 10.0 * (baseline_mae - mae) / (baseline_mae - 0.01)
    else:
        return max(0.0, 10.0 * (1.0 - (mae - baseline_mae) / baseline_mae))


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    property_name: str = "property",
    baseline_mae: Optional[float] = None,
) -> Dict:
    """Compute full evaluation metrics for one property."""
    from scipy import stats

    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    pearson_r, _ = stats.pearsonr(predictions, targets)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    result = {
        "property": property_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": float(pearson_r),
        "n_samples": len(predictions),
    }
    if baseline_mae is not None:
        hack_score = score_property(mae, baseline_mae)
        result["hackathon_score"] = hack_score
        result["beats_baseline"] = mae < baseline_mae
        result["baseline_mae"] = baseline_mae
    return result


def compute_full_score(mae_ef: float, mae_bg: float) -> Dict:
    """Compute the full hackathon performance score (0-40 pts)."""
    score_ef = score_property(mae_ef, BASELINE_MAE_EF)
    score_bg = score_property(mae_bg, BASELINE_MAE_BG)
    total = score_ef + score_bg
    return {
        "score_ef": score_ef,
        "score_bg": score_bg,
        "total_performance_score": total,
        "beats_baseline_ef": mae_ef < BASELINE_MAE_EF,
        "beats_baseline_bg": mae_bg < BASELINE_MAE_BG,
        "qualifies_for_stage2": (
            mae_ef < BASELINE_MAE_EF or mae_bg < BASELINE_MAE_BG
        ),
    }


def per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    class_name: str = "class",
) -> Dict:
    """Compute MAE broken down by material class."""
    results = {}
    unique_classes = np.unique(labels)
    for cls in unique_classes:
        mask = labels == cls
        if mask.sum() < 2:
            continue
        cls_mae = float(np.mean(np.abs(predictions[mask] - targets[mask])))
        results[str(cls)] = {
            "mae": cls_mae,
            "n": int(mask.sum()),
        }
    return results
