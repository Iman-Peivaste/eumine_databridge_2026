"""
Explore and visualize the EuMINe Bridge Dataset.

Run:
    python scripts/visualize_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "data_exploration"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW / "bridge_dataset_train.csv")
    val = pd.read_csv(RAW / "bridge_dataset_val.csv")
    combined = pd.concat([train.assign(split="train"), val.assign(split="val")])

    c_mp, c_jarvis = "#059669", "#dc2626"

    # 1. Property distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pairs = [
        (
            "formation_energy_per_atom_mp",
            "formation_energy_per_atom_jarvis",
            "Formation energy (eV/atom)",
            "EF",
        ),
        ("band_gap_mp", "band_gap_jarvis", "Band gap (eV)", "BG"),
    ]
    for ax, mp_col, jar_col, xlab, title in zip(axes, *zip(*pairs)):
        ax.hist(
            combined[mp_col],
            bins=40,
            alpha=0.65,
            label="MP (official label)",
            color=c_mp,
            density=True,
        )
        ax.hist(
            combined[jar_col],
            bins=40,
            alpha=0.5,
            label="JARVIS",
            color=c_jarvis,
            density=True,
        )
        ax.set_xlabel(xlab)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title(f"{title}: MP vs JARVIS (n=850)")
    fig.tight_layout()
    fig.savefig(OUT / "01_property_distributions.png", dpi=150)
    plt.close()

    # 2. MP vs JARVIS scatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ef_mp = combined["formation_energy_per_atom_mp"]
    ef_jar = combined["formation_energy_per_atom_jarvis"]
    bg_mp = combined["band_gap_mp"]
    bg_jar = combined["band_gap_jarvis"]

    lim_ef = [
        min(ef_mp.min(), ef_jar.min()) - 0.2,
        max(ef_mp.max(), ef_jar.max()) + 0.2,
    ]
    axes[0].scatter(ef_jar, ef_mp, alpha=0.35, s=18, c="#2563eb")
    axes[0].plot(lim_ef, lim_ef, "k--", lw=1, label="y = x")
    axes[0].set_xlabel("JARVIS EF (eV/atom)")
    axes[0].set_ylabel("MP EF — official label (eV/atom)")
    axes[0].set_title("Formation energy")
    axes[0].legend()

    lim_bg = [0, max(bg_mp.max(), bg_jar.max()) + 0.5]
    axes[1].scatter(bg_jar, bg_mp, alpha=0.35, s=18, c="#2563eb")
    axes[1].plot(lim_bg, lim_bg, "k--", lw=1, label="y = x")
    axes[1].set_xlabel("JARVIS band gap (eV)")
    axes[1].set_ylabel("MP band gap — official label (eV)")
    axes[1].set_title("Band gap")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "02_mp_vs_jarvis_scatter.png", dpi=150)
    plt.close()

    # 3. Discrepancy
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ef_diff = ef_mp - ef_jar
    bg_diff = bg_mp - bg_jar
    for ax, diff, xlab, title in [
        (axes[0], ef_diff, "MP − JARVIS (eV/atom)", "EF discrepancy"),
        (axes[1], bg_diff, "MP − JARVIS (eV)", "Band gap discrepancy"),
    ]:
        ax.hist(diff, bins=40, color="#7c3aed", edgecolor="white")
        ax.axvline(diff.mean(), color="red", ls="--", label=f"mean = {diff.mean():.3f}")
        ax.set_xlabel(xlab)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "03_discrepancy_histograms.png", dpi=150)
    plt.close()

    # 4. Band-gap categories
    def bg_cat(bg: float) -> str:
        if bg == 0:
            return "Metal (BG=0)"
        if bg < 3:
            return "Semiconductor (0–3 eV)"
        return "Wide-gap (≥3 eV)"

    combined["bg_category"] = combined["band_gap_label"].apply(bg_cat)
    cat_order = [
        "Metal (BG=0)",
        "Semiconductor (0–3 eV)",
        "Wide-gap (≥3 eV)",
    ]
    counts = combined.groupby(["split", "bg_category"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=cat_order)
    ax = counts.plot(
        kind="bar",
        color=["#64748b", "#22c55e", "#3b82f6"],
        figsize=(8, 4.5),
    )
    ax.set_title("Electronic character by split")
    ax.set_ylabel("Count")
    ax.set_xlabel("Split")
    plt.xticks(rotation=0)
    plt.legend(title="Category", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(OUT / "04_band_gap_categories.png", dpi=150)
    plt.close()

    # 5. Composition and size
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ne_counts = combined["nelements"].value_counts().sort_index()
    axes[0].bar(ne_counts.index.astype(str), ne_counts.values, color=c_mp)
    axes[0].set_xlabel("Number of elements")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Chemical diversity (nelements)")
    axes[1].hist(combined["nsites"], bins=30, color="#2563eb", edgecolor="white")
    axes[1].set_xlabel("Atoms per unit cell (nsites)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Structure size distribution")
    fig.tight_layout()
    fig.savefig(OUT / "05_composition_and_size.png", dpi=150)
    plt.close()

    summary = (
        f"Bridge Dataset summary\n"
        f"Train: {len(train)} | Val: {len(val)} | Test: 150 (labels hidden)\n"
        f"Mean |MP−JARVIS| EF: {ef_diff.abs().mean():.4f} eV/atom\n"
        f"Mean |MP−JARVIS| BG: {bg_diff.abs().mean():.4f} eV\n"
    )
    (OUT / "summary.txt").write_text(summary)
    print(summary)
    print(f"Plots saved to {OUT}")


if __name__ == "__main__":
    main()
