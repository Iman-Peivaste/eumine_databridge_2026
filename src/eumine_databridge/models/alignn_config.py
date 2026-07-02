"""
ALIGNN training configurations for EF and BG prediction.
Tuned for RTX A4000 (16GB VRAM), Bridge Dataset size (~700 train samples).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# JARVIS figshare keys used by alignn.pretrained.get_figshare_model
PRETRAINED_MODEL_KEYS = {
    "formation_energy_peratom": "jv_formation_energy_peratom_alignn",
    "optb88vdw_bandgap": "jv_optb88vdw_bandgap_alignn",
}


@dataclass
class ALIGNNTrainConfig:
    """
    Full training configuration for one ALIGNN model.
    All hyperparameters documented with rationale.
    """

    # ── Target ────────────────────────────────────────────────────────────────
    target: str = "formation_energy_per_atom"
    # "formation_energy_per_atom" or "band_gap"

    # ── Graph construction ────────────────────────────────────────────────────
    cutoff: float = 8.0
    max_neighbors: int = 12
    atom_features: str = "cgcnn"

    # ── Model architecture ────────────────────────────────────────────────────
    alignn_layers: int = 4
    gcn_layers: int = 4
    hidden_features: int = 256
    output_features: int = 1

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    warmup_epochs: int = 10

    # ── Early stopping ────────────────────────────────────────────────────────
    patience: int = 50
    min_delta: float = 1e-4

    # ── Data ──────────────────────────────────────────────────────────────────
    val_split: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True

    # ── Pretrained weights ────────────────────────────────────────────────────
    use_pretrained: bool = True
    pretrained_target: str = "formation_energy_peratom"
    freeze_encoder_epochs: int = 20

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss: str = "huber"
    huber_delta: float = 0.5

    # ── Output paths ──────────────────────────────────────────────────────────
    output_dir: Path = Path("models/alignn_ef")

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_project: str = "eumine_databridge_2026"
    wandb_run_name: str = "alignn_ef_finetune"
    log_every_n_epochs: int = 5


def get_ef_config() -> ALIGNNTrainConfig:
    """Configuration optimized for formation energy prediction."""
    return ALIGNNTrainConfig(
        target="formation_energy_per_atom",
        cutoff=8.0,
        alignn_layers=4,
        gcn_layers=4,
        hidden_features=256,
        epochs=300,
        batch_size=32,
        learning_rate=1e-3,
        loss="huber",
        huber_delta=0.5,
        use_pretrained=True,
        pretrained_target="formation_energy_peratom",
        freeze_encoder_epochs=20,
        output_dir=Path("models/alignn_ef"),
        wandb_run_name="alignn_ef_v1",
    )


def get_bg_config() -> ALIGNNTrainConfig:
    """Configuration optimized for band gap prediction."""
    return ALIGNNTrainConfig(
        target="band_gap",
        cutoff=8.0,
        alignn_layers=6,
        gcn_layers=4,
        hidden_features=256,
        epochs=400,
        batch_size=32,
        learning_rate=1e-3,
        loss="huber",
        huber_delta=1.0,
        use_pretrained=True,
        pretrained_target="optb88vdw_bandgap",
        freeze_encoder_epochs=30,
        output_dir=Path("models/alignn_bg"),
        wandb_run_name="alignn_bg_v1",
        patience=60,
    )
