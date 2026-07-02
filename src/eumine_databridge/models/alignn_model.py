"""
ALIGNN model wrapper with fine-tuning logic.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

import wandb
from alignn.data import get_train_val_loaders
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from eumine_databridge.models.alignn_config import (
    ALIGNNTrainConfig,
    PRETRAINED_MODEL_KEYS,
)
from eumine_databridge.models.alignn_pretrained import load_pretrained_alignn
from eumine_databridge.models.alignn_data import ALIGNN_ID_KEY, ALIGNN_TARGET_KEY
from eumine_databridge.utils.metrics import score_property

load_dotenv()


class EarlyStoppingCallback:
    """Stop training when validation MAE stops improving."""

    def __init__(self, patience: int = 50, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_mae = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_mae: float) -> bool:
        if val_mae < self.best_mae - self.min_delta:
            self.best_mae = val_mae
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ALIGNNFineTuner:
    """Fine-tune a pretrained ALIGNN model on the Bridge Dataset."""

    def __init__(self, config: ALIGNNTrainConfig):
        self.config = config
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if _device.type == "cuda":
            # Verify DGL can actually move graphs to CUDA before committing.
            # CPU-only DGL wheels raise a fatal error when .to("cuda") is called.
            try:
                import dgl as _dgl
                _g = _dgl.graph(([0], [1]))
                _g.to(_device)
                del _g
            except Exception:
                _device = torch.device("cpu")
                print("  DGL CUDA unavailable — falling back to CPU inference")
        self.device = _device
        print(f"ALIGNNFineTuner — device: {self.device}")
        self.model: Optional[ALIGNN] = None
        self.optimizer: Optional[Adam] = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.best_val_mae = float("inf")
        self.best_epoch = 0

    def setup(
        self,
        train_dataset: List[dict],
        val_dataset: List[dict],
        target_stats: Optional[Dict] = None,
    ):
        """Initialize model, dataloaders, optimizer, scheduler."""
        output_dir = Path(self.config.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        # Remove stale LMDB dirs (alignn uses filename+'train_data' under cwd by default)
        for stale in Path.cwd().glob("bridge*data"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)

        combined = train_dataset + val_dataset
        n_train = len(train_dataset)
        n_val = len(val_dataset)

        print(f"\nSetting up dataloaders for target: {self.config.target}")
        print(f"  Train: {n_train}, Val: {n_val}")

        batch_size = min(
            self.config.batch_size, n_train, max(n_val, 1)
        )
        batch_size = max(batch_size, 1)
        # ALIGNN val loader uses drop_last=True — batch_size must divide n_val.
        if n_val > 0 and n_val % batch_size != 0:
            for bs in range(min(self.config.batch_size, n_val), 0, -1):
                if n_val % bs == 0:
                    batch_size = bs
                    break
            else:
                batch_size = 1

        self.train_loader, self.val_loader, _, _ = get_train_val_loaders(
            dataset_array=combined,
            target=ALIGNN_TARGET_KEY,
            atom_features=self.config.atom_features,
            neighbor_strategy="k-nearest",
            cutoff=self.config.cutoff,
            max_neighbors=self.config.max_neighbors,
            workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            batch_size=batch_size,
            n_train=n_train,
            n_val=n_val,
            n_test=0,
            train_ratio=None,
            val_ratio=0.1,
            test_ratio=0.0,
            keep_data_order=True,
            id_tag=ALIGNN_ID_KEY,
            line_graph=True,
            output_dir=str(output_dir),
            filename=str(output_dir / "alignn_"),
            use_lmdb=True,
        )
        print(
            f"  Train batches: {len(self.train_loader)}, "
            f"Val batches: {len(self.val_loader)}"
        )

        if self.config.use_pretrained:
            model_key = PRETRAINED_MODEL_KEYS.get(
                self.config.pretrained_target,
                self.config.pretrained_target,
            )
            print(f"\nLoading pretrained ALIGNN: {model_key}")
            self.model = load_pretrained_alignn(model_key, self.device)
            print("  Pretrained weights loaded successfully")
        else:
            print("\nInitializing ALIGNN from scratch")
            alignn_config = ALIGNNConfig(
                name="alignn",
                alignn_layers=self.config.alignn_layers,
                gcn_layers=self.config.gcn_layers,
                hidden_features=self.config.hidden_features,
                output_features=self.config.output_features,
                atom_features=self.config.atom_features,
                cutoff=self.config.cutoff,
                max_neighbors=self.config.max_neighbors,
            )
            self.model = ALIGNN(alignn_config).to(self.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6,
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=20,
                factor=0.5,
            )

    def _freeze_encoder(self):
        """Freeze all layers except readout / fc head."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if not any(k in name for k in ("readout", "fc", "link")):
                param.requires_grad = False
                frozen += 1
        print(f"  Encoder frozen: {frozen} parameter groups")

    def _unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
        print("  All parameters unfrozen")

    def _compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.config.loss == "huber":
            return nn.HuberLoss(delta=self.config.huber_delta)(pred, target)
        if self.config.loss == "mae":
            return nn.L1Loss()(pred, target)
        return nn.MSELoss()(pred, target)

    def _forward_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run model on a collated batch (g, lg, lat, target)."""
        g, lg, lat, target = batch
        g = g.to(self.device)
        lg = lg.to(self.device)
        lat = lat.to(self.device)
        target = target.to(self.device).float().view(-1)
        pred = self.model([g, lg, lat]).view(-1)
        return pred, target

    def train(self) -> Dict:
        """Full training loop with W&B logging and early stopping."""
        assert self.model is not None, "Call setup() first"
        output_dir = Path(self.config.output_dir)
        early_stopper = EarlyStoppingCallback(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        )

        wandb_kwargs = {
            "project": os.getenv("WANDB_PROJECT", self.config.wandb_project),
            "name": self.config.wandb_run_name,
            "config": {
                **asdict(self.config),
                "output_dir": str(self.config.output_dir),
            },
            "reinit": True,
            "mode": os.getenv("WANDB_MODE", "online"),
        }
        entity = os.getenv("WANDB_ENTITY", "").strip()
        if entity and entity not in ("your_wandb_username",):
            wandb_kwargs["entity"] = entity
        try:
            wandb_run = wandb.init(**wandb_kwargs)
        except Exception as exc:
            print(f"W&B init failed ({exc}); continuing with mode=disabled")
            wandb_kwargs["mode"] = "disabled"
            wandb_run = wandb.init(**wandb_kwargs)

        print(f"\n{'='*60}")
        print(f"Training ALIGNN for: {self.config.target.upper()}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs} (patience: {self.config.patience})")
        print(f"{'='*60}\n")

        history = {"train_loss": [], "val_mae": [], "lr": []}

        for epoch in range(1, self.config.epochs + 1):
            if epoch == 1 and self.config.freeze_encoder_epochs > 0:
                self._freeze_encoder()
            elif epoch == self.config.freeze_encoder_epochs + 1:
                self._unfreeze_all()

            train_loss = self._train_epoch()
            val_mae, val_preds, val_targets = self._validate()

            if self.config.scheduler == "cosine":
                self.scheduler.step()
            else:
                self.scheduler.step(val_mae)

            current_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["val_mae"].append(val_mae)
            history["lr"].append(current_lr)

            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_epoch = epoch
                torch.save(
                    self.model.state_dict(),
                    output_dir / "best_model.pt",
                )

            if epoch % self.config.log_every_n_epochs == 0 or epoch == 1:
                hack_score = score_property(val_mae, self._get_baseline_mae())
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_mae": val_mae,
                    "hackathon_score": hack_score,
                    "learning_rate": current_lr,
                    "best_val_mae": self.best_val_mae,
                })
                print(
                    f"Epoch {epoch:4d}/{self.config.epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Val MAE: {val_mae:.4f} | "
                    f"Best: {self.best_val_mae:.4f} | "
                    f"Score: {hack_score:.2f}/20 | "
                    f"LR: {current_lr:.2e}"
                )

            if early_stopper.step(val_mae):
                print(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"Best val MAE: {self.best_val_mae:.4f} "
                    f"at epoch {self.best_epoch}"
                )
                break

        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        final_score = score_property(
            self.best_val_mae, self._get_baseline_mae()
        )
        print(f"\n{'='*60}")
        print(f"Training complete — {self.config.target}")
        print(f"Best Val MAE : {self.best_val_mae:.4f}")
        print(f"Best Epoch   : {self.best_epoch}")
        print(f"Hackathon score (this property): {final_score:.2f}/20")
        if wandb_run is not None:
            print(f"W&B run URL: {wandb_run.url}")
        print(f"{'='*60}\n")

        wandb.summary["best_val_mae"] = self.best_val_mae
        wandb.summary["best_epoch"] = self.best_epoch
        wandb.summary["hackathon_score"] = final_score
        wandb.finish()

        return history

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            pred, target = self._forward_batch(batch)
            loss = self._compute_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _validate(self) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in self.val_loader:
                pred, target = self._forward_batch(batch)
                all_preds.extend(pred.cpu().numpy().tolist())
                all_targets.extend(target.cpu().numpy().tolist())
        preds = np.array(all_preds)
        targets = np.array(all_targets)
        mae = float(np.mean(np.abs(preds - targets)))
        return mae, preds, targets

    def load_best_model(self):
        checkpoint_path = Path(self.config.output_dir) / "best_model.pt"
        if checkpoint_path.exists():
            state = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state)
            print(f"Loaded best model from {checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}"
            )

    def _get_baseline_mae(self) -> float:
        if "formation" in self.config.target:
            return 0.2378
        return 0.6414

    def _init_model_only(self) -> None:
        """
        Initialize ALIGNN architecture without DataLoaders.
        Used by MatFed load_model() before load_best_model().
        """
        if self.config.use_pretrained:
            model_key = PRETRAINED_MODEL_KEYS.get(
                self.config.pretrained_target,
                self.config.pretrained_target,
            )
            print(f"  Initializing pretrained ALIGNN: {model_key}")
            self.model = load_pretrained_alignn(model_key, self.device)
        else:
            alignn_config = ALIGNNConfig(
                name="alignn",
                alignn_layers=self.config.alignn_layers,
                gcn_layers=self.config.gcn_layers,
                hidden_features=self.config.hidden_features,
                output_features=self.config.output_features,
                atom_features=self.config.atom_features,
                cutoff=self.config.cutoff,
                max_neighbors=self.config.max_neighbors,
            )
            self.model = ALIGNN(alignn_config).to(self.device)
        print(f"  Model architecture initialized on {self.device}")

    def predict(
        self,
        structures: List,
        material_ids: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Run inference on structures in input order."""
        from alignn.data import get_train_val_loaders
        from eumine_databridge.models.alignn_data import (
            ALIGNN_ID_KEY,
            ALIGNN_TARGET_KEY,
            structures_to_alignn_dataset,
        )

        if material_ids is None:
            material_ids = [f"mat_{i}" for i in range(len(structures))]

        dataset = structures_to_alignn_dataset(
            structures=structures,
            targets=[0.0] * len(structures),
            material_ids=material_ids,
            cutoff=self.config.cutoff,
            max_neighbors=self.config.max_neighbors,
        )
        if not dataset:
            return np.array([])

        # ALIGNN requires n_train >= 1; prepend dummy so all real samples are in val.
        infer_dataset = list(dataset)
        n_val = len(dataset)
        n_train = 1
        pad = dict(infer_dataset[0])
        pad[ALIGNN_ID_KEY] = f"{pad[ALIGNN_ID_KEY]}__alignn_pad"
        infer_dataset = [pad] + infer_dataset

        output_dir = Path(self.config.output_dir).resolve()
        import hashlib

        id_key = hashlib.md5(
            ",".join(material_ids).encode(), usedforsecurity=False
        ).hexdigest()[:12]
        infer_tag = f"alignn_infer_{id_key}_"
        batch_size = min(self.config.batch_size, n_val)
        batch_size = max(batch_size, 1)
        if n_val % batch_size != 0:
            for bs in range(min(self.config.batch_size, n_val), 0, -1):
                if n_val % bs == 0:
                    batch_size = bs
                    break
            else:
                batch_size = 1

        _, val_loader, _, _ = get_train_val_loaders(
            dataset_array=infer_dataset,
            target=ALIGNN_TARGET_KEY,
            atom_features=self.config.atom_features,
            neighbor_strategy="k-nearest",
            cutoff=self.config.cutoff,
            max_neighbors=self.config.max_neighbors,
            workers=self.config.num_workers,
            pin_memory=False,
            batch_size=batch_size,
            n_train=n_train,
            n_val=n_val,
            n_test=0,
            train_ratio=None,
            val_ratio=None,
            test_ratio=0.0,
            keep_data_order=True,
            id_tag=ALIGNN_ID_KEY,
            line_graph=True,
            output_dir=str(output_dir),
            filename=str(output_dir / infer_tag),
            use_lmdb=True,
        )

        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in val_loader:
                pred, _ = self._forward_batch(batch)
                preds.extend(pred.cpu().numpy().tolist())
        return np.array(preds[:n_val])

    def save_config(self):
        output_dir = Path(self.config.output_dir)
        config_dict = asdict(self.config)
        config_dict["output_dir"] = str(config_dict["output_dir"])
        with open(output_dir / "train_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
