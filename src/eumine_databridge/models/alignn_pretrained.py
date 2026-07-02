"""
Load JARVIS pretrained ALIGNN weights without alignn.pretrained (jarvis-tools compat).
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path

import requests
import torch
from tqdm import tqdm

from alignn.models.alignn import ALIGNN, ALIGNNConfig

# Subset of alignn.pretrained.all_models (figshare URLs)
FIGSHARE_MODELS = {
    "jv_formation_energy_peratom_alignn": (
        "https://ndownloader.figshare.com/files/31458679"
    ),
    "jv_optb88vdw_bandgap_alignn": (
        "https://ndownloader.figshare.com/files/31459636"
    ),
}


def _cache_dir() -> Path:
    root = Path(
        os.getenv(
            "ALIGNN_MODEL_CACHE",
            Path.home() / ".cache" / "eumine_alignn_models",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_pretrained_alignn(
    model_name: str,
    device: torch.device | str = "cpu",
) -> ALIGNN:
    """Download (if needed) and load a figshare ALIGNN checkpoint."""
    if model_name not in FIGSHARE_MODELS:
        raise ValueError(
            f"Unknown model {model_name}. "
            f"Available: {list(FIGSHARE_MODELS)}"
        )

    url = FIGSHARE_MODELS[model_name]
    zpath = _cache_dir() / f"{model_name}.zip"

    if not zpath.is_file():
        print(f"Downloading pretrained model: {model_name}")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(zpath, "wb") as f:
            for chunk in tqdm(
                response.iter_content(8192),
                total=max(total // 8192, 1),
                unit="KB",
                desc=model_name,
            ):
                f.write(chunk)

    with zipfile.ZipFile(zpath) as zp:
        names = zp.namelist()
        cfg_name = next(n for n in names if n.endswith("config.json"))
        ckpt_name = next(
            n for n in names
            if "best_model.pt" in n or ("checkpoint_" in n and n.endswith(".pt"))
        )
        config = json.loads(zp.read(cfg_name))
        ckpt_bytes = zp.read(ckpt_name)

    if config["model"]["name"] != "alignn":
        raise RuntimeError(
            f"Expected alignn model, got {config['model']['name']}"
        )

    model = ALIGNN(ALIGNNConfig(**config["model"]))
    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(ckpt_bytes)
        state = torch.load(tmp_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    model.to(device)
    model.eval()
    return model
