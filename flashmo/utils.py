"""Configuration, checkpoints, and deterministic experiment bookkeeping."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ("data", "vae", "model", "text", "transport", "train", "sample"):
        if key not in config:
            raise ValueError(f"Missing configuration section: {key}")
    compression = config["vae"]["compression"]
    for key in ("context_frames", "prediction_frames"):
        if config["train"][key] % compression:
            raise ValueError(f"{key} must be a multiple of VAE compression={compression}")
    if config["train"]["prediction_frames"] <= 0:
        raise ValueError("prediction_frames must be positive")
    return config


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(checkpoint: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, stage: str | None = None) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected a FlashMo format_version=1 checkpoint")
    if stage is not None and checkpoint.get("stage") != stage:
        raise ValueError(f"Expected {stage} checkpoint, got {checkpoint.get('stage')!r}")
    return checkpoint


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}
