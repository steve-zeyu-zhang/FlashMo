"""Rotation sequence dataset with real captions and true-valid padding masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self, manifest: str | Path, max_frames: int | None = None, require_text: bool = False,
                 random_crop: bool = False):
        self.manifest = Path(manifest)
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        self.max_frames = max_frames
        self.random_crop = random_crop
        if not self.rows:
            raise ValueError("Empty motion manifest")
        if max_frames is not None and max_frames < 2:
            raise ValueError("max_frames must be at least two")
        if require_text and any(not r.get("paired") or not r.get("text") for r in self.rows):
            raise ValueError("Text-conditioned training requires real paired captions; unpaired samples are smoke/VAE only")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.manifest.parent / row["path"]
        with np.load(path, allow_pickle=False) as data:
            rotations = np.array(data["rotations"], dtype=np.float32, copy=True)
            translations = np.array(data["translations"], dtype=np.float32, copy=True)
        if rotations.ndim != 4 or rotations.shape[1:] != (22, 3, 3) or translations.shape != (len(rotations), 3):
            raise ValueError(f"{path}: expected rotations [T,22,3,3] and translations [T,3]")
        start, end = row.get("start_frame", 0), row.get("end_frame", len(rotations))
        if not (0 <= start < end <= len(rotations)):
            raise ValueError(f"{path}: invalid caption interval")
        if self.max_frames is not None:
            if self.random_crop and end - start > self.max_frames:
                start += int(torch.randint(end - start - self.max_frames + 1, ()).item())
            end = min(end, start + self.max_frames)
        rotations, translations = rotations[start:end], translations[start:end]
        # Both VAE and flow learn displacement from the selected crop's origin.
        # Original world translations remain intact in the prepared NPZ.
        translations = translations - translations[:1]
        if not np.isfinite(rotations).all() or not np.isfinite(translations).all():
            raise ValueError(f"{path}: nonfinite motion values")
        return {"rotations": torch.from_numpy(rotations), "translations": torch.from_numpy(translations),
                "mask": torch.ones(len(rotations), dtype=torch.bool), "text": row.get("text") or "",
                "id": row["id"], "fps": float(row["fps"]), "paired": bool(row.get("paired", False))}


def collate_motion(items):
    """Batch [B,T,22,3,3] matrices, [B,T,3] translations and [B,T] masks.

    Padded rotations are identity (valid SO(3)); mask=False excludes them.
    """
    if not items:
        raise ValueError("Cannot collate an empty batch")
    length = max(len(item["rotations"]) for item in items)
    batch = len(items)
    rotations = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(batch, length, 22, 1, 1)
    translations = torch.zeros(batch, length, 3)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    for i, item in enumerate(items):
        n = len(item["rotations"])
        rotations[i, :n] = item["rotations"]
        translations[i, :n] = item["translations"]
        mask[i, :n] = item["mask"]
    return {"rotations": rotations, "translations": translations, "mask": mask,
            "text": [item["text"] for item in items], "id": [item["id"] for item in items],
            "fps": torch.tensor([item["fps"] for item in items]),
            "paired": torch.tensor([item["paired"] for item in items], dtype=torch.bool)}
