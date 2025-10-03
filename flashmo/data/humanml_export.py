"""Licensed SMPL-H reconstruction followed by official HumanML3D features."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ._humanml import processor
from ._humanml.skeleton import Skeleton


_PROCESSOR_LOCK = Lock()


def _validate_joints(joints, name):
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3) or len(joints) < 2:
        raise ValueError(f"{name} must have shape [T>=2,22,3]")
    if not np.isfinite(joints).all():
        raise ValueError(f"{name} contains nonfinite positions")
    skel = Skeleton(processor.n_raw_offsets, processor.kinematic_chain, "cpu")
    parents = skel.parents()
    lengths = np.linalg.norm(joints[:, 1:] - joints[:, np.asarray(parents[1:])], axis=-1)
    if np.any(lengths < 1e-6):
        raise ValueError(f"{name} contains zero-length bones")
    return joints


def joints_to_humanml(joints, reference_joints, feet_threshold=0.002):
    """Canonicalize y-up SMPL22 positions and return raw [T-1,263] features.

    reference_joints must be the real HumanML reference skeleton (000021.npy)
    for benchmark conversion. Synthetic fixtures are suitable only for tests.
    """
    joints = _validate_joints(joints, "joints")
    reference = np.asarray(reference_joints, dtype=np.float32)
    if reference.ndim == 2:
        reference = np.stack([reference, reference])
    reference = _validate_joints(reference, "reference_joints")
    if not np.isfinite(feet_threshold) or feet_threshold <= 0:
        raise ValueError("Foot contact threshold must be positive and finite")
    # Original notebook function uses module globals. Serialize configuration
    # so callers supplying different target bodies cannot race each other.
    with _PROCESSOR_LOCK:
        target = Skeleton(processor.n_raw_offsets, processor.kinematic_chain, "cpu")
        processor.tgt_offsets = target.get_offsets_joints(torch.from_numpy(reference[0]))
        with np.errstate(invalid="ignore", divide="ignore"):
            features, canonical, _, _ = processor.process_file(joints.copy(), feet_threshold)
    if features.shape != (len(joints) - 1, 263) or not np.isfinite(features).all():
        raise ValueError("Official HumanML preprocessing produced invalid features; check skeleton geometry/orientation")
    if not np.isfinite(canonical).all():
        raise ValueError("Official HumanML preprocessing produced invalid canonical joints")
    return features.astype(np.float32), canonical.astype(np.float32)


def reconstruct_smplh(rotations, translations, body_model: str | Path,
                      betas=None, device="cpu", batch_size=128):
    """Reconstruct real SMPL-H joints using caller-provided licensed assets.

    Uses human_body_prior's BodyModel, as in official HumanML preprocessing.
    Hand pose is zero because the model predicts only 22 body joints. The first
    22 joints do not depend on distal hand rotations. No synthetic rest skeleton
    or fallback body model is provided.
    """
    model_path = Path(body_model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Licensed SMPL-H model is required: {model_path}")
    rotations = np.asarray(rotations, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    if rotations.ndim != 4 or rotations.shape[1:] != (22, 3, 3) or translations.shape != (len(rotations), 3):
        raise ValueError("Expected rotations [T,22,3,3] and translations [T,3]")
    if not np.isfinite(rotations).all() or not np.isfinite(translations).all():
        raise ValueError("Nonfinite generated motion")
    gram = np.swapaxes(rotations, -1, -2) @ rotations
    if not np.allclose(gram, np.eye(3), atol=2e-3) or not np.allclose(np.linalg.det(rotations), 1, atol=2e-3):
        raise ValueError("Generated matrices must be valid SO(3), not arbitrary matrix features")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        from human_body_prior.body_model.body_model import BodyModel
    except ImportError as exc:
        raise ImportError("Install optional human_body_prior to use licensed SMPL-H export; see docs/DATA.md") from exc
    shape = np.zeros(10, dtype=np.float32) if betas is None else np.asarray(betas, dtype=np.float32).reshape(-1)[:10]
    if len(shape) != 10 or not np.isfinite(shape).all():
        raise ValueError("Expected at least 10 finite SMPL shape coefficients")
    model = BodyModel(bm_fname=str(model_path), num_betas=10).to(device).eval()
    if getattr(model, "model_type", None) != "smplh":
        raise ValueError("Provide the extended SMPL-H model used by AMASS, not a different skeleton model")
    axis_angle = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 66).astype(np.float32)
    output = []
    with torch.no_grad():
        for start in range(0, len(rotations), batch_size):
            end = min(start + batch_size, len(rotations))
            aa = torch.from_numpy(axis_angle[start:end]).to(device)
            result = model(root_orient=aa[:, :3], pose_body=aa[:, 3:],
                           pose_hand=torch.zeros(end - start, 90, device=device),
                           trans=torch.from_numpy(translations[start:end]).to(device),
                           betas=torch.from_numpy(shape).to(device).expand(end - start, -1))
            output.append(result.Jtr[:, :22].detach().cpu().numpy())
    positions = np.concatenate(output, axis=0)
    # Exact coordinate change from HumanML's raw_pose_processing notebook:
    # first swap y/z, then negate x. This is a proper world-space rotation.
    positions = positions[..., [0, 2, 1]].copy()
    positions[..., 0] *= -1
    return positions.astype(np.float32)


def normalize_humanml(features, mean, std):
    mean, std = np.asarray(mean), np.asarray(std)
    if mean.shape != (263,) or std.shape != (263,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Evaluator mean/std must be finite vectors of length 263")
    if np.any(std <= 0):
        raise ValueError("Evaluator std must be strictly positive")
    return ((features - mean) / std).astype(np.float32)
