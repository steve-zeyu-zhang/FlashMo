"""Reconstruction diagnostics and standard metrics from external T2M embeddings.

These modes are deliberately separate: raw rotation errors are NOT FID or
HumanML3D R-Precision. Benchmark embeddings must come from its frozen evaluator.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader

from flashmo.data import MotionDataset, collate_motion
from flashmo.geometry import so3_geodesic_distance
from flashmo.models import SO3MotionVAE
from flashmo.utils import load_checkpoint, to_device, write_json


def embedding_metrics(real, generated, text=None, seed=42):
    """Compute FID plus retrieval@1/2/3 in the standard random pools of 32."""
    real, generated = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("Expected real/generated [N,D] embeddings from the SAME pretrained evaluator")
    if min(len(real), len(generated)) < 2 or not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("Need at least two finite embeddings in each distribution")
    mean_r, mean_g = real.mean(0), generated.mean(0)
    cov_r, cov_g = np.atleast_2d(np.cov(real, rowvar=False)), np.atleast_2d(np.cov(generated, rowvar=False))
    covariance_mean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covariance_mean):
        if np.max(np.abs(covariance_mean.imag)) > 1e-5:
            raise ValueError("Unstable covariance square root; use more valid evaluator samples")
        covariance_mean = covariance_mean.real
    fid = np.sum((mean_r - mean_g) ** 2) + np.trace(cov_r + cov_g - 2 * covariance_mean)
    rng = np.random.default_rng(seed)
    pairs = min(300, len(generated))
    first, second = rng.choice(len(generated), pairs, replace=False), rng.choice(len(generated), pairs, replace=False)
    result = {"fid": max(0., float(fid)), "diversity": float(np.linalg.norm(generated[first] - generated[second], axis=-1).mean()),
              "real_samples": len(real), "generated_samples": len(generated), "evaluator": "externally supplied pretrained HumanML3D embeddings"}
    if text is not None:
        text = np.asarray(text, dtype=np.float64)
        if text.shape != generated.shape or not np.isfinite(text).all():
            raise ValueError("Text and generated embeddings must be finite, paired and identically shaped")
        if len(text) < 32:
            raise ValueError("HumanML3D R-Precision requires at least 32 paired embeddings")
        order = rng.permutation(len(text))
        ranks, distance_sum, count = [], 0., 0
        for start in range(0, len(text) - 31, 32):
            ids = order[start:start + 32]
            distances = np.linalg.norm(text[ids, None] - generated[None, ids], axis=-1)
            ranked = np.argsort(distances, axis=-1)
            ranks.extend(np.argmax(ranked == np.arange(32)[:, None], axis=-1).tolist())
            distance_sum += float(np.trace(distances))
            count += 32
        result.update({f"r_precision_top_{k}": float((np.asarray(ranks) < k).mean()) for k in (1, 2, 3)})
        result.update(matching_distance=distance_sum / count, retrieval_samples=count)
    return result


@torch.no_grad()
def reconstruction(checkpoint_path, manifest, device):
    checkpoint = load_checkpoint(checkpoint_path)
    vae = SO3MotionVAE(**checkpoint["config"]["vae"]).to(device).eval()
    vae.load_state_dict(checkpoint["model"] if checkpoint["stage"] == "vae" else checkpoint["vae"])
    dataset = MotionDataset(manifest, max_frames=checkpoint["config"]["data"]["max_frames"])
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_motion)
    rotation_sum = translation_sum = valid_frames = 0.
    for batch in loader:
        batch = to_device(batch, device)
        result = vae(batch["rotations"], batch["translations"], batch["mask"], sample=False)
        valid = batch["mask"]
        error = so3_geodesic_distance(result["rotations"], batch["rotations"]).mean(-1)
        rotation_sum += float(error[valid].sum())
        translation_sum += float((result["translations"] - batch["translations"]).square().sum(-1)[valid].sum())
        valid_frames += float(valid.sum())
    return {"rotation_mean_degrees": rotation_sum / valid_frames * 180 / np.pi,
        "translation_rmse_m": (translation_sum / valid_frames) ** .5, "frames": int(valid_frames),
        "motions": len(dataset), "metric_scope": "VAE reconstruction on supplied manifest; not benchmark FID",
        "benchmark_reproduced": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    reconstruction_parser = sub.add_parser("reconstruction")
    reconstruction_parser.add_argument("--checkpoint", required=True)
    reconstruction_parser.add_argument("--manifest", required=True)
    reconstruction_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    embeddings_parser = sub.add_parser("embeddings")
    embeddings_parser.add_argument("--input", required=True, help="npz with real_motion, generated_motion, optionally text")
    embeddings_parser.add_argument("--seed", type=int, default=42)
    for child in (reconstruction_parser, embeddings_parser):
        child.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.mode == "reconstruction":
        result = reconstruction(args.checkpoint, args.manifest, torch.device(args.device))
    else:
        with np.load(args.input, allow_pickle=False) as arrays:
            result = embedding_metrics(arrays["real_motion"], arrays["generated_motion"],
                arrays["text"] if "text" in arrays else None, args.seed)
    write_json(args.output, result)
    print(Path(args.output).read_text())


if __name__ == "__main__":
    main()
