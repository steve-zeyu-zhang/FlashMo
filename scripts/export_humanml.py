#!/usr/bin/env python3
"""Export generated rotations through licensed SMPL-H to official T2M features."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashmo.data.amass import sha256_file
from flashmo.data.humanml_export import joints_to_humanml, normalize_humanml, reconstruct_smplh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Generated rotations/translations NPZ")
    parser.add_argument("--output", type=Path, required=True, help="Raw [T-1,263] .npy features")
    parser.add_argument("--body-model", type=Path, required=True, help="Licensed AMASS SMPL-H model.npz")
    parser.add_argument("--reference-joints", type=Path, required=True, help="Official HumanML3D joints/000021.npy")
    parser.add_argument("--betas", type=Path, help="Optional shape .npy; else NPZ betas or explicitly recorded mean body")
    parser.add_argument("--fps", type=float, help="Required only if the NPZ does not store fps; must be 20")
    parser.add_argument("--sample-index", type=int, default=0, help="Which example to export from a batch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mean", type=Path, help="Optional official evaluator Mean.npy")
    parser.add_argument("--std", type=Path, help="Optional official evaluator Std.npy")
    parser.add_argument("--save-joints", type=Path, help="Also save canonical y-up SMPL22 joints")
    args = parser.parse_args()
    if args.output.suffix != ".npy":
        parser.error("--output must be a .npy file")
    if not args.body_model.is_file() or not args.reference_joints.is_file():
        parser.error("Both licensed --body-model and real --reference-joints are required; see docs/DATA.md")
    if bool(args.mean) != bool(args.std):
        parser.error("--mean and --std must be supplied together")
    with np.load(args.input, allow_pickle=False) as data:
        rotations = np.array(data["rotations"], dtype=np.float32)
        translations = np.array(data["translations"], dtype=np.float32)
        stored_fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else None
        fps = stored_fps if stored_fps is not None else args.fps
        mask = np.array(data["mask"], dtype=bool) if "mask" in data else None
        betas = np.array(data["betas"]) if "betas" in data else None
    if stored_fps is not None and args.fps is not None and not np.isclose(stored_fps, args.fps, atol=1e-6, rtol=0):
        parser.error("--fps disagrees with the stored source FPS; it cannot relabel unresampled motion")
    if fps is None or not np.isclose(fps, 20, atol=1e-6, rtol=0):
        parser.error("HumanML conversion requires known 20 Hz input; resample motion first")
    if rotations.ndim == 5:
        rotations, translations = rotations[args.sample_index], translations[args.sample_index]
        if mask is not None:
            mask = mask[args.sample_index]
        if betas is not None and betas.ndim == 2:
            betas = betas[args.sample_index]
    if mask is not None:
        if mask.ndim != 1 or len(mask) != len(rotations) or not np.array_equal(mask, np.arange(len(mask)) < mask.sum()):
            parser.error("Mask must mark a contiguous valid prefix of frames")
        rotations, translations = rotations[:mask.sum()], translations[:mask.sum()]
    if args.betas:
        betas = np.load(args.betas, allow_pickle=False)
    joints = reconstruct_smplh(rotations, translations, args.body_model, betas=betas,
                               device=args.device, batch_size=args.batch_size)
    reference = np.load(args.reference_joints, allow_pickle=False)
    if reference.ndim == 3:
        reference = reference[:, :22]
    features, canonical = joints_to_humanml(joints, reference)
    if args.mean:
        features = normalize_humanml(features, np.load(args.mean, allow_pickle=False), np.load(args.std, allow_pickle=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, features)
    if args.save_joints:
        args.save_joints.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_joints, canonical)
    metadata = {"input": str(args.input), "input_sha256": sha256_file(args.input), "fps": fps,
                "body_model": str(args.body_model), "body_model_sha256": sha256_file(args.body_model),
                "reference_joints": str(args.reference_joints), "reference_sha256": sha256_file(args.reference_joints),
                "shape": "zero_mean_body" if betas is None else "provided_betas_first10",
                "features_shape": list(features.shape), "normalized": bool(args.mean),
                "humanml_commit": "9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23",
                "benchmark_metrics_computed": False}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {features.shape} HumanML3D features to {args.output}")


if __name__ == "__main__":
    main()
