"""Read original SMPL/SMPL-H/SMPL-X AMASS poses, never HumanML velocity vectors.

The first 22 axis-angle joints are root + 21 SMPL body joints. Rotations remain
in native AMASS coordinates (z-up); joint 0 is global, the others are local.
No SMPL body model is needed to read rotations or translations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


HUMANML_FPS = 20.0
PREFIX_SECONDS = {
    "Eyes_Japan_Dataset": 3.0,
    "MPI_HDM05": 3.0,
    "TotalCapture": 1.0,
    "MPI_Limits": 1.0,
    "Transitions_mocap": 0.5,
}
SOURCE_ALIASES = {
    "BioMotionLab_NTroje": ("BioMotionLab_NTroje", "BMLrub"),
    "DFaust_67": ("DFaust_67", "DFaust"),
    "MPI_HDM05": ("MPI_HDM05", "HDM05"),
    "MPI_Limits": ("MPI_Limits", "PosePrior"),
    "MPI_mosh": ("MPI_mosh", "MoSh"),
    "SSM_synced": ("SSM_synced", "SSM"),
    "TCD_handMocap": ("TCD_handMocap", "TCDHands"),
    "Transitions_mocap": ("Transitions_mocap", "Transitions"),
    "Eyes_Japan_Dataset": ("Eyes_Japan_Dataset", "EyesJapanDataset"),
}
LEFT_JOINTS = np.array([1, 4, 7, 10, 13, 16, 18, 20])
RIGHT_JOINTS = np.array([2, 5, 8, 11, 14, 17, 19, 21])


@dataclass(frozen=True)
class AMASSMotion:
    rotations: np.ndarray
    translations: np.ndarray
    fps: float
    source: str
    betas: np.ndarray | None = None
    gender: str | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_amass(path: str | Path) -> AMASSMotion:
    """Validate original AMASS fields and decode the first 22 axis-angle joints."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Expected original AMASS .npz; HumanML3D .npy velocity features are unsupported")
    with np.load(path, allow_pickle=False) as data:
        required = {"poses", "trans"}
        if not required.issubset(data.files):
            raise ValueError(f"{path}: original AMASS poses/trans missing; do not use velocity features")
        fps_key = next((k for k in ("mocap_framerate", "mocap_frame_rate") if k in data), None)
        if fps_key is None:
            raise ValueError(f"{path}: source frame rate is missing")
        fps_value = np.asarray(data[fps_key])
        if fps_value.size != 1:
            raise ValueError(f"{path}: source frame rate must be scalar")
        fps = float(fps_value.reshape(()))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{path}: source frame rate must be positive and finite")
        poses = np.asarray(data["poses"], dtype=np.float64)
        trans = np.asarray(data["trans"], dtype=np.float64)
        if poses.ndim == 3 and poses.shape[-1] == 3:
            poses = poses.reshape(poses.shape[0], -1)
        if poses.ndim != 2 or poses.shape[1] < 66 or poses.shape[1] % 3:
            raise ValueError(f"{path}: expected axis-angle poses [T,>=66], got {poses.shape}")
        if len(poses) < 2 or trans.shape != (len(poses), 3):
            raise ValueError(f"{path}: need >=2 frames and trans shape [T,3]")
        if not np.isfinite(poses).all() or not np.isfinite(trans).all():
            raise ValueError(f"{path}: nonfinite motion values")
        betas = np.asarray(data["betas"], dtype=np.float32) if "betas" in data else None
        gender = None
        if "gender" in data:
            value = data["gender"].item()
            gender = value.decode("utf8") if isinstance(value, bytes) else str(value)
    rotations = Rotation.from_rotvec(poses[:, :66].reshape(-1, 3)).as_matrix()
    return AMASSMotion(rotations.reshape(-1, 22, 3, 3).astype(np.float32),
                       trans.astype(np.float32), fps, str(path.resolve()), betas, gender)


def resample_motion(
    motion: AMASSMotion, fps: float = 20.0, *, start_time: float = 0.0,
    end_time: float | None = None,
) -> AMASSMotion:
    """SLERP rotations and interpolate translation on the actual source time grid.

    end_time is exclusive. No source extrapolation or integer-stride FPS rounding.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Target fps must be positive and finite")
    last_time = (len(motion.rotations) - 1) / motion.fps
    if end_time is None:
        end_time = len(motion.rotations) / motion.fps
    if not (math.isfinite(start_time) and math.isfinite(end_time)) or start_time < 0 or end_time <= start_time:
        raise ValueError("Invalid crop interval")
    # Include only output sample times present in the source. Half-open crop.
    count = max(0, int(math.ceil((end_time - start_time) * fps - 1e-9)))
    target_times = start_time + np.arange(count, dtype=np.float64) / fps
    target_times = target_times[target_times <= last_time + 1e-9]
    if len(target_times) < 2:
        raise ValueError("Crop contains fewer than two resampled frames")
    target_times = np.minimum(target_times, last_time)
    source_times = np.arange(len(motion.rotations), dtype=np.float64) / motion.fps
    output = np.stack([
        Slerp(source_times, Rotation.from_matrix(motion.rotations[:, j]))(target_times).as_matrix()
        for j in range(22)
    ], axis=1)
    trans = np.stack([np.interp(target_times, source_times, motion.translations[:, j]) for j in range(3)], axis=-1)
    return replace(motion, rotations=output.astype(np.float32), translations=trans.astype(np.float32), fps=float(fps))


def mirror_motion(motion: AMASSMotion) -> AMASSMotion:
    """Sagittal reflection (x -> -x) plus SMPL left/right joint permutation."""
    order = np.arange(22)
    order[LEFT_JOINTS], order[RIGHT_JOINTS] = RIGHT_JOINTS, LEFT_JOINTS
    reflection = np.diag([-1.0, 1.0, 1.0])
    rotations = reflection @ motion.rotations[:, order] @ reflection
    trans = motion.translations @ reflection
    return replace(motion, rotations=rotations.astype(np.float32), translations=trans.astype(np.float32))


def resolve_source(amass_root: str | Path, source_path: str) -> tuple[Path | None, str]:
    """Resolve HumanML's pose_data paths and official AMASS release aliases."""
    parts = list(Path(source_path.replace("\\", "/")).parts)
    if ".." in parts:
        raise ValueError("Source paths may not escape the AMASS root")
    for prefix in ("pose_data", "amass_data"):
        if prefix in parts:
            parts = parts[parts.index(prefix) + 1:]
            break
    if len(parts) < 2 or Path(source_path).is_absolute():
        raise ValueError(f"Invalid HumanML3D source path: {source_path}")
    dataset = parts[0]
    if dataset.lower() == "humanact12":
        return None, dataset  # HumanAct12 joint positions are not original AMASS rotations.
    relative = Path(*parts[1:]).with_suffix(".npz")
    filenames = [relative]
    if relative.name.endswith("_poses.npz"):
        filenames.append(relative.with_name(relative.name.replace("_poses.npz", "_stageii.npz")))
    for name in SOURCE_ALIASES.get(dataset, (dataset,)):
        for filename in filenames:
            candidate = Path(amass_root) / name / filename
            if candidate.is_file():
                return candidate, dataset
    return None, dataset


def humanml_source_interval(motion: AMASSMotion, dataset: str, start: int, end: int) -> tuple[float, float, float]:
    """Map index.csv's official decimated frame coordinates to source seconds.

    HumanML used stride=floor(fps/20), even for nonmultiples of 20. Therefore
    timed captions need the same time-scale correction after true-FPS resampling.
    """
    stride = int(motion.fps / HUMANML_FPS)
    if stride < 1:
        raise ValueError("Cannot align official HumanML index for source FPS below 20")
    offset = int(PREFIX_SECONDS.get(dataset, 0) * HUMANML_FPS)
    available = (len(motion.rotations) + stride - 1) // stride - offset
    stop = available + end if end < 0 else min(end, available)
    if start < 0 or start >= stop:
        raise ValueError(f"Invalid HumanML interval [{start}, {end}) for {available} frames")
    scale = HUMANML_FPS * stride / motion.fps
    return (offset + start) * stride / motion.fps, (offset + stop) * stride / motion.fps, scale


def read_captions(path: str | Path, num_frames: int, fps: float, time_scale: float = 1.0) -> list[dict[str, Any]]:
    """Read exact HumanML captions and their optional timed subsequence labels."""
    captions = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.rsplit("#", 3)
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number}: expected text#tokens#start#end")
        text, tokens, start_raw, end_raw = fields
        start_time, end_time = float(start_raw), float(end_raw)
        # Official loader treats missing NaN annotation times as 0.
        start_time = 0 if math.isnan(start_time) else start_time
        end_time = 0 if math.isnan(end_time) else end_time
        if not (math.isfinite(start_time) and math.isfinite(end_time)) or min(start_time, end_time) < 0:
            raise ValueError(f"{path}:{line_number}: invalid annotation times")
        if start_time == end_time == 0:
            start, end = 0, num_frames
        else:
            start = int(start_time * fps * time_scale)
            end = min(int(end_time * fps * time_scale), num_frames)
        if end - start < 2:
            continue
        captions.append({"text": text, "tokens": tokens, "start_frame": start, "end_frame": end})
    return captions


def _save_motion(output: Path, name: str, motion: AMASSMotion) -> str:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Invalid motion ID")
    relative = Path("motions") / f"{name}.npz"
    (output / "motions").mkdir(parents=True, exist_ok=True)
    kwargs = {"rotations": motion.rotations, "translations": motion.translations, "fps": np.array(motion.fps)}
    if motion.betas is not None:
        kwargs["betas"] = motion.betas
    if motion.gender is not None:
        kwargs["gender"] = np.array(motion.gender)
    np.savez_compressed(output / relative, **kwargs)
    return relative.as_posix()


def _write_manifest(output: Path, records: list[dict[str, Any]], report: dict[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf8")
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    return manifest


def prepare_samples(amass_root: str | Path, output: str | Path, fps: float = 20.0) -> Path:
    """Prepare unpaired authentic AMASS samples for VAE/smoke use only."""
    output = Path(output)
    records = []
    for index, source in enumerate(sorted(Path(amass_root).rglob("*.npz"))):
        raw = load_amass(source)
        motion = resample_motion(raw, fps)
        name = f"sample_{index:04d}_{source.stem}"
        records.append({"id": name, "path": _save_motion(output, name, motion),
                        "fps": fps, "num_frames": len(motion.rotations), "text": None,
                        "paired": False, "source": str(source.resolve()), "source_sha256": sha256_file(source),
                        "source_fps": raw.fps, "coordinate_system": "amass_z_up",
                        "start_frame": 0, "end_frame": len(motion.rotations),
                        "humanml_subset_verified": False})
    if not records:
        raise ValueError(f"No original AMASS .npz samples found under {amass_root}")
    return _write_manifest(output, records, {"mode": "unpaired_samples", "motions": len(records),
                                           "benchmark_data": False})


def prepare_humanml(
    amass_root: str | Path, index_path: str | Path, texts: str | Path,
    split_path: str | Path, output: str | Path, fps: float = 20.0,
    allow_missing: bool = False,
) -> Path:
    """Prepare exactly the AMASS portion of one official HumanML3D split.

    Every mirrored clip uses its own official Mxxxxxx caption file. HumanAct12
    is reported separately because its original AMASS rotations do not exist.
    Missing AMASS files/captions fail by default; --allow-missing is explicit.
    """
    output = Path(output)
    split = {line.strip().removesuffix(".npy") for line in Path(split_path).read_text().splitlines() if line.strip()}
    with Path(index_path).open(newline="", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    records: list[dict[str, Any]] = []
    report: dict[str, Any] = {"mode": "humanml_amass_subset", "split": str(split_path),
                             "requested_ids": len(split), "humanact12_excluded": [],
                             "missing": [], "motions": 0, "caption_examples": 0}
    found_ids: set[str] = set()
    for row in rows:
        base_name = Path(row["new_name"]).stem
        requested = [name for name in (base_name, "M" + base_name) if name in split]
        if not requested:
            continue
        found_ids.update(requested)
        source, dataset = resolve_source(amass_root, row["source_path"])
        if dataset.lower() == "humanact12":
            report["humanact12_excluded"].extend(requested)
            continue
        if source is None:
            report["missing"].append({"ids": requested, "source": row["source_path"], "reason": "missing_amass"})
            continue
        raw = load_amass(source)
        start_time, end_time, scale = humanml_source_interval(raw, dataset, int(row["start_frame"]), int(row["end_frame"]))
        motion = resample_motion(raw, fps, start_time=start_time, end_time=end_time)
        source_hash = sha256_file(source)
        for name in requested:
            caption_path = Path(texts) / f"{name}.txt"
            if not caption_path.is_file():
                report["missing"].append({"ids": [name], "reason": "missing_caption"})
                continue
            captions = read_captions(caption_path, len(motion.rotations), fps, scale)
            if not captions:
                report["missing"].append({"ids": [name], "reason": "no_valid_captions"})
                continue
            mirrored = name.startswith("M")
            prepared = mirror_motion(motion) if mirrored else motion
            relative = _save_motion(output, name, prepared)
            for caption_index, caption in enumerate(captions):
                records.append({"id": name, "caption_index": caption_index, "path": relative,
                                "fps": fps, "num_frames": len(motion.rotations), "paired": True,
                                "source": str(source.resolve()), "source_sha256": source_hash,
                                "source_fps": raw.fps, "source_start_seconds": start_time,
                                "source_end_seconds": end_time, "annotation_time_scale": scale,
                                "mirrored": mirrored, "coordinate_system": "amass_z_up",
                                "humanml_subset_verified": True, **caption})
            report["motions"] += 1
    unknown = sorted(split - found_ids)
    if unknown:
        report["missing"].append({"ids": unknown, "reason": "unknown_index_ids"})
    report["caption_examples"] = len(records)
    output.mkdir(parents=True, exist_ok=True)
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["missing"] and not allow_missing:
        raise FileNotFoundError(f"Missing {len(report['missing'])} sources/captions; see {output / 'preparation_report.json'}. "
                                "For an explicitly partial subset, pass --allow-missing.")
    if not records:
        raise ValueError("No matched AMASS motions with official HumanML3D captions")
    return _write_manifest(output, records, report)
