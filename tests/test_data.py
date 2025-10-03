"""Semantic tests for original-rotation ingestion and caption/source alignment."""

import csv
import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from flashmo.data import MotionDataset, collate_motion
from flashmo.data.amass import (
    humanml_source_interval, load_amass, mirror_motion, prepare_humanml,
    prepare_samples, read_captions, resample_motion, resolve_source,
)


def write_amass(path, fps=60.0, frames=180):
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(frames) / fps
    poses = np.zeros((frames, 156))
    poses[:, 2] = times * 0.4
    poses[:, 3] = times * 0.2
    trans = np.stack([times, 2 * times, times * 0], axis=-1)
    np.savez(path, poses=poses, trans=trans, mocap_framerate=fps)
    return path


@pytest.mark.parametrize("fps", [25.0, 59.94, 60.0, 120.0])
def test_true_fps_resample_preserves_physical_time(tmp_path, fps):
    motion = load_amass(write_amass(tmp_path / "source.npz", fps, int(fps * 3)))
    output = resample_motion(motion, 20.0)
    times = np.arange(len(output.rotations)) / 20.0
    np.testing.assert_allclose(output.translations[:, 0], times, atol=1e-6)
    angles = Rotation.from_matrix(output.rotations[:, 0]).as_rotvec()[:, 2]
    np.testing.assert_allclose(angles, times * 0.4, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(output.rotations), 1, atol=1e-6)


def test_reject_velocity_arrays_and_missing_source_rate(tmp_path):
    np.save(tmp_path / "humanml.npy", np.zeros((100, 263)))
    with pytest.raises(ValueError, match="velocity"):
        load_amass(tmp_path / "humanml.npy")
    np.savez(tmp_path / "features.npz", features=np.zeros((100, 263)))
    with pytest.raises(ValueError, match="velocity"):
        load_amass(tmp_path / "features.npz")
    np.savez(tmp_path / "no_rate.npz", poses=np.zeros((100, 156)), trans=np.zeros((100, 3)))
    with pytest.raises(ValueError, match="frame rate"):
        load_amass(tmp_path / "no_rate.npz")


def test_mirror_is_involution_and_swaps_left_right(tmp_path):
    motion = load_amass(write_amass(tmp_path / "motion.npz"))
    mirrored = mirror_motion(motion)
    restored = mirror_motion(mirrored)
    np.testing.assert_allclose(restored.rotations, motion.rotations)
    np.testing.assert_allclose(restored.translations, motion.translations)
    np.testing.assert_allclose(mirrored.translations[:, 0], -motion.translations[:, 0])
    np.testing.assert_allclose(np.linalg.det(mirrored.rotations), 1, atol=1e-6)
    reflection = np.diag([-1., 1., 1.])
    np.testing.assert_allclose(mirrored.rotations[:, 2], reflection @ motion.rotations[:, 1] @ reflection)


def test_official_index_prefix_offset_and_noninteger_fps(tmp_path):
    raw = load_amass(write_amass(tmp_path / "source.npz", fps=59.94, frames=600))
    start, end, scale = humanml_source_interval(raw, "MPI_HDM05", 10, 50)
    # Original notebook uses floor(59.94/20)=2, and removes nominal 3*20 frames.
    assert start == pytest.approx((60 + 10) * 2 / 59.94)
    assert end == pytest.approx((60 + 50) * 2 / 59.94)
    assert scale == pytest.approx(40 / 59.94)
    output = resample_motion(raw, 20, start_time=start, end_time=end)
    assert output.translations[0, 0] == pytest.approx(start)


def test_annotations_full_and_timed_crops(tmp_path):
    captions = tmp_path / "000001.txt"
    captions.write_text("walk#walk/VERB#0#0\nturn#turn/VERB#1#2\nstand#stand/VERB#nan#nan\n")
    rows = read_captions(captions, 100, 20, time_scale=0.75)
    assert [(r["start_frame"], r["end_frame"]) for r in rows] == [(0, 100), (15, 30), (0, 100)]


def test_alias_resolution_and_no_humanact_inverse_kinematics(tmp_path):
    source = write_amass(tmp_path / "BMLrub" / "subject" / "run_stageii.npz")
    resolved, name = resolve_source(tmp_path, "./pose_data/BioMotionLab_NTroje/subject/run_poses.npy")
    assert resolved == source
    assert name == "BioMotionLab_NTroje"
    assert resolve_source(tmp_path, "./pose_data/humanact12/clip.npy") == (None, "humanact12")
    with pytest.raises(ValueError, match="escape"):
        resolve_source(tmp_path, "./pose_data/../../outside.npy")


def test_prepare_humanml_preserves_split_mirror_and_segment(tmp_path):
    write_amass(tmp_path / "amass" / "CMU" / "01" / "01_01_poses.npz", frames=300)
    index = tmp_path / "index.csv"
    with index.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "start_frame", "end_frame", "new_name"])
        writer.writeheader()
        writer.writerow({"source_path": "./pose_data/CMU/01/01_01_poses.npy", "start_frame": 20,
                         "end_frame": 80, "new_name": "000001.npy"})
        writer.writerow({"source_path": "./pose_data/humanact12/x.npy", "start_frame": 0,
                         "end_frame": -1, "new_name": "000002.npy"})
    texts = tmp_path / "texts"
    texts.mkdir()
    (texts / "000001.txt").write_text("left#left/ADJ#0#0\nturn left#turn/VERB left/ADJ#1#2\n")
    (texts / "M000001.txt").write_text("right#right/ADJ#0#0\n")
    split = tmp_path / "train.txt"
    split.write_text("000001\nM000001\n000002\n")
    manifest = prepare_humanml(tmp_path / "amass", index, texts, split, tmp_path / "processed")
    dataset = MotionDataset(manifest, require_text=True)
    assert len(dataset) == 3
    assert dataset[0]["text"] == "left"
    assert dataset[1]["text"] == "turn left"
    assert dataset[1]["translations"][0, 0].item() == pytest.approx(0.0)
    assert dataset[1]["translations"][-1, 0].item() == pytest.approx(0.95)
    assert len(dataset[1]["rotations"]) == 20
    assert dataset[2]["id"] == "M000001"
    assert dataset[2]["translations"][0, 0].item() == pytest.approx(0.0)
    assert dataset[2]["translations"][-1, 0].item() == pytest.approx(-2.95)
    report = json.loads((tmp_path / "processed" / "preparation_report.json").read_text())
    assert report["humanact12_excluded"] == ["000002"]
    batch = collate_motion([dataset[0], dataset[1]])
    assert batch["rotations"].shape == (2, 60, 22, 3, 3)
    assert batch["mask"].sum(1).tolist() == [60, 20]
    assert torch.equal(batch["rotations"][1, 30, 0], torch.eye(3))


def test_public_samples_cannot_be_used_as_paired_training_data(tmp_path):
    write_amass(tmp_path / "raw" / "sample.npz")
    manifest = prepare_samples(tmp_path / "raw", tmp_path / "prepared")
    with pytest.raises(ValueError, match="real paired captions"):
        MotionDataset(manifest, require_text=True)
    sample = MotionDataset(manifest)[0]
    assert sample["text"] == ""
    assert not sample["paired"]


def test_missing_data_fails_instead_of_silent_incomplete_benchmark(tmp_path):
    index = tmp_path / "index.csv"
    index.write_text("source_path,start_frame,end_frame,new_name\n./pose_data/CMU/01/missing_poses.npy,0,60,000001.npy\n")
    split = tmp_path / "train.txt"
    split.write_text("000001\n")
    with pytest.raises(FileNotFoundError, match="Missing"):
        prepare_humanml(tmp_path, index, tmp_path, split, tmp_path / "processed")


def test_training_crop_is_random_and_respects_caption_interval(tmp_path):
    write_amass(tmp_path / "raw" / "sample.npz", frames=300)
    manifest = prepare_samples(tmp_path / "raw", tmp_path / "processed")
    row = json.loads(manifest.read_text())
    row.update(start_frame=20, end_frame=60)
    manifest.write_text(json.dumps(row) + "\n")
    fixed = MotionDataset(manifest, max_frames=10)[0]
    angle = Rotation.from_matrix(fixed["rotations"][:, 0].numpy()).as_rotvec()[0, 2]
    assert angle == pytest.approx(0.4)
    dataset = MotionDataset(manifest, max_frames=10, random_crop=True)
    torch.manual_seed(7)
    first = dataset[0]
    torch.manual_seed(7)
    assert torch.equal(dataset[0]["rotations"], first["rotations"])
    assert not torch.equal(fixed["rotations"], first["rotations"])
    assert torch.equal(first["translations"][0], torch.zeros(3))
    for _ in range(20):
        sample = dataset[0]
        angles = Rotation.from_matrix(sample["rotations"][:, 0].numpy()).as_rotvec()[:, 2]
        assert angles[0] >= 0.4 - 1e-6
        assert angles[-1] <= 59 / 20 * 0.4 + 1e-6


def canonical_fixture(frames=40):
    """Synthetic nondegenerate skeleton fixture, never an evaluation asset."""
    from flashmo.data._humanml.param_util import t2m_raw_offsets, t2m_kinematic_chain
    lengths = np.array([0, .11, .11, .12, .4, .4, .12, .4, .4, .12, .15, .15,
                        .12, .2, .2, .15, .25, .25, .24, .24, .12, .12])
    offsets = t2m_raw_offsets * lengths[:, None]
    joints = np.zeros((frames, 22, 3), dtype=np.float32)
    joints[:, 0, 1] = 1.0
    joints[:, 0, 0] = np.arange(frames) * 0.01
    for chain in t2m_kinematic_chain:
        for parent, child in zip(chain[:-1], chain[1:]):
            joints[:, child] = joints[:, parent] + offsets[child]
    return joints


def test_official_humanml_export_features_and_canonicalization():
    from flashmo.data.humanml_export import joints_to_humanml, normalize_humanml
    joints = canonical_fixture()
    features, canonical = joints_to_humanml(joints, joints)
    assert features.shape == (39, 263)
    assert np.isfinite(features).all()
    assert canonical.shape == (40, 22, 3)
    assert canonical[..., 1].min() == pytest.approx(0, abs=1e-6)
    np.testing.assert_allclose(canonical[0, 0, [0, 2]], 0, atol=1e-6)
    # Uniform-skeleton rescaling should remove a global size difference.
    scaled, _ = joints_to_humanml(joints * 2.0, joints)
    np.testing.assert_allclose(features, scaled, atol=2e-5)
    np.testing.assert_array_equal(normalize_humanml(features, np.zeros(263), np.ones(263)), features)


def test_export_requires_real_body_assets_and_valid_reference(tmp_path):
    from flashmo.data.humanml_export import joints_to_humanml, reconstruct_smplh
    with pytest.raises(FileNotFoundError, match="Licensed SMPL-H"):
        reconstruct_smplh(np.zeros((2, 22, 3, 3)), np.zeros((2, 3)), tmp_path / "missing.npz")
    with pytest.raises(ValueError, match="zero-length"):
        joints_to_humanml(canonical_fixture(), np.zeros((2, 22, 3)))
