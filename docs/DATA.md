# Original AMASS rotations and HumanML3D alignment

FlashMo reads the original AMASS `poses`, `trans`, and source frame rate from
SMPL-family `.npz` sequences. The first 66 pose coordinates are axis-angle
parameters for the root and 21 body joints. They become rotation matrices
`[frames, 22, 3, 3]`; translations have shape `[frames, 3]`. HumanML3D's
263-dimensional velocity/position feature vectors are not accepted as original
rotations. No inverse kinematics or reconstructed rotations are used.

Rotations use native AMASS coordinates: z-up, a global root rotation, and
parent-relative body rotations. Translations are the original SMPL global
translation parameters in metres. They are not the world-space pelvis joint:
the licensed body model must apply its shape-dependent rest-pelvis offset.
This convention is recorded in each manifest row. SMPL shape and
gender are retained in the prepared files when available. Licensed SMPL assets
are required for exact
mesh/joint reconstruction and the original HumanML3D evaluation conversion;
they are not required to read rotation matrices.

## The original-AMASS portion of HumanML3D

Obtain the original SMPL-H AMASS sequences from the
[official dataset portal](https://amass.is.tue.mpg.de/download.php) under its
access terms. Preserve the dataset/subject/sequence directory structure, for
example `data/amass/CMU/80/80_63_poses.npz`.

The metadata downloader fetches the official `index.csv`, split
files, and annotation archive from
[HumanML3D commit 9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23](https://github.com/EricGuo5513/HumanML3D/tree/9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23).
It produces 29,232 caption files and the official train/validation/test ID lists.

```bash
python scripts/download_humanml_metadata.py --output data/humanml_metadata

python scripts/prepare_amass.py humanml \
  --amass-root data/amass \
  --index data/humanml_metadata/index.csv \
  --texts data/humanml_metadata/texts \
  --split data/humanml_metadata/train.txt \
  --output data/humanml3d/train --fps 20
```

Repeat for `val.txt` and `test.txt`, writing separate output directories. Missing
AMASS sequences or caption files fail preparation by default and are listed in
`preparation_report.json`. `--allow-missing` explicitly permits a partial
development subset and records every omitted ID. Evaluation of a partial
subset must account for those omissions. HumanAct12 records are excluded and reported:
they are part of HumanML3D, but they do not provide original AMASS rotations.

Alignment follows the official
[raw pose processing notebook](https://github.com/EricGuo5513/HumanML3D/blob/9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23/raw_pose_processing.ipynb):

1. Resolve each `index.csv` source path to an original `.npz`. Release aliases
   include `BioMotionLab_NTroje`/`BMLrub`, `DFaust_67`/`DFaust`,
   `MPI_HDM05`/`HDM05`, `MPI_Limits`/`PosePrior`, `MPI_mosh`/`MoSh`, and
   `Eyes_Japan_Dataset`/`EyesJapanDataset`, `SSM_synced`/`SSM`,
   `Transitions_mocap`/`Transitions`, and `*_poses.npz`/`*_stageii.npz` names.
2. Apply the dataset-specific prefix removal: 3 nominal seconds for
   Eyes Japan and HDM05, 1 for TotalCapture and MPI Limits, and 0.5 for
   Transitions. Then apply the indexed start and exclusive end frames.
3. The original notebook sampled every `floor(source_fps / 20)` source frames.
   Map its index coordinates through that exact stride to original timestamps,
   then resample at a true 20 Hz using SLERP. This avoids using a rounded source
   FPS or incorrectly treating index coordinates as original AMASS frames.
4. Scale timed captions by `20 * floor(source_fps / 20) / source_fps` to preserve
   their correspondence to the original notebook's nominal 20-Hz grid.
   Standard 60/120-Hz data has a scale of one. The scale is recorded explicitly.
5. Produce an `Mxxxxxx` motion only if that ID appears in the requested split.
   Mirroring reflects x, swaps the eight SMPL left/right joint pairs, and
   conjugates rotations by the reflection matrix. Its caption is read from
   the official `Mxxxxxx.txt`; directional words are never guessed.
6. Honor nonzero start/end times in each caption. Full-sequence `0#0` labels
   select the whole clip. Each manifest row represents one real caption and
   its matching motion interval.

The official preprocessing later changes axes, normalizes the skeleton, and
derives T2M features. The rotation dataset stores native AMASS poses for the
SO(3) VAE. The optional export bridge below performs that
conversion when licensed assets and the real target skeleton are supplied.

## Loader contract

```python
from torch.utils.data import DataLoader
from flashmo.data import MotionDataset, collate_motion

dataset = MotionDataset(
    "data/humanml3d/train/manifest.jsonl",
    max_frames=196,
    require_text=True,
    random_crop=True,
)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_motion)
```

Each batch contains `rotations [B,T,22,3,3]`, `translations [B,T,3]`,
`mask [B,T]`, `text`, `id`, `fps`, and `paired`. `mask=True` means a valid frame.
Padding rotations are identity matrices. Captions select their annotated
interval before optional `max_frames` truncation. `random_crop=True` draws a
training crop inside that interval using PyTorch's seeded worker RNG; the
default `False` starts at the interval's first frame. The returned translation
is recentered at the selected crop's first frame, consistently for VAE and flow
training. Original world translations remain in prepared NPZ files. For fair
evaluation, use the evaluation protocol's length handling, set
`random_crop=False`, and disable arbitrary truncation.

The preparation report and manifests retain original file hashes, source FPS,
crop timestamps, annotation time scaling, mirror flags, and caption IDs, so
each training sample can be traced to its source.

## Export to the HumanML3D evaluator representation

The optional bridge reconstructs SMPL-H joints with the licensed AMASS body
model, changes native axes to HumanML's `[-x,z,y]` convention, retargets bone
lengths to the real `000021.npy` reference skeleton, places the sequence on the
floor, centers it, aligns the initial facing direction, and runs the official
HumanML3D 263-dimensional feature extraction. The pinned upstream helpers and
their license are in `flashmo/data/_humanml/`.

Install `human_body_prior` separately for this optional export and supply an
Extended SMPL-H `model.npz` obtained from the official model provider. Use the
same body-model gender/shape convention for reference and generated motions.
`--reference-joints` must point to the real HumanML3D `joints/000021.npy` produced
by its licensed preprocessing.

```bash
python scripts/export_humanml.py \
  --input outputs/generated/sample_000.npz \
  --output outputs/generated_humanml.npy \
  --body-model /path/to/body_models/smplh/male/model.npz \
  --reference-joints /path/to/HumanML3D/joints/000021.npy \
  --fps 20 --device cuda \
  --save-joints outputs/generated_joints.npy
```

Input NPZ fields are `rotations [T,22,3,3]`, `translations [T,3]`, and optional
`fps`, `mask`, and `betas`. Batched `[B,T,...]` outputs are supported via
`--sample-index`. The export checks SO(3) validity and requires 20-Hz input.
Ten supplied shape coefficients are used when present; otherwise the licensed
model's zero mean-shape coefficients are used and recorded explicitly. Hands
are set to zero because this model predicts the 22 body joints. The original
AMASS body shape is retained in prepared input files, when available.
The exporter passes `translations` into `BodyModel(trans=...)` and reads its
`Jtr` output, which includes the body model's shape-dependent pelvis offset.
Recentered training trajectories differ
from original AMASS trajectories by a constant displacement, which the
HumanML floor/origin canonicalization subsequently removes.

The resulting `.npy` has `[T-1,263]` raw features because velocity extraction
uses consecutive frames. Pass `--mean /path/to/Mean.npy --std /path/to/Std.npy`
only when the downstream evaluator expects already normalized features;
otherwise let its existing loader normalize them once. A neighboring JSON
file records body/reference/input hashes and normalization status.

The bridge requires licensed body models and reference joints and stops when
either is missing. Canonicalization and feature extraction have unit-test
coverage. End-to-end SMPL-H export remains unvalidated without the licensed
assets. FID, retrieval, and diversity evaluation additionally require the
pretrained text/motion evaluator, matching real-data split, sampling protocol,
and complete benchmark data.
