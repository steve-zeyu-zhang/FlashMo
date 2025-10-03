# Verification

Validation environment: Python 3.10.12, PyTorch 2.6.0+cu124,
CUDA 12.4, NVIDIA H200 GPUs, Transformers 4.48.3, and the compiled
sparse attention backend (version 0.0.2). Upstream revisions and license checks match
the [pinned source manifest](../third_party/upstream.json).

## Automated checks

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q
```

**56 passed, no skips.** Three deprecation warnings originate in the preserved
upstream HumanML3D conversion helpers.

Coverage includes SO(3) maps and gradients at zero/near pi, degenerate 6D
rotation inputs, right-Jacobian velocity targets, wrapped density normalization,
masked VAE training, source time alignment and mirrored captions,
attention-based adaptive token selection, and actual CUDA forward/backward agreement with SDPA. Integration
tests run both training stages, restore the MotionSiT optimizer from a checkpoint,
generate multiple continuation chunks, check deterministic sampling and prefix
world translations, and validate reconstruction and embedding metrics.

## Training and inference runs

These runs used two authentic public AMASS sequences from the AMASS authors,
with explicit unpaired-data mode. Their HumanML3D membership is unverified and
they do not supply text labels. All downloaded data, checkpoints, and generated
motions remain local and Git-ignored.

| Run | Configuration | Result | Local artifacts |
| --- | --- | --- | --- |
| VAE | Width 64, 2 layers, 100 steps | Finite losses/gradients; loss 2.0240 to 0.2357 | `outputs/smoke/vae/` |
| MotionSiT | Width 64, 2 layers, 100 steps | Training and checkpoint reload completed | `outputs/smoke/flow/` |
| VAE | Width 512, 4 layers, 20 steps | Finite losses/gradients; loss 2.5542 to 0.0861 | `outputs/paper_smoke/vae/` |
| MotionSiT | Width 512, 8 layers, 20 steps | Compiled sparse attention and frozen CLIP path completed | `outputs/paper_smoke/flow/` |
| Two-GPU DDP | Small VAE and MotionSiT, 3 steps each | Both stages completed; saved tensors finite | `outputs/ddp_smoke/` |
| Generation | 2 motions × 196 frames, 10 Lie-Euler steps per chunk, CFG 3 | All 8 layers used the compiled sparse backend; finite rotations/translations | `outputs/paper_smoke/generated/` |

The 196-frame generation exercised a real CLIP text prompt and autoregressive
continuation. Maximum rotation orthogonality and determinant errors were both
`4.77e-7`. Model, optimizer, and embedded VAE tensors in all saved checkpoints
were checked for finite values. MotionSiT training losses are stochastic; these short runs
do not establish convergence or meaningful text adherence.

The complete pinned HumanML3D metadata was also checked: 29,232 caption files
parse successfully, split IDs match the index, and original/mirrored pairs
remain in the same disjoint split. The metadata-only download command requires
no AMASS example files.

## Limits

The checks cover numerical behavior and short training runs. Benchmark
validation requires full training on the licensed AMASS-derived HumanML3D
corpus. End-to-end export with licensed SMPL-H
assets and the real reference skeleton has not been run; its conversion
subroutines are covered by fixtures. MotionHub V2 pretraining, KIT-ML results,
official evaluator confidence intervals, and paper timing/quality claims are
outside this validation scope. See [Training and evaluation](TRAINING.md).
