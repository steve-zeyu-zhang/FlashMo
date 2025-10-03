# Training and evaluation

MotionSiT uses the geometric factorized interpolant (GFI) and frequency-aware
sparsification to generate motion from text. Training has two stages: a temporal
SO(3) VAE, followed by velocity matching in the VAE latent space.

## Training and inference

Prepare the original AMASS rotations, HumanML3D captions, and official splits
as described in [Data preparation](DATA.md). The training manifest must contain
paired motions and text.

```bash
source .venv/bin/activate

python -m flashmo.train --stage vae --config configs/humanml3d.yaml \
  --output outputs/vae

python -m flashmo.train --stage flow --config configs/humanml3d.yaml \
  --vae-checkpoint outputs/vae/last.pt --output outputs/flow

python -m flashmo.sample --checkpoint outputs/flow/last.pt \
  --text "A person walks forward and turns left." \
  --frames 196 --steps 10 --guidance 3.0 --output outputs/generated

python -m flashmo.evaluate reconstruction \
  --checkpoint outputs/vae/last.pt \
  --manifest data/humanml3d/val/manifest.jsonl \
  --output outputs/reconstruction.json
```

`configs/humanml3d.yaml` contains the full architecture and training schedule
and uses the compiled sparse attention backend. `configs/smoke.yaml` provides
reduced dimensions for functional tests. Use `--batch-size` to control the
global batch and `--max-steps` to run a limited number of optimizer steps.
Replace `python` with `torchrun --standalone --nproc_per_node=4` for multi-GPU
training.

Resume a stage with `--resume path/to/last.pt`. Resume restores the model,
optimizer, and saved step/epoch; it does not restore a partially consumed
data-loader iterator or guarantee bitwise equivalence to an uninterrupted
multi-worker run. Sampling uses a separate seeded generator.

Metrics, dataset hashes, saved configuration, and checkpoints are written under
the configured output directory. Generated files and `generation.json` record
rotation validity, backend selection, seed, and checkpoint hash.

## Configuration

- The rotation latent has 22 SO(3) factors, independent of transformer width
  512. Learned stride-four projections perform temporal downsampling and
  upsampling.
- Root translations use a separate Euclidean latent. The default VAE loss is
  squared rotation geodesic error, root MSE, and KL with weight `1e-4`.
  Optional FK loss requires rest-joint offsets and is disabled by default.
- The default signal schedules are `alpha_T=t` and `alpha_S=t`, giving a product
  of `t^2`. The default noise schedule is `1-t`; `noise_schedule: paper_sqrt`
  selects the square-root alternative.
- Sampling predicts body angular velocities and integrates rotations by right
  multiplication. The SO(3) right Jacobian accounts for noncommuting
  exponential-coordinate paths. Sampling uses an ODE; an SDE sampler is not
  implemented.
- Autoregressive generation uses 20 clean context frames and predicts 40 frames
  per call. Half of training calls omit context to support initial generation.
  The prefix is encoded separately to prevent future-frame leakage through
  the VAE.
- The frozen text encoder is CLIP ViT-B/32. Defaults are guidance `3`, weight
  decay `0.01`, condition dropout `0.1`, and gradient clipping `1`.
- Adaptive token selection uses exact cumulative attention scores by default.
  Optional query subsampling approximates those scores. Dense selection
  scoring remains quadratic; benchmark the complete attention path when
  measuring speedups.

See [Models](MODELS.md), [Math](MATH.md), and [Upstream sources](UPSTREAM.md)
for architecture, equations, and dependencies. [Verification results](VERIFICATION.md)
describes functional coverage and validation limits.

## Benchmark evaluation

Full paired training requires the licensed AMASS files referenced by HumanML3D
and separate official train/validation/test manifests. Record missing sources
and excluded HumanAct12 entries; HumanAct12 motions are not original AMASS
sequences.

For benchmark metrics, convert generated rotations using licensed SMPL-H
assets and the HumanML reference skeleton. Use the official frozen motion/text
evaluator with its normalization and length protocol, and run 20 evaluation
repeats for confidence intervals. `flashmo.evaluate embeddings` computes FID,
diversity, matching distance, and 32-candidate text-to-motion retrieval from
those evaluator embeddings. Multi-modality and full 20-run orchestration use
the official evaluator. Rotation reconstruction errors are reported separately
from embedding metrics.

End-to-end SMPL-H export has not been validated with licensed body assets and
the reference skeleton; the canonicalization and feature-extraction routines
have fixture coverage. Functional checks also do not validate MotionHub V2
pretraining, KIT-ML benchmark results, perceptual generation quality, or H20
timings. Those require complete training and dataset-specific evaluation.

## Acknowledgements

MotionSiT adapts the transformer decoder, clean-prefix conditioning, and
optional DistilBERT text encoder from
[DiP / Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model/blob/main/DiP.md).
GFI adapts velocity matching and integration structure from
[SiT](https://github.com/willisma/SiT). The manifold posterior uses the
SO(3) construction of [Falorsi et al.](https://arxiv.org/abs/1807.04689) and the
[Lie-VAE reference implementation](https://github.com/pimdh/lie-vae).

Frequency-aware head partition draws on
[LITv2 / HiLo attention](https://github.com/ziplab/LITv2).
Adaptive token selection uses the cumulative attention scoring in
[ZipVL](https://arxiv.org/abs/2410.08584v2).
The compiled attention backend is
[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention).
Source revisions, attribution, and licenses are recorded in
[Upstream sources](UPSTREAM.md).
