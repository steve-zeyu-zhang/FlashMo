# Reproduction status and commands

This document describes training, inference, and evaluation for MotionSiT with
the geometric factorized interpolant (GFI) and frequency-aware sparsification.
Implementation settings and the scope of benchmark verification are detailed below.

## What is implemented

- Original AMASS rotations matched to HumanML3D source indices, captions, and
  splits, with native sampling rates, exact source offsets, mirroring, and masks.
- A temporal SO(3) VAE with the `R_mu @ Exp(v)` posterior,
  wrapped group-density KL, and a separate Euclidean root translation branch.
- MotionSiT with text cross-attention and a clean motion prefix, predicting
  angular and root translation velocities.
- The geometric factorized interpolant (GFI), exact body-velocity targets using
  the SO(3) right Jacobian, and manifold-preserving ODE sampling.
- Frequency-aware head partition, attention-based adaptive token selection,
  and sparse multi-head attention using a compiled kernel on compacted
  selected keys and values.
- Two training stages, optimizer checkpoints/resume, multi-GPU DDP, text-guided
  sampling, variable lengths, and autoregressive rollout.
- Original HumanML3D canonicalization and 263-feature export with licensed
  SMPL-H assets installed separately; reconstruction and embedding-based metrics.

See [models](MODELS.md), [math](MATH.md), [data](DATA.md), and
[upstream provenance](UPSTREAM.md) for equations, source revisions, and details.
The completed execution checks are recorded in [Verification results](VERIFICATION.md).

## Small-sample execution check

These commands use two public AMASS example motions. Their
membership in HumanML3D is not established and they have no paired text. They
are explicitly **unconditional engineering checks**, not text-to-motion
training data.

```bash
source .venv/bin/activate
python scripts/download_amass_samples.py --output data/amass_samples
python scripts/prepare_amass.py samples \
  --amass-root data/amass_samples --output data/processed_samples

python -m flashmo.train --stage vae --config configs/smoke.yaml \
  --max-steps 100 --allow-unpaired --output outputs/smoke/vae
python -m flashmo.train --stage flow --config configs/smoke.yaml \
  --vae-checkpoint outputs/smoke/vae/last.pt \
  --max-steps 100 --allow-unpaired --output outputs/smoke/flow
python -m flashmo.sample --checkpoint outputs/smoke/flow/last.pt \
  --frames 100 --num-samples 2 --steps 10 --output outputs/smoke/generated
python -m flashmo.evaluate reconstruction \
  --checkpoint outputs/smoke/vae/last.pt \
  --manifest data/processed_samples/manifest.jsonl \
  --output outputs/smoke/reconstruction.json
python -m pytest -q
```

`configs/smoke.yaml` intentionally reduces width/depth. It is not the paper
architecture. `configs/humanml3d.yaml` provides the reported architecture and
training hyperparameters; it expects the compiled attention backend. For an
architecture-sized smoke run, add `--manifest data/processed_samples/manifest.jsonl
--batch-size 2 --num-workers 0 --max-steps 20 --allow-unpaired` to the paper
configuration in each stage, using separate output directories.

Metrics, dataset hashes, saved configuration, and checkpoints are written under
the configured output directory. Sample files and `generation.json` record
rotation validity, backend choice, seed, and checkpoint hash. These artifacts,
downloaded assets, local build tools, and virtual environments are Git-ignored.

## Explicit reconstruction choices

The paper and discussion do not fully specify every implementation detail:

1. The number of SO(3) factors is set to 22 independently of the transformer
   width 512. Temporal down/up sampling uses learned stride-four projections.
2. Root translations use a separate Euclidean latent. The default VAE loss is
   squared rotation geodesic error plus root MSE and KL with weight `1e-4`.
   Optional FK loss needs actual rest-joint offsets and is inactive by default.
3. Both default signal schedules are linear (`alpha_T=t`, `alpha_S=t`), so
   their product is `t^2`. The default noise schedule is `1-t`. The appendix's
   square-root noise alternative is available as `noise_schedule: paper_sqrt`.
4. Sampling predicts body angular velocities and integrates rotations by right
   multiplication. The Jacobian correction is necessary for the paper's
   noncommuting exponential-coordinate path. Sampling integrates this ODE;
   a manifold SDE sampler is not implemented.
5. This MotionSiT autoregressive configuration uses 20 clean context frames and
   predicts 40 frames per call. Half of the
   training calls omit context to support initial generation. The prefix is
   encoded separately to prevent the VAE from exposing future frames.
6. The default frozen text encoder is CLIP ViT-B/32, as specified by FlashMo.
   Guidance 3, weight decay 0.01, condition dropout 0.1, and gradient clipping 1
   are configurable engineering choices where not specified in the paper.
7. Attention-based adaptive token selection computes exact cumulative attention
   scores by default; optional query subsampling is an approximation. Dense
   selection scoring remains quadratic, so no speedup is claimed from the
   existence of a sparse kernel alone.

Seeding makes a fresh run repeatable in a given environment, and sampling uses
its own generator. Resume restores model and optimizer and continues from the
saved step/epoch; it does not promise bitwise equivalence to an uninterrupted
multi-worker run or restore a partially consumed data-loader iterator.

## Completing benchmark reproduction

Full paired training requires the original licensed AMASS files referenced by
HumanML3D. The public examples cannot replace that corpus. Prepare separate
official train/val/test manifests and record all excluded HumanAct12/missing
sources, then train the VAE and flow stages to completion. The subset excludes
HumanAct12 entries because their original motions are not AMASS sequences.

For paper-comparable metrics, convert generated rotations with real SMPL-H
assets and the HumanML reference skeleton, use the official frozen motion/text
evaluator with its normalization and length protocol, and repeat evaluation
with the paper's 20-run confidence intervals. `flashmo.evaluate embeddings`
computes FID, diversity, matching distance and 32-candidate text-to-motion
retrieval from those evaluator embeddings. It does not create a
replacement evaluator or label raw rotation errors as FID. Multi-modality and
full 20-run benchmark orchestration are left to that official evaluator.

The licensed SMPL-H reconstruction stage cannot be validated end to end without
the body assets and the real reference skeleton. Its canonicalization/feature
extraction subroutines are tested with fixtures. MotionHub V2 pretraining,
KIT-ML results, perceptual generation quality, and the reported H20 timings have
not been reproduced by the small-sample execution checks.

## Acknowledgements

MotionSiT's implementation adapts the transformer decoder, clean-prefix
conditioning, and optional DistilBERT text encoder from
[DiP / Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model/blob/main/DiP.md).
The GFI implementation adapts velocity matching and integration structure from
[SiT](https://github.com/willisma/SiT); the manifold posterior follows
[Falorsi et al.](https://arxiv.org/abs/1807.04689) and the
[Lie-VAE reference implementation](https://github.com/pimdh/lie-vae).

Frequency-aware head partition draws on
[LITv2 / HiLo attention](https://github.com/ziplab/LITv2).
Attention-based adaptive token selection independently adapts the cumulative
attention scoring described in [ZipVL](https://arxiv.org/abs/2410.08584v2).
The compiled sparse
attention backend is [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention).
These dependencies retain their original attribution and licenses.
Exact revisions, source mappings, and license texts
are recorded in [upstream provenance](UPSTREAM.md).
