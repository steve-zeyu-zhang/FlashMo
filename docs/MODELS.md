# Motion models

MotionSiT combines the SO(3) posterior `R_z = R_mu @ Exp(v)` with
frequency-aware sparsification. Training uses original AMASS rotations matched
to the HumanML3D index. Rotation inputs come from those original sequences;
HumanML3D velocity features use a different representation.

## Architecture

| Component | Full configuration | Design notes |
|---|---|---|
| Motion VAE | 4 encoder and 4 decoder layers, width 512, 4 heads, temporal compression 4 | Learned temporal down/up projections |
| Rotation latent | 22 SO(3) factors per compressed frame | Factor count is configured separately from transformer hidden width |
| Translation latent | Independent 3-D diagonal Gaussian per compressed frame | Euclidean root motion, separate from SO(3) rotations |
| MotionSiT | Transformer decoder, 8 layers, width 512, 4 heads; continuous time embedding | Clean-prefix conditioning and text cross-attention |
| Text | Frozen CLIP ViT-B/32 projected pooled token | An optional alternative encoder is documented in Acknowledgements |
| Frequency-aware sparse multi-head attention (Sparse-MHA) | First half of heads Lo-Fi, other half Hi-Fi; mass threshold 0.95 | Frequency-aware head partition and attention-based adaptive token selection |

The MotionSiT implementation uses a transformer decoder: concatenate a clean
motion prefix with the noisy suffix, project motion frames, add positional
encodings, cross-attend to text plus a time embedding, then discard prefix
outputs. Self-attention is bidirectional within that call; motion
generation becomes autoregressive through repeated prefix completion calls.
The outputs are body angular velocities and root translation velocities for
the geometric factorized interpolant (GFI). The decoder uses post-norm
`nn.TransformerDecoderLayer` blocks. Compatible backbone initialization and
source attribution are described in [Acknowledgements](#acknowledgements).

## Manifold posterior and reconstruction

The encoder predicts an SO(3) mean through a 6D frame and a diagonal tangent
standard deviation. It samples `epsilon ~ N(0,I)`, then computes
`R_z = R_mu @ Exp(std * epsilon)`. Every latent rotation is a matrix on SO(3).
The loss uses a wrapped group density with exponential-map volume correction
and preimages at angle offsets `2*pi*k`, `k=-3,...,3`. Mathematical
references are credited in [Acknowledgements](#acknowledgements).

The default rotation prior is `Exp(N(0,I))`; the Monte Carlo KL uses
`log q_SO3 - log p_SO3`. This default aligns its base distribution with the
geometric transport implementation. `prior_type: haar` instead selects the
uniform Haar prior, with density `1/(8*pi^2)` under
the adopted Riemannian volume. The rotation KL is never replaced by the
Euclidean Gaussian KL. Finite wrapping is an approximation and `num_wraps`
is configurable. A single Monte Carlo KL estimate may be negative.

The decoder predicts joint rotation matrices and root translations. Training
combines squared geodesic rotation error, root translation MSE, and KL. Optional
joint-position MSE uses differentiable forward kinematics when actual rest
offsets and ground-truth joint positions are available. SMPL joint reconstruction
requires the licensed body model. The default loss leaves the optional
joint-position term inactive. The 22 learned latent factors need not correspond one-to-one to
the 22 decoded physical joints.

## Frequency-aware sparsification

### Frequency-aware head partition

Attention heads are partitioned into Hi-Fi and Lo-Fi groups, with
full temporal attention for Hi-Fi heads and selected temporal keys/values for
Lo-Fi heads. The head partition is fixed. `frequency_energy` provides an FFT
diagnostic without changing head assignments.

### Attention-based adaptive token selection

The implementation applies softmax to attention logits, sums each key column,
chooses the smallest number
of keys covering the configured attention mass, normalizes column scores by their
nonzero counts, and retains the top keys. Selection keeps **all queries**.
Both training and inference use this selection. The top-k decision is
discrete; selected attention values and projections remain differentiable.

`selection_queries=0` computes exact full-query selection scores in chunks.
A positive value activates an optional query-subset approximation, using a
uniformly spaced subset of each valid sequence. Exact selection has
quadratic scoring cost. Acceleration and AIT require separate benchmarks
with matched sequence lengths, hardware, and attention backends.

### Sparse multi-head attention and CUDA execution

The frequency-aware sparse multi-head attention (Sparse-MHA) implementation
supports these `attention_backend` settings:

- `sdpa`: PyTorch scaled-dot-product attention over physically gathered keys.
- `block_sparse`: the compiled sparse CUDA kernel credited in [Acknowledgements](#acknowledgements), with an error if unavailable.
- `auto`: use that CUDA kernel when supported, otherwise SDPA; each layer exposes `last_backend` and `backend_reason`.

The CUDA kernel masks 128-token blocks, while FlashMo selects individual
tokens. To preserve the exact operator, the adapter packs selected keys per
batch/head into variable-length sequences and activates their compacted
blocks. It does **not** round selection to blocks of the original sequence.
The mask is dense within compacted sequences; sparsity comes from removing
unselected keys before launching the kernel. This detail matters when
interpreting speed measurements, especially for short motion latents.

Supported kernel inputs are CUDA FP16/BF16 and head dimensions 32, 64, 128.
The full configuration's width 512 / 4 heads gives dimension 128. Explicit `block_sparse`
uses BF16 for Q/K/V even if the surrounding model is FP32, then returns FP32
outputs. `attention_dtype` can override the kernel precision. Geometry and
wrapped-density calculations remain FP32; whole-model autocast is not needed.

## Interfaces

```python
posterior = vae.encode(rotations, translations, mask=valid, sample=True)
# posterior["rotations"]: [B, ceil(F/4), K, 3, 3]
# posterior["translations"]: [B, ceil(F/4), 3]
# posterior["mask"]: [B, ceil(F/4)] (True = valid)

decoded = vae.decode(posterior["rotations"], posterior["translations"],
                     length=F, mask=posterior["mask"])

text = text_encoder(["A person walks forward."])
velocity = model(R_t, root_t, t,
                 text_features=text["features"], text_mask=text["mask"],
                 mask=latent_mask,
                 prefix_rotations=clean_prefix_R,
                 prefix_translations=clean_prefix_root)
# rotation_velocity: [B,L,K,3], body-trivialized angular velocity
# translation_velocity: [B,L,3]
```

Text-conditioned generation requires frozen text features.
`force_unconditional=True` enables unconditional generation; smoke
configurations use unconditional training.

## Acknowledgements

The MotionSiT decoder implementation adapts the DiP backbone, including its
clean-prefix structure, bidirectional attention within each call, and post-norm
decoder layers. The implementation class is `FlashMoDiP`.
Frequency-aware head partition builds on the LITv2 HiLo concept, replacing its
image windows and pooling with full Hi-Fi temporal attention and selected
Lo-Fi temporal keys/values. Attention-based adaptive token selection draws on
ZipVL's cumulative-mass rule and optional query-subset approximation.
The optional DistilBERT text encoder also follows the DiP implementation.

- [DiP / MDM](https://github.com/GuyTevet/motion-diffusion-model): `ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778`, MIT, copyright Guy Tevet 2022.
- [Falorsi et al.](https://arxiv.org/abs/1807.04689) and [Lie VAE](https://github.com/pimdh/lie-vae): `c473d91c494b9e80f6ac2998b1b6b927d09df259`, MIT; manifold posterior and group-density reparameterization.
- [LITv2 HiLo](https://github.com/ziplab/LITv2/blob/main/hilo.py): `7501c8662991cad6db780b61fac0886b56a76588`, Apache 2.0, copyright Zhuang Intelligent Processing Lab 2022.
- [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention): `49d6c39e4dc0303442cda3bb758b3925d4399c49`; optional compiled upstream dependency.
- [ZipVL, Equations 6–9](https://arxiv.org/html/2410.08584v2): algorithmic reference for token selection.

Optional `--dip-checkpoint path/to/dip.pt` initializes compatible backbone
weights. The underlying `load_dip_checkpoint(path_or_state_dict)` helper returns
`loaded`, `missing`, `mismatched`, and `unexpected` lists/maps. Compatible
self-attention, cross-attention, feed-forward, normalization, time and text
projection weights can transfer. Rotation input/output projections need
training, and changing the text encoder can also prevent text projection
transfer. A loaded DiP checkpoint alone is not a trained FlashMo model.
