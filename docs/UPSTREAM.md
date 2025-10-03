# Upstream snapshots and licenses

`third_party/upstream.json` pins source revisions, usage, and licenses.
Complete upstream license texts are retained byte-for-byte
under `third_party/licenses/`, including their original copyright notices.
The manifest records SHA256 hashes of those files and how each source is used.

MotionSiT predicts manifold velocities using the geometric factorized
interpolant (GFI). Frequency-aware sparsification combines frequency-aware
head partition with attention-based adaptive token selection. The CUDA backend
stays external and is called through an adapter. The source projects and their
roles are credited in [Acknowledgements](#acknowledgements). These code licenses
do not grant model weights, AMASS licenses, or SMPL body-model assets.

## Inspect exact source snapshots

```bash
# Fetch all five implementation repositories, shallow and detached.
python scripts/fetch_upstream.py

# Read-only verification of existing revisions and license hashes; no network.
python scripts/fetch_upstream.py --check
```

Snapshots go to ignored `third_party/checkouts/`. Existing directories with a
different revision
or tracked edits are rejected; the script never resets or overwrites them.
Build submodules can be included when creating new snapshots with
`--submodules`. Existing checkouts remain read-only, including submodules.
For the optional CUDA backend and its dependencies, use
`scripts/install_block_sparse.py` as described in the README.

The fetch script installs nothing and the default local model runtime does not
import these checkouts. SDPA operation needs no upstream clone; text encoders
can use downloaded local Hugging Face caches for offline runs. Selecting the
compiled sparse attention backend requires installing that optional extension.

## Acknowledgements

| Upstream | Pinned revision | Use | License |
|---|---|---|---|
| [DiP / MDM](https://github.com/GuyTevet/motion-diffusion-model) | `ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778` | MotionSiT decoder and prefix/text conditioning | MIT |
| [SiT](https://github.com/willisma/SiT) | `cbde832a40b153ccc79603412409da9c9b0c568c` | Velocity matching / integration structure adapted for GFI | MIT |
| [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) | `49d6c39e4dc0303442cda3bb758b3925d4399c49` | Actual optional CUDA backend | BSD 3-Clause |
| [LITv2](https://github.com/ziplab/LITv2) | `7501c8662991cad6db780b61fac0886b56a76588` | HiLo head-partition design adapted for frequency-aware head partition | Apache 2.0 |
| [Lie VAE](https://github.com/pimdh/lie-vae) | `c473d91c494b9e80f6ac2998b1b6b927d09df259` | Falorsi SO(3) posterior and group density | MIT |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | `a75b4ac483166189a45290783cb0a18af5ff0ea5` | Pinned Block-Sparse-Attention build dependency | BSD 3-Clause |

The local implementations are modified adaptations. MotionSiT adapts DiP's
decoder to predict manifold velocities instead of denoised HumanML3D features.
SiT's transport is replaced by GFI, and LITv2 HiLo's image windows/pooling are
replaced by FlashMo's full Hi-Fi attention and selected Lo-Fi temporal keys.

Attention-based adaptive token selection implements the cumulative scoring in
[ZipVL v2, Equations 6–9](https://arxiv.org/html/2410.08584v2).
The manifest records the paper and project links. The scorer is implemented
within FlashMo; no ZipVL source files or website assets are included.

To fetch only selected source projects for an attribution audit, use their
manifest identifiers, for example `python scripts/fetch_upstream.py --only mdm lie_vae`.
