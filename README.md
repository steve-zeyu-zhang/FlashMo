<div align="center"><h1>FlashMo: Geometric Interpolants and Frequency-Aware Sparsity for Scalable Efficient Motion Generation<br>
<sub><sup><a href="https://neurips.cc/Conferences/2025">NeurIPS 2025</a></sup></sub>
</h1>

[Zeyu Zhang](https://steve-zeyu-zhang.github.io/)<sup>\*</sup>, [Yiran Wang](https://www.linkedin.com/in/yiran-wang-101739246)<sup>\*</sup>, [Danning Li](https://www.linkedin.com/in/danning-li-448039229/)<sup>\*</sup>, [Dong Gong](https://donggong1.github.io/), [Ian Reid](https://mbzuai.ac.ae/study/faculty/ian-reid/), [Richard Hartley](https://users.cecs.anu.edu.au/~hartley/)

<sup>*</sup>Equal contribution

[![Website](https://img.shields.io/badge/Website-Demo-fedcba?style=flat-square)](https://steve-zeyu-zhang.github.io/FlashMo/)
[![OpenReview](https://img.shields.io/badge/OpenReview-8c1b13?style=flat-square)](https://openreview.net/forum?id=B9T5kKcKTy)
[![BibTeX](https://img.shields.io/badge/BibTeX-Citation-eeeeee?style=flat-square)](#citation)

</div>

_Diffusion models have recently advanced 3D human motion generation by producing smoother and more realistic sequences from natural language. However, existing approaches face two major challenges: high computational cost during training and inference, and limited scalability due to reliance on U-Net inductive bias. To address these challenges, we propose FlashMo, a frequency-aware sparse motion diffusion model that prunes low-frequency tokens to enhance efficiency without custom kernel design. We further introduce MotionSiT, a scalable diffusion transformer based on a joint-temporal factorized interpolant with Lie group geodesics over SO(3) manifolds, enabling principled generation of joint rotations. Extensive experiments on the large-scale MotionHub V2 dataset and standard benchmarks including HumanML3D and KIT-ML demonstrate that our method significantly outperforms previous approaches in motion quality, efficiency, and scalability. Compared to the state-of-the-art 1-step distillation baseline, FlashMo reduces 12.9% inference time and FID by 34.1%._

![FlashMo teaser](https://steve-zeyu-zhang.github.io/FlashMo/static/images/teaser.svg)

## Usage

### Environment

Python 3.10+, PyTorch 2.6.0, and CUDA 12.4 are used for GPU training.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

# Build the CUDA backend for frequency-aware sparse attention on a Hopper GPU.
# Downloads a local CUDA compiler if a system toolkit is unavailable.
python scripts/install_block_sparse.py --download-cuda --arch 90 --jobs 4
```

The paper configuration uses the compiled sparse backend. Set `model.attention_backend: sdpa` in your config for the portable PyTorch implementation. SO(3) computations remain in float32; sparse attention uses bfloat16. [Pinned upstream sources and licenses](docs/UPSTREAM.md) are included.

### Data Preparation

Use **original AMASS SMPL-H rotations**, indexed by the HumanML3D subset. Training requires the original `poses`, `trans`, and frame-rate fields in the AMASS `.npz` files. The 263-dimensional HumanML3D features are used for evaluation, not as rotation inputs to the VAE.

1. Register at the [AMASS website](https://amass.is.tue.mpg.de/register.php), accept the dataset terms, and sign in to the [official download page](https://amass.is.tue.mpg.de/download.php). Download the **SMPL+H** releases of all 17 source datasets referenced by the HumanML3D index: `ACCAD`, `BMLhandball`, `BMLmovi`, `BioMotionLab_NTroje`, `CMU`, `DFaust_67`, `EKUT`, `Eyes_Japan_Dataset`, `HumanEva`, `KIT`, `MPI_HDM05`, `MPI_Limits`, `MPI_mosh`, `SFU`, `SSM_synced`, `TotalCapture`, and `Transitions_mocap`. Portal names may differ; [Data preparation](docs/DATA.md) lists supported aliases.
2. Extract the archives into `data/amass/`, preserving the dataset/subject/sequence structure, such as `data/amass/CMU/80/80_63_poses.npz`.
3. Download the complete [HumanML3D metadata](https://github.com/EricGuo5513/HumanML3D), including the source index, text annotations, and official splits:

```bash
python scripts/download_humanml_metadata.py --output data/humanml_metadata
```

4. Prepare the training, validation, and test rotation sequences:

```bash
for split in train val test; do
  python scripts/prepare_amass.py humanml \
    --amass-root data/amass \
    --index data/humanml_metadata/index.csv \
    --texts data/humanml_metadata/texts \
    --split "data/humanml_metadata/${split}.txt" \
    --output "data/humanml3d/${split}" --fps 20
done
```

The resulting manifests are saved to `data/humanml3d/{train,val,test}/manifest.jsonl`. Preparation checks for missing source files and reports omissions in `preparation_report.json`. HumanAct12 entries are excluded because they are not original AMASS sequences; this pipeline covers the AMASS portion of HumanML3D. See [Data preparation](docs/DATA.md) for source alignment and mirroring details.

### Training

```bash
# Stage 1: SO(3) motion VAE.
python -m flashmo.train --stage vae \
  --config configs/humanml3d.yaml --output outputs/vae

# Stage 2: MotionSiT with the geometric factorized interpolant (GFI).
python -m flashmo.train --stage flow \
  --config configs/humanml3d.yaml \
  --vae-checkpoint outputs/vae/last.pt --output outputs/flow
```

The paper configuration uses width 512, four attention heads, VAE depth four, temporal compression four, MotionSiT depth eight, `beta=0.5`, `gamma=0.95`, AdamW at `1e-4`, and global batch size 256. VAE and MotionSiT training use 6,000 and 3,000 epochs respectively. CLIP ViT-B/32 is frozen; DistilBERT is also supported as an optional text encoder.

Use `--batch-size` to fit GPU memory, `--max-steps` for a short run, and `--resume outputs/flow/last.pt` to continue training. For multiple GPUs, replace `python` with `torchrun --standalone --nproc_per_node=4`; batch size is global. Compatible backbone initialization is described in [Model acknowledgements](docs/MODELS.md#acknowledgements).

### Inference

```bash
python -m flashmo.sample \
  --checkpoint outputs/flow/last.pt \
  --text "A person walks forward and turns left." \
  --frames 196 --steps 10 --guidance 3.0 \
  --output outputs/generated
```

Outputs contain rotation matrices, body axis-angle poses, and root translations in native AMASS coordinates. MotionSiT generates longer sequences through clean-prefix continuation. Use `--prefix path/to/motion.npz` to condition on an existing motion. Ten Lie-Euler steps are used per chunk by default.

For HumanML3D evaluation, first reconstruct joints with licensed SMPL-H assets and convert them to the official 263-dimensional representation using [the export bridge](docs/DATA.md#export-to-the-humanml3d-evaluator-representation). Reconstruction diagnostics and metrics from pretrained evaluator embeddings are available through `python -m flashmo.evaluate --help`.

## Citation

```bibtex
@inproceedings{zhang2025flashmo,
  title={FlashMo: Geometric Interpolants and Frequency-Aware Sparsity for Scalable Efficient Motion Generation},
  author={Zhang, Zeyu and Wang, Yiran and Li, Danning and Gong, Dong and Reid, Ian and Hartley, Richard},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgements

We thank the authors of the following papers and open-source projects:

- [Human Motion Diffusion Model (MDM)](https://github.com/GuyTevet/motion-diffusion-model) and [CLoSD / DiP](https://github.com/GuyTevet/motion-diffusion-model/blob/main/DiP.md) for the motion backbone.
- [Explorations in Homeomorphic Variational Auto-Encoding](https://arxiv.org/abs/1807.04689) and [Reparameterizing Distributions on Lie Groups](https://arxiv.org/abs/1903.02958) for the SO(3) VAE formulation.
- [SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://github.com/willisma/SiT) for stochastic interpolants and velocity matching.
- [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) for the sparse CUDA attention backend.
- [Fast Vision Transformers with HiLo Attention (LITv2)](https://github.com/ziplab/LITv2) for frequency-separated attention heads.
- [ZipVL: Efficient Large Vision-Language Models with Dynamic Token Sparsification](https://arxiv.org/abs/2410.08584) for adaptive cumulative-attention token selection.
- [AMASS: Archive of Motion Capture as Surface Shapes](https://amass.is.tue.mpg.de/) for original motion rotations.
- [Generating Diverse and Natural 3D Human Motions from Text (HumanML3D)](https://github.com/EricGuo5513/HumanML3D) for subset indexing, text annotations, and the evaluation representation.
