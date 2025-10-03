"""Generate native AMASS-frame joint rotations with a trained FlashMo checkpoint."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from flashmo.geometry import so3_log
from flashmo.models import SO3MotionVAE, FlashMoDiP, FrozenTextEncoder
from flashmo.transport import GeometricInterpolant, GeometricTransport
from flashmo.utils import file_sha256, load_checkpoint, seed_everything, write_json


def load_models(path, device):
    checkpoint = load_checkpoint(path, "flow")
    config = checkpoint["config"]
    vae = SO3MotionVAE(**config["vae"]).to(device).eval()
    vae.load_state_dict(checkpoint["vae"])
    network = FlashMoDiP(**config["model"]).to(device).eval()
    network.load_state_dict(checkpoint["model"])
    transport = GeometricTransport(GeometricInterpolant(**config["transport"]))
    return checkpoint, vae, network, transport


@torch.no_grad()
def generate(vae, network, transport, config, frames, batch_size=1, text_features=None,
             text_mask=None, guidance=1., steps=10, method="euler", prefix=None, generator=None):
    """DiP-style chunk rollout; clean context is never integrated as noise.

    Output contains requested NEW frames only. Prefix, if supplied, is a dict
    with rotations[B,P,22,3,3], translations[B,P,3] in native AMASS coordinates.
    Each chunk's translation frame is anchored at its context's first frame.
    """
    if frames <= 0 or batch_size <= 0:
        raise ValueError("frames and batch_size must be positive")
    device = next(network.parameters()).device
    context_frames = config["train"]["context_frames"]
    prediction_frames = config["train"]["prediction_frames"]
    kwargs = {"force_unconditional": text_features is None}
    if text_features is not None:
        kwargs.update(text_features=text_features, text_mask=text_mask)
    history_rotations, history_translations = [], []
    if prefix is not None:
        if prefix["rotations"].shape[0] != batch_size:
            raise ValueError("Prefix batch size does not match requested batch")
        history_rotations.append(prefix["rotations"].to(device))
        history_translations.append(prefix["translations"].to(device))
    generated_rotations, generated_translations = [], []
    remaining = frames
    while remaining:
        length = min(remaining, prediction_frames)
        chunk_kwargs = dict(kwargs)
        origin = torch.zeros(batch_size, 1, 3, device=device)
        if history_rotations and context_frames > 0:
            r_context = torch.cat(history_rotations, dim=1)[:, -context_frames:]
            x_context = torch.cat(history_translations, dim=1)[:, -context_frames:]
            origin = x_context[:, :1]
            encoded = vae.encode(r_context, x_context - origin, sample=False)
            chunk_kwargs.update(prefix_rotations=encoded["rotations"],
                prefix_translations=encoded["translations"], prefix_mask=encoded["mask"])

        def velocity(rotations, translations, t, **extra):
            return network.forward_with_cfg(rotations, translations, t, guidance_scale=guidance, **extra)

        latent_length = math.ceil(length / vae.compression)
        rotations, translations = transport.sample(velocity,
            (batch_size, latent_length, vae.latent_joints), steps=steps, method=method,
            model_kwargs=chunk_kwargs, device=device, generator=generator)
        decoded = vae.decode(rotations, translations, length)
        r_new, x_new = decoded["rotations"], decoded["translations"] + origin
        history_rotations.append(r_new)
        history_translations.append(x_new)
        generated_rotations.append(r_new)
        generated_translations.append(x_new)
        remaining -= length
        # Only the tail context is needed by the next DiP prediction.
        if context_frames:
            history_rotations = [torch.cat(history_rotations, 1)[:, -context_frames:]]
            history_translations = [torch.cat(history_translations, 1)[:, -context_frames:]]
    return torch.cat(generated_rotations, 1), torch.cat(generated_translations, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--frames", type=int, default=196)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--prefix", help="Prepared or generated .npz rotations+translations (single motion)")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--method", choices=["euler", "midpoint"])
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/samples")
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    device = torch.device(args.device)
    torch.set_num_threads(min(8, __import__("os").cpu_count() or 1))
    seed_everything(args.seed)
    checkpoint, vae, network, transport = load_models(args.checkpoint, device)
    config = checkpoint["config"]
    conditioning = config["text"]["kind"]
    if conditioning == "none" and args.text.strip():
        parser.error("This unconditional smoke checkpoint has no text training; omit --text or use a paired-trained checkpoint")
    text_features = text_mask = None
    if conditioning != "none":
        encoder = FrozenTextEncoder(**config["text"]).to(device).eval()
        encoded = encoder([args.text] * args.num_samples)
        text_features, text_mask = encoded["features"], encoded["mask"]
    prefix = None
    if args.prefix:
        with np.load(args.prefix, allow_pickle=False) as data:
            prefix = {k: torch.from_numpy(np.array(data[k], dtype=np.float32)).unsqueeze(0)
                      .repeat(args.num_samples, *([1] * data[k].ndim)) for k in ("rotations", "translations")}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    steps = args.steps if args.steps is not None else config["sample"]["steps"]
    method = args.method or config["sample"]["method"]
    guidance = args.guidance if args.guidance is not None else config["sample"]["guidance"]
    rotations, translations = generate(vae, network, transport, config, args.frames, args.num_samples,
        text_features=text_features, text_mask=text_mask, guidance=guidance, steps=steps, method=method,
        prefix=prefix, generator=torch.Generator(device=device).manual_seed(args.seed))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    identity = torch.eye(3, device=device)
    orthogonality = float((rotations.transpose(-1, -2) @ rotations - identity).abs().max())
    determinant = float((torch.linalg.det(rotations) - 1).abs().max())
    if not torch.isfinite(rotations).all() or not torch.isfinite(translations).all():
        raise FloatingPointError("Generation produced nonfinite outputs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    axis_angle = so3_log(rotations).cpu().numpy()
    filenames = []
    for index in range(args.num_samples):
        path = output / f"sample_{index:03d}.npz"
        np.savez_compressed(path, rotations=rotations[index].cpu().numpy(),
            translations=translations[index].cpu().numpy(), poses_body=axis_angle[index].reshape(args.frames, -1),
            fps=np.array(config["data"]["fps"]), text=np.array(args.text),
            coordinate_system=np.array("AMASS native z-up; root global, body local"))
        filenames.append(str(path))
    backend = [{"layer": i, "backend": layer.self_attn.last_backend,
                "reason": getattr(layer.self_attn, "backend_reason", "")}
               for i, layer in enumerate(network.seqTransDecoder.layers)]
    write_json(output / "generation.json", {"checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint), "seed": args.seed, "frames": args.frames,
        "num_samples": args.num_samples, "text": args.text, "steps_per_chunk": steps, "method": method,
        "guidance": guidance, "elapsed_seconds": elapsed, "includes_warmup": True,
        "orthogonality_max_error": orthogonality, "determinant_max_error": determinant,
        "smoke_only": checkpoint.get("smoke_only", False), "attention_backends": backend,
        "files": filenames, "benchmark_reproduced": False})
    print(f"Saved {len(filenames)} motion(s) to {output}; SO(3) max error {orthogonality:.3g}")


if __name__ == "__main__":
    main()
