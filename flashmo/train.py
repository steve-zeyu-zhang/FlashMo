"""Two-stage training: manifold VAE, then DiP geometric velocity matching.

Works with python -m flashmo.train and torchrun. The global batch size is
divided across ranks. Geometry is evaluated in float32 for stable SO(3) maps.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from flashmo.data import MotionDataset, collate_motion
from flashmo.models import SO3MotionVAE, FlashMoDiP, FrozenTextEncoder
from flashmo.transport import GeometricInterpolant, GeometricTransport
from flashmo.utils import atomic_save, file_sha256, load_checkpoint, load_config, seed_everything, to_device, write_json


def encode_training_window(vae, batch, config, use_prefix):
    """Encode prefix separately so its encoder never observes the future."""
    rotations, translations, mask = (batch[k] for k in ("rotations", "translations", "mask"))
    context = config["train"]["context_frames"]
    prediction = config["train"]["prediction_frames"]
    # Every example needs at least one target frame. Short motions use no prefix.
    use_prefix = use_prefix and context > 0 and bool((mask.sum(-1) > context).all())
    start = context if use_prefix else 0
    stop = min(start + prediction, rotations.shape[1])
    origin = translations[:, :1]
    target = vae.encode(rotations[:, start:stop], translations[:, start:stop] - origin,
                        mask[:, start:stop], sample=False)
    kwargs = {}
    if use_prefix:
        prefix = vae.encode(rotations[:, :context], translations[:, :context] - origin,
                            mask[:, :context], sample=False)
        kwargs.update(prefix_rotations=prefix["rotations"], prefix_translations=prefix["translations"],
                      prefix_mask=prefix["mask"])
    return target, kwargs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["vae", "flow"], required=True)
    parser.add_argument("--config", default="configs/humanml3d.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vae-checkpoint")
    parser.add_argument("--dip-checkpoint", help="Optional original DiP weights; transfer compatible tensors and save a report")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-unpaired", action="store_true",
                        help="Explicitly allow uncaptioned samples for engineering tests; not T2M training.")
    args = parser.parse_args(argv)
    if args.dip_checkpoint and (args.stage != "flow" or args.resume):
        parser.error("--dip-checkpoint is only for a fresh flow run; do not combine with --resume")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    config = load_config(args.config)
    manifest = args.manifest or config["data"]["manifest"]
    config["data"]["manifest"] = str(Path(manifest).resolve())
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}" if args.device.startswith("cuda") else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    else:
        device = torch.device(args.device)
    seed_everything(config.get("seed", 42) + rank)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    batch_size = config["train"]["batch_size"]
    if batch_size < world or batch_size % world:
        raise ValueError("Global batch size must be positive and divisible by WORLD_SIZE")
    flow = args.stage == "flow"
    if flow and config["text"]["kind"] == "none" and not args.allow_unpaired:
        raise ValueError("text.kind=none is an unconditional smoke setting; explicitly pass --allow-unpaired")
    dataset = MotionDataset(manifest, max_frames=(config["train"]["context_frames"] +
        config["train"]["prediction_frames"] if flow else config["data"]["max_frames"]),
        require_text=flow and not args.allow_unpaired, random_crop=True)
    sampler = DistributedSampler(dataset, shuffle=True, seed=config.get("seed", 42)) if world > 1 else None
    loader = DataLoader(dataset, batch_size=batch_size // world, shuffle=sampler is None,
                        sampler=sampler, num_workers=config["data"]["num_workers"],
                        collate_fn=collate_motion, pin_memory=device.type == "cuda", drop_last=False)
    if not len(loader):
        raise ValueError("Training manifest has no usable motions")
    vae = SO3MotionVAE(**config["vae"]).to(device)
    text_encoder = None
    dip_report = None
    resumed = load_checkpoint(args.resume, args.stage) if args.resume else None
    if resumed and resumed["config"]["vae"] != config["vae"]:
        raise ValueError("Resume VAE architecture differs from configuration")
    if flow:
        if resumed:
            vae.load_state_dict(resumed["vae"])
        elif args.vae_checkpoint:
            source = load_checkpoint(args.vae_checkpoint, "vae")
            if source["config"]["vae"] != config["vae"]:
                raise ValueError("VAE checkpoint architecture differs from flow config")
            vae.load_state_dict(source["model"])
        else:
            parser.error("--stage flow requires --vae-checkpoint or --resume")
        vae.eval().requires_grad_(False)
        config["model"]["cond_mask_prob"] = config["train"]["condition_dropout"]
        network = FlashMoDiP(**config["model"]).to(device)
        if args.dip_checkpoint:
            dip_report = network.load_dip_checkpoint(args.dip_checkpoint)
            if not dip_report["loaded"]:
                raise ValueError("DiP checkpoint contains no compatible tensors; inspect its model size and format")
        transport = GeometricTransport(GeometricInterpolant(**config["transport"]))
        if config["text"]["kind"] != "none":
            text_encoder = FrozenTextEncoder(**config["text"]).to(device).eval()
            if text_encoder.output_dim != config["model"]["text_dim"]:
                raise ValueError("Text encoder output_dim does not match model.text_dim")
    else:
        network = vae
    if resumed:
        for key in (["model", "text", "transport"] if flow else []):
            if resumed["config"][key] != config[key]:
                raise ValueError(f"Resume {key} configuration mismatch")
        network.load_state_dict(resumed["model"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=config["train"]["learning_rate"],
                                  weight_decay=config["train"].get("weight_decay", 0.01))
    if resumed:
        optimizer.load_state_dict(resumed["optimizer"])
    wrapped = DistributedDataParallel(network, device_ids=[device.index] if device.type == "cuda" else None,
                                      find_unused_parameters=flow) if world > 1 else network
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    step = int(resumed["step"]) if resumed else 0
    start_epoch = int(resumed.get("epoch", 0)) if resumed else 0
    epochs = config["train"]["flow_epochs" if flow else "vae_epochs"]
    log_every = config["train"].get("log_every", 10)
    save_every = config["train"].get("save_every", 1000)
    losses = []
    started = time.monotonic()
    provenance = {"manifest": str(Path(manifest).resolve()), "sha256": file_sha256(manifest),
                  "motions": len(dataset), "allow_unpaired": args.allow_unpaired,
                  "world_size": world, "device": str(device), "torch": str(torch.__version__)}
    if rank == 0:
        write_json(output / "config.json", config)
        write_json(output / "provenance.json", provenance)
        if dip_report is not None:
            write_json(output / "dip_initialization.json", dip_report)
        print(json.dumps({"stage": args.stage, "motions": len(dataset), "device": str(device),
                          "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad)}), flush=True)

    def save(epoch):
        if rank != 0:
            return
        checkpoint = {"format_version": 1, "stage": args.stage, "config": config,
                      "model": network.state_dict(), "optimizer": optimizer.state_dict(),
                      "step": step, "epoch": epoch, "provenance": provenance,
                      "torch_rng_state": torch.get_rng_state(), "smoke_only": args.allow_unpaired}
        if flow:
            checkpoint["vae"] = vae.state_dict()
        atomic_save(checkpoint, output / "last.pt")

    epoch = start_epoch
    for epoch in range(start_epoch, epochs):
        if args.max_steps is not None and step >= args.max_steps:
            break
        if sampler:
            sampler.set_epoch(epoch)
        network.train()
        for batch in loader:
            if args.max_steps is not None and step >= args.max_steps:
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if flow:
                with torch.no_grad():
                    use_prefix = torch.rand((), device=device).item() < config["train"].get("prefix_probability", 0.5)
                    latent, kwargs = encode_training_window(vae, batch, config, use_prefix)
                    if text_encoder is not None:
                        texts = [s if s else "" for s in batch["text"]]
                        encoded_text = text_encoder(texts)
                        kwargs.update(text_features=encoded_text["features"], text_mask=encoded_text["mask"])
                    else:
                        kwargs["force_unconditional"] = True
                    bins = config["train"].get("time_bins", 1000)
                    t = (torch.randint(bins, (len(batch["rotations"]),), device=device).float() + 0.5) / bins
                values = transport.training_losses(wrapped, latent["rotations"], latent["translations"],
                    t=t, mask=latent["mask"], model_kwargs=kwargs)
            else:
                result = wrapped(batch["rotations"], batch["translations"], batch["mask"])
                values = vae.loss(result, batch["rotations"], batch["translations"], mask=batch["mask"],
                                  kl_weight=config["train"]["kl_weight"])
            loss = values["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {args.stage} loss at step {step}; checkpoint not overwritten")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), config["train"]["grad_clip"],
                                                       error_if_nonfinite=True)
            optimizer.step()
            step += 1
            scalars = {k: float(v.detach()) for k, v in values.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
            if world > 1:
                keys = sorted(scalars)
                vector = torch.tensor([scalars[k] for k in keys], device=device)
                dist.all_reduce(vector)
                scalars = dict(zip(keys, (vector / world).tolist()))
            record = {"step": step, "epoch": epoch, **scalars, "grad_norm": float(grad_norm)}
            losses.append(scalars["loss"])
            if rank == 0:
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                if step == 1 or step % log_every == 0:
                    print(json.dumps(record), flush=True)
            if step % save_every == 0:
                save(epoch)
    save(epoch)
    if rank == 0:
        write_json(output / "summary.json", {"stage": args.stage, "step": step,
            "elapsed_seconds": time.monotonic() - started, "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "checkpoint": str((output / "last.pt").resolve()), "smoke_only": args.allow_unpaired,
            "benchmark_reproduced": False})
        print(f"Saved {output / 'last.pt'}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
