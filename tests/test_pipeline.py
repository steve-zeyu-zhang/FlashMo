"""Small, self-contained integration fixtures; these are not benchmark motions."""

import json
import math

import numpy as np
import pytest
import torch
import yaml

from flashmo.data import MotionDataset, collate_motion
from flashmo.evaluate import embedding_metrics, reconstruction
from flashmo.geometry import so3_exp
from flashmo.sample import generate, load_models, main as sample_main
from flashmo.train import encode_training_window, main as train_main
from flashmo.utils import load_checkpoint


@pytest.fixture(scope="module")
def trained_pipeline(tmp_path_factory):
    """Run the real training entrypoint once per stage on tiny rotation clips."""
    root = tmp_path_factory.mktemp("flashmo_pipeline")
    rng = np.random.default_rng(23)
    rows = []
    for index, length in enumerate((11, 9)):
        tangent = torch.from_numpy(rng.normal(0, 0.15, (length, 22, 3)).astype(np.float32))
        rotations = so3_exp(tangent).numpy()
        translations = rng.normal(0, 0.02, (length, 3)).astype(np.float32).cumsum(axis=0)
        name = f"unit_fixture_{index}.npz"
        np.savez(root / name, rotations=rotations, translations=translations)
        rows.append({"id": f"unit_fixture_{index}", "path": name, "fps": 20,
                     "text": "", "paired": False})
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = {
        "seed": 23,
        "data": {"manifest": str(manifest), "max_frames": 12, "num_workers": 0, "fps": 20},
        "vae": {"num_joints": 22, "latent_joints": 3, "hidden_dim": 16, "num_layers": 1,
                "num_heads": 2, "compression": 2, "ff_size": 32, "dropout": 0,
                "prior_type": "wrapped_normal"},
        "model": {"latent_joints": 3, "hidden_dim": 16, "num_layers": 1, "num_heads": 2,
                  "ff_size": 32, "text_dim": 8, "dropout": 0, "beta": 0.5, "gamma": 0.9,
                  "cond_mask_prob": 0, "attention_backend": "sdpa"},
        "text": {"kind": "none"},
        "transport": {"temporal_power": 1.0, "spatial_power": 1.0,
                      "noise_schedule": "linear", "noise_scale": 1.0},
        "train": {"batch_size": 2, "learning_rate": 1e-3, "weight_decay": 0.01,
                  "vae_epochs": 4, "flow_epochs": 4, "kl_weight": 1e-4, "grad_clip": 1.0,
                  "log_every": 1, "save_every": 1, "condition_dropout": 0.0,
                  "time_bins": 1000, "context_frames": 4, "prediction_frames": 4,
                  "prefix_probability": 1.0},
        "sample": {"steps": 2, "method": "euler", "guidance": 1.0},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    common = ["--config", str(config_path), "--max-steps", "1", "--device", "cpu",
              "--allow-unpaired"]
    train_main(["--stage", "vae", "--output", str(root / "vae"), *common])
    vae_path = root / "vae" / "last.pt"
    train_main(["--stage", "flow", "--output", str(root / "flow"),
                "--vae-checkpoint", str(vae_path), *common])
    return {"root": root, "config": config, "config_path": config_path,
            "manifest": manifest, "vae_path": vae_path,
            "flow_path": root / "flow" / "last.pt"}


def test_real_training_updates_both_stages_and_serializes_optimizer(trained_pipeline):
    for stage in ("vae", "flow"):
        checkpoint = load_checkpoint(trained_pipeline[f"{stage}_path"], stage)
        assert checkpoint["step"] == 1
        assert checkpoint["smoke_only"] is True
        optimizer_states = list(checkpoint["optimizer"]["state"].values())
        assert optimizer_states
        assert all(float(state["step"]) == 1 for state in optimizer_states)
        assert any(torch.count_nonzero(state["exp_avg"]) for state in optimizer_states)
        assert all(torch.isfinite(tensor).all() for tensor in checkpoint["model"].values())
        record = json.loads((trained_pipeline["root"] / stage / "metrics.jsonl").read_text().splitlines()[0])
        assert math.isfinite(record["loss"])
        assert record["grad_norm"] > 0


def test_flow_checkpoint_reloads_and_resumes_optimizer(trained_pipeline):
    resumed_directory = trained_pipeline["root"] / "resumed_flow"
    train_main(["--stage", "flow", "--config", str(trained_pipeline["config_path"]),
                "--resume", str(trained_pipeline["flow_path"]), "--output", str(resumed_directory),
                "--max-steps", "2", "--device", "cpu", "--allow-unpaired"])
    resumed = load_checkpoint(resumed_directory / "last.pt", "flow")
    assert resumed["step"] == 2
    assert all(float(state["step"]) == 2 for state in resumed["optimizer"]["state"].values())


@pytest.mark.parametrize("with_prefix", [False, True])
def test_checkpoint_rollout_is_deterministic_and_stays_on_so3(trained_pipeline, with_prefix):
    checkpoint, vae, network, transport = load_models(trained_pipeline["flow_path"], torch.device("cpu"))
    config = checkpoint["config"]
    torch.manual_seed(11)
    prefix = None
    if with_prefix:
        # Shorter than context_frames and not divisible by compression, so
        # prefix encoding must construct and propagate the correct latent mask.
        prefix = {"rotations": so3_exp(torch.randn(2, 3, 22, 3) * 0.1),
                  "translations": torch.randn(2, 3, 3)}
    features = torch.randn(2, 3, 8)
    text_mask = torch.tensor([[True, True, False], [True, False, False]])
    kwargs = dict(frames=11, batch_size=2, text_features=features, text_mask=text_mask,
                  guidance=2.0, steps=2, method="euler", prefix=prefix)
    first = generate(vae, network, transport, config,
                     generator=torch.Generator().manual_seed(81), **kwargs)
    # Deterministic prefix encoding currently computes a stochastic KL, but its
    # global RNG consumption must not change the separately seeded sampler.
    torch.manual_seed(987654)
    second = generate(vae, network, transport, config,
                      generator=torch.Generator().manual_seed(81), **kwargs)
    assert first[0].shape == (2, 11, 22, 3, 3)
    assert first[1].shape == (2, 11, 3)
    for one, two in zip(first, second):
        torch.testing.assert_close(one, two, rtol=0, atol=0)
        assert torch.isfinite(one).all()
        assert not one.requires_grad
    rotation = first[0]
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation,
                               torch.eye(3).expand_as(rotation), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(rotation.shape[:-2]),
                               rtol=1e-5, atol=1e-5)


def test_prefix_encoding_does_not_see_future_or_padding(trained_pipeline):
    _, vae, network, _ = load_models(trained_pipeline["flow_path"], torch.device("cpu"))
    dataset = MotionDataset(trained_pipeline["manifest"], max_frames=8)
    batch = collate_motion([dataset[0], dataset[1]])
    batch["mask"][1, 5:] = False
    config = trained_pipeline["config"]
    target, conditioning = encode_training_window(vae, batch, config, use_prefix=True)
    assert target["mask"].tolist() == [[True, True], [True, False]]
    assert conditioning["prefix_rotations"].shape[:3] == (2, 2, 3)
    assert conditioning["prefix_mask"].tolist() == [[True, True], [True, True]]
    corrupted_padding = {key: value.clone() if isinstance(value, torch.Tensor) else value
                         for key, value in batch.items()}
    corrupted_padding["rotations"][1, 5:] = so3_exp(torch.tensor([2.0, -1.0, 0.4]))
    corrupted_padding["translations"][1, 5:] = 999
    other, other_conditioning = encode_training_window(vae, corrupted_padding, config, use_prefix=True)
    torch.testing.assert_close(target["rotations"], other["rotations"])
    torch.testing.assert_close(target["translations"], other["translations"])
    changed_future = {key: value.clone() if isinstance(value, torch.Tensor) else value
                      for key, value in batch.items()}
    changed_future["rotations"][:, 4:] = so3_exp(torch.tensor([1.0, 0.4, -0.8]))
    changed_future["translations"][:, 4:] += 5
    _, future_conditioning = encode_training_window(vae, changed_future, config, use_prefix=True)
    for key in ("prefix_rotations", "prefix_translations", "prefix_mask"):
        torch.testing.assert_close(conditioning[key], other_conditioning[key])
        torch.testing.assert_close(conditioning[key], future_conditioning[key])
    # Guidance zero exactly selects unconditional velocities while retaining
    # the clean prefix and valid suffix mask.
    kwargs = {**conditioning, "mask": target["mask"],
              "text_features": torch.randn(2, 2, 8),
              "text_mask": torch.tensor([[True, True], [True, False]])}
    t = torch.tensor([0.3, 0.7])
    guided = network.forward_with_cfg(target["rotations"], target["translations"], t,
                                      guidance_scale=0, **kwargs)
    unconditional = network(target["rotations"], target["translations"], t,
                             force_unconditional=True, **kwargs)
    for key in guided:
        torch.testing.assert_close(guided[key], unconditional[key])
        assert torch.count_nonzero(guided[key][1, 1:]) == 0


def test_reconstruction_is_repeatable_after_checkpoint_reload(trained_pipeline):
    kwargs = dict(checkpoint_path=trained_pipeline["vae_path"], manifest=trained_pipeline["manifest"],
                  device=torch.device("cpu"))
    first = reconstruction(**kwargs)
    torch.manual_seed(9191)
    second = reconstruction(**kwargs)
    assert first == second
    assert first["motions"] == 2 and first["frames"] == 20
    assert math.isfinite(first["rotation_mean_degrees"])
    assert math.isfinite(first["translation_rmse_m"])
    assert first["benchmark_reproduced"] is False


def test_prefix_world_translation_is_preserved_during_rollout(trained_pipeline):
    checkpoint, vae, network, transport = load_models(trained_pipeline["flow_path"], torch.device("cpu"))
    prefix = {"rotations": torch.eye(3).expand(1, 3, 22, 3, 3).clone(),
              "translations": torch.tensor([[[0., 0., 0.], [0.1, 0.2, 0.], [0.2, 0.3, 0.]]])}
    shift = torch.tensor([[[3.0, -2.0, 4.0]]])
    shifted_prefix = {"rotations": prefix["rotations"], "translations": prefix["translations"] + shift}
    first = generate(vae, network, transport, checkpoint["config"], frames=7, prefix=prefix,
                     steps=2, generator=torch.Generator().manual_seed(5))
    shifted = generate(vae, network, transport, checkpoint["config"], frames=7, prefix=shifted_prefix,
                       steps=2, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(first[0], shifted[0], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(first[1] + shift, shifted[1], atol=3e-6, rtol=3e-6)


def test_sampler_cli_rejects_zero_steps(trained_pipeline):
    with pytest.raises((ValueError, SystemExit)):
        sample_main(["--checkpoint", str(trained_pipeline["flow_path"]), "--steps", "0",
                     "--frames", "4", "--device", "cpu",
                     "--output", str(trained_pipeline["root"] / "invalid_sampling")])


def test_embedding_fid_identity_mean_shift_and_paired_retrieval():
    real = np.random.default_rng(17).normal(size=(64, 5))
    identity = embedding_metrics(real, real.copy(), real.copy(), seed=42)
    assert identity["fid"] < 1e-10
    assert identity["matching_distance"] == 0
    assert identity["retrieval_samples"] == 64
    for k in (1, 2, 3):
        assert identity[f"r_precision_top_{k}"] == 1
    shift = np.array([1.0, -0.2, 0.3, 0.0, 0.8])
    shifted = embedding_metrics(real, real + shift)
    assert shifted["fid"] == pytest.approx(float(shift @ shift), abs=1e-10)
    assert shifted["diversity"] == pytest.approx(identity["diversity"], abs=1e-10)


def test_embedding_retrieval_direction_matches_humanml3d_evaluator():
    # Asymmetric neighborhoods distinguish text->motion from motion->text.
    rng = np.random.default_rng(0)
    generated = rng.normal(size=(32, 4))
    text = generated * rng.uniform(0.2, 3.0, size=(32, 1)) + rng.normal(size=(32, 4)) * 0.3
    distances = np.linalg.norm(text[:, None] - generated[None], axis=-1)
    order = distances.argsort(axis=-1)
    actual = embedding_metrics(generated, generated, text)
    for k in (1, 2, 3):
        expected = np.any(order[:, :k] == np.arange(32)[:, None], axis=-1).mean()
        assert actual[f"r_precision_top_{k}"] == pytest.approx(expected)


def test_embedding_rejects_unpaired_or_invalid_inputs():
    array = np.random.default_rng(31).normal(size=(32, 4))
    with pytest.raises(ValueError, match="paired"):
        embedding_metrics(array, array, array[:-1])
    with pytest.raises(ValueError, match="at least 32"):
        embedding_metrics(array[:20], array[:20], array[:20])
    invalid = array.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        embedding_metrics(array, invalid)
