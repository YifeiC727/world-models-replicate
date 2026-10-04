"""Reproducible CarRacing-v3 implementation of the 2018 World Models pipeline."""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
import tomllib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .carracing_data import (collect_rollouts, EpisodeChunkSampler, FrameDataset,
                             LatentSequenceDataset, split_episodes)
from .carracing_env import CarRacingAdapter
from .data import frames_to_tensor
from .models import VAE, MDNRNN, LinearController
from .pipeline import device_name


def load_config(path):
    with open(path, "rb") as stream:
        return tomllib.load(stream)


def paths(config):
    root = Path(config["experiment"]["workdir"])
    return root, root / "record", root / "series"


def load_splits(config):
    """Episode names per split, frozen in ``splits.json`` on first use."""
    root, record, _ = paths(config); path = root / "splits.json"
    names = sorted(p.name for p in record.glob("episode_*.npz"))
    if path.exists():
        splits = json.loads(path.read_text())
        missing = set(names) - {n for split in splits.values() for n in split}
        if missing: raise ValueError(f"{len(missing)} episodes are not in {path}; delete it to re-split")
        return splits
    data = config.get("data", {})
    splits = split_episodes(names, data.get("validation_fraction", .1), data.get("test_fraction", .1),
                            config["experiment"]["seed"])
    path.write_text(json.dumps(splits, indent=1) + "\n")
    print("split " + " ".join(f"{k}={len(v)}" for k, v in splits.items()))
    return splits


def split_files(directory, names): return [directory / name for name in names]


@torch.no_grad()
def vae_metrics(model, dataset, device, batch_size=256):
    """Held-out VAE loss; reconstruction decodes the posterior mean, so it is deterministic."""
    model.eval(); totals = {"loss": 0., "reconstruction": 0., "kl": 0.}
    for x in DataLoader(dataset, batch_size=batch_size):
        x = x.to(device); mu, logvar = model.encode(x)
        loss, parts = model.loss(model.decode(mu), x, mu, logvar)
        totals["loss"] += loss.item() * len(x)
        for key in parts: totals[key] += parts[key].item() * len(x)
    return {key: value / len(dataset) for key, value in totals.items()}


@torch.no_grad()
def mdn_nll(model, dataset, device, batch_size=32):
    model.eval(); total = 0.
    for z0, actions, z1 in DataLoader(dataset, batch_size=batch_size):
        total += model.nll(z0.to(device), actions.to(device), z1.to(device)).item() * len(z0)
    return total / len(dataset)


def save_reconstructions(model, dataset, path, device, count=8):
    model.eval()
    x = torch.stack([dataset[i] for i in np.linspace(0, len(dataset)-1, count, dtype=int)]).to(device)
    with torch.no_grad():
        mu, _ = model.encode(x); recon = model.decode(mu)
    pair = torch.cat((x.cpu(), recon.cpu()), dim=0).permute(0, 2, 3, 1).numpy()
    pair = (pair.clip(0, 1) * 255).astype(np.uint8)
    canvas = np.concatenate([np.concatenate(pair[:count], axis=1),
                             np.concatenate(pair[count:], axis=1)], axis=0)
    Image.fromarray(canvas).save(path)


def train_vae(config, device, resume=False):
    root, record, _ = paths(config); root.mkdir(parents=True, exist_ok=True)
    settings = config["vae"]; chunk = settings.get("shuffle_chunk_episodes", 16)
    splits = load_splits(config)
    dataset = FrameDataset(split_files(record, splits["train"]), cache_size=chunk)
    validation = FrameDataset(split_files(record, splits["validation"]))
    sampler = EpisodeChunkSampler(dataset, chunk, seed=config["experiment"]["seed"])
    loader = DataLoader(dataset, batch_size=settings["batch_size"], sampler=sampler, num_workers=0)
    model = VAE(settings["latent_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), settings["learning_rate"])
    ckpt_path = root / "vae.pt"; start = 0
    if resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"]); optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["epoch"]
    sampler.epoch = start
    for epoch in range(start, settings["epochs"]):
        model.train(); totals = {"loss": 0., "reconstruction": 0., "kl": 0.}
        for x in loader:
            x = x.to(device); recon, mu, logvar = model(x)
            loss, parts = model.loss(recon, x, mu, logvar,
                                     kl_tolerance=settings["kl_tolerance"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            totals["loss"] += loss.item() * len(x)
            for key in parts: totals[key] += parts[key].item() * len(x)
        metrics = {key: value / len(dataset) for key, value in totals.items()}
        metrics.update({f"val_{k}": v for k, v in vae_metrics(model, validation, device).items()})
        print(f"vae epoch={epoch+1} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch+1, "latent": settings["latent_size"], "metrics": metrics}, ckpt_path)
    save_reconstructions(model, validation, root / "vae_reconstructions.png", device)


@torch.no_grad()
def encode_rollouts(config, device):
    root, record, series = paths(config); series.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(root / "vae.pt", map_location=device)
    model = VAE(checkpoint["latent"]).to(device); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    vae_time = (root / "vae.pt").stat().st_mtime
    for index, path in enumerate(sorted(record.glob("episode_*.npz"))):
        target = series / path.name
        if target.exists() and target.stat().st_mtime > vae_time: continue  # encoded by this VAE
        data = np.load(path); obs = data["observations"]; mus, logvars = [], []
        for start in range(0, len(obs), 256):
            mu, logvar = model.encode(frames_to_tensor(obs[start:start+256], device))
            mus.append(mu.cpu().numpy()); logvars.append(logvar.cpu().numpy())
        partial = series / ("tmp_" + path.name)  # not matched by episode_*.npz
        np.savez_compressed(partial,
                            mu=np.concatenate(mus).astype(np.float16),
                            logvar=np.concatenate(logvars).astype(np.float16),
                            actions=data["actions"].astype(np.float16),
                            rewards=data["rewards"].astype(np.float16), seed=data["seed"])
        partial.replace(target); print(f"encode episode={index+1} frames={len(obs)}")


def train_mdn(config, device, resume=False):
    root, _, series = paths(config); settings = config["mdn"]; splits = load_splits(config)
    dataset = LatentSequenceDataset(split_files(series, splits["train"]), settings["sequence_length"])
    validation = LatentSequenceDataset(split_files(series, splits["validation"]),
                                       settings["sequence_length"], deterministic=True)
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True, num_workers=0)
    model = MDNRNN(config["vae"]["latent_size"], action_size=3,
                   hidden_size=settings["hidden_size"], mixtures=settings["mixtures"],
                   discrete_actions=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), settings["learning_rate"])
    ckpt_path = root / "mdn.pt"; start = 0
    if resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"]); optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["epoch"]
    for epoch in range(start, settings["epochs"]):
        model.train(); total = 0.
        for z0, actions, z1 in loader:
            loss = model.nll(z0.to(device), actions.to(device), z1.to(device))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item() * len(z0)
        nll = total / len(dataset); val_nll = mdn_nll(model, validation, device)
        print(f"mdn epoch={epoch+1} nll={nll:.5f} val_nll={val_nll:.5f}")
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch+1, "latent": config["vae"]["latent_size"],
                    "hidden": settings["hidden_size"], "mixtures": settings["mixtures"],
                    "nll": nll, "val_nll": val_nll}, ckpt_path)


@torch.no_grad()
def heldout(config, device):
    """Final test-split metrics for V and M; run once, after model selection on validation."""
    root, record, series = paths(config); splits = load_splits(config); length = config["mdn"]["sequence_length"]
    vae, mdn, _ = load_agent(config, device, True)
    test_frames = FrameDataset(split_files(record, splits["test"]))
    test_seqs = LatentSequenceDataset(split_files(series, splits["test"]), length, deterministic=True)
    # Baseline: z[t+1] ~ N(z[t], sigma^2) with per-dimension sigma fitted on training windows.
    train_seqs = LatentSequenceDataset(split_files(series, splits["train"][:200]), length, deterministic=True)
    diffs = torch.cat([(z1 - z0).reshape(-1, z0.shape[-1]) for z0, _, z1 in DataLoader(train_seqs, 32)])
    var = diffs.var(0)
    baseline = torch.cat([(0.5 * ((z1 - z0).square() / var + var.log() + math.log(2 * math.pi))).mean().view(1)
                          for z0, _, z1 in DataLoader(test_seqs, 1)]).mean().item()
    result = {"episodes": {k: len(v) for k, v in splits.items()},
              "vae_test": vae_metrics(vae, test_frames, device),
              "mdn_test_nll": mdn_nll(mdn, test_seqs, device),
              "baseline_random_walk_test_nll": baseline,
              "mdn_val_nll_at_checkpoint": torch.load(root / "mdn.pt", map_location="cpu").get("val_nll")}
    (root / "heldout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def load_agent(config, device, use_memory):
    root, _, _ = paths(config)
    vc = torch.load(root / "vae.pt", map_location=device)
    mc = torch.load(root / "mdn.pt", map_location=device)
    vae = VAE(vc["latent"]).to(device); vae.load_state_dict(vc["state_dict"]); vae.eval()
    mdn = MDNRNN(mc["latent"], 3, mc["hidden"], mc["mixtures"], False).to(device)
    mdn.load_state_dict(mc["state_dict"]); mdn.eval()
    controller = LinearController(vc["latent"], mc["hidden"], 3, use_memory).to(device)
    return vae, mdn, controller


def action_from_logits(logits):
    raw = torch.tanh(logits)
    action = raw.clone(); action[:, 1] = (raw[:, 1] + 1.) / 2.; action[:, 2] = raw[:, 2].clamp(0, 1)
    return action


@torch.no_grad()
def rollout(params, config, vae, mdn, controller, seed):
    torch.manual_seed(seed)  # z sampling depends only on the seed, so serial == parallel
    controller.set_flat_parameters(params)
    env_cfg = config["environment"]
    env = CarRacingAdapter(seed, invert=env_cfg["invert_colors"], max_steps=env_cfg["max_steps"])
    obs, _ = env.reset(seed); hidden = None; total = 0.; done = False
    while not done:
        mu, logvar = vae.encode(frames_to_tensor(obs[None], next(vae.parameters()).device))
        z = vae.reparameterize(mu, logvar)
        h = hidden[0][-1] if hidden else torch.zeros(1, mdn.hidden_size, device=z.device)
        action_tensor = action_from_logits(controller(z, h)); action = action_tensor[0].cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(action); total += reward
        hidden, _ = mdn.step_hidden(z, action, hidden); done = terminated or truncated
    env.close(); return float(total)


_WORKER = None


def _init_worker(config, use_memory):
    global _WORKER
    torch.set_num_threads(1); _WORKER = (config, *load_agent(config, "cpu", use_memory))


def _worker_rollout(job):
    config, vae, mdn, controller = _WORKER
    return rollout(job[0], config, vae, mdn, controller, job[1])


class RolloutPool:
    """Runs (params, seed) rollouts serially, or across CPU worker processes."""
    def __init__(self, config, device, use_memory, workers=1):
        self.config = config; self.pool = None
        if workers > 1:
            import multiprocessing as mp
            self.pool = mp.get_context("spawn").Pool(workers, _init_worker, (config, use_memory))
        else:
            self.agent = load_agent(config, device, use_memory)

    def map(self, jobs):
        if self.pool: return self.pool.map(_worker_rollout, jobs, chunksize=1)
        return [rollout(params, self.config, *self.agent, seed) for params, seed in jobs]

    def __enter__(self): return self

    def __exit__(self, *exc):
        if self.pool: self.pool.close(); self.pool.join()


def train_controller(config, device, use_memory=True, resume=False, workers=1):
    """CMA-ES following the reference train.py (optimizer 'cma', 64 x 16 episodes).

    - Antithetic seeding: both halves of the population share episode seeds.
    - Fitness is mean reward minus ``weight_decay * mean(params**2)``.
    - Every ``eval_every`` generations the CMA mean is scored on ``eval_rollouts``
      fixed validation tracks; the best such mean is saved as the controller.
    """
    import cma
    import time
    root, _, _ = paths(config); settings = config["controller"]; seed = config["experiment"]["seed"]
    population, episodes = settings["population"], settings["episodes_per_candidate"]
    antithetic = settings.get("antithetic", True); decay = settings.get("weight_decay", .01)
    eval_every = settings.get("eval_every", 25)
    eval_rollouts = settings.get("eval_rollouts", population * episodes)
    if antithetic and population % 2: raise ValueError("antithetic seeding needs an even population")
    controller = LinearController(config["vae"]["latent_size"], config["mdn"]["hidden_size"], 3, use_memory)
    suffix = "vm" if use_memory else "v"
    state_path = root / f"controller_{suffix}_cma.pkl"; result_path = root / f"controller_{suffix}.npy"
    log_path = root / f"controller_{suffix}_log.json"
    if resume and state_path.exists():
        with open(state_path, "rb") as stream: state = pickle.load(stream)
    else:
        es = cma.CMAEvolutionStrategy(np.zeros(controller.parameter_count), settings.get("sigma_init", .1),
                                      {"popsize": population, "seed": seed + 1})  # cma treats seed 0 as random
        state = {"es": es, "generation": 0, "seeder": np.random.RandomState(seed),
                 "best": None, "history": [], "evaluations": [], "seconds": 0.}
    es = state["es"]
    with RolloutPool(config, device, use_memory, workers) as pool:
        for generation in range(state["generation"], settings["generations"]):
            started = time.time(); candidates = np.asarray(es.ask())
            # Training tracks come from seeds >= 1e7, disjoint from validation and test tracks.
            base = state["seeder"].randint(10_000_000, 2**31 - 1 - episodes,
                                           size=population // 2 if antithetic else population)
            if antithetic: base = np.concatenate([base, base])
            jobs = [(params, int(b) + k) for params, b in zip(candidates, base) for k in range(episodes)]
            rewards = np.asarray(pool.map(jobs)).reshape(population, episodes).mean(1)
            penalty = decay * np.mean(np.square(candidates), axis=1)
            es.tell(list(candidates), (-(rewards - penalty)).tolist())
            g = generation + 1
            if g % eval_every == 0 or g == settings["generations"]:
                mean = np.asarray(es.mean).round(4)
                validation = float(np.mean(pool.map([(mean, 2_000_000 + i) for i in range(eval_rollouts)])))
                state["evaluations"].append([g, validation])
                if state["best"] is None or validation > state["best"][1]:
                    state["best"] = (mean, validation); np.save(result_path, mean)
                print(f"controller mode={suffix} generation={g} validation_reward={validation:.3f} "
                      f"best_validation={state['best'][1]:.3f} over {eval_rollouts} tracks")
            state["seconds"] += time.time() - started
            state["history"].append([g, round(state["seconds"]), float(rewards.mean()), float(rewards.min()),
                                     float(rewards.max()), float(rewards.std()), float(es.sigma)])
            state["generation"] = g
            with open(state_path, "wb") as stream: pickle.dump(state, stream)
            log_path.write_text(json.dumps({"columns": ["generation", "seconds", "mean", "min", "max", "std", "sigma"],
                                            "history": state["history"], "evaluations": state["evaluations"]}) + "\n")
            print(f"controller mode={suffix} generation={g} mean_reward={rewards.mean():.3f} "
                  f"max_reward={rewards.max():.3f} sigma={es.sigma:.4f}")


def evaluate(config, device, use_memory=True, workers=1):
    root, _, _ = paths(config); suffix = "vm" if use_memory else "v"
    params = np.load(root / f"controller_{suffix}.npy")
    count = config["controller"]["evaluation_episodes"]
    with RolloutPool(config, device, use_memory, workers) as pool:
        scores = np.asarray(pool.map([(params, 1_000_000+i) for i in range(count)]))
    result = {"mode": suffix, "episodes": count, "mean": float(scores.mean()),
              "std": float(scores.std()), "scores": scores.tolist()}
    (root / f"evaluation_{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "vae", "encode", "mdn", "heldout", "controller", "evaluate", "all"])
    parser.add_argument("--config", default="configs/carracing-smoke.toml")
    parser.add_argument("--device", default="auto"); parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU processes for collection and controller rollouts (default: controller.workers or 1)")
    args = parser.parse_args(); config = load_config(args.config); device = device_name(args.device)
    root, record, _ = paths(config); root.mkdir(parents=True, exist_ok=True)
    workers = args.workers or config["controller"].get("workers", 1)
    commands = ["collect", "vae", "encode", "mdn", "heldout", "controller", "evaluate"] if args.command == "all" else [args.command]
    env = config["environment"]
    for command in commands:
        if command == "collect": collect_rollouts(record, env["episodes"], config["experiment"]["seed"], env["max_steps"], env["invert_colors"], workers,
                                                         env.get("collection_policy", "correlated"), env.get("collection_full_episode", False))
        elif command == "vae": train_vae(config, device, args.resume)
        elif command == "encode": encode_rollouts(config, device)
        elif command == "mdn": train_mdn(config, device, args.resume)
        elif command == "heldout": heldout(config, device)
        elif command == "controller": train_controller(config, device, not args.no_memory, args.resume, workers)
        elif command == "evaluate": evaluate(config, device, not args.no_memory, workers)


if __name__ == "__main__": main()
