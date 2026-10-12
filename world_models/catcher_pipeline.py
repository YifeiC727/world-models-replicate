"""World Models on PixelCatcher with a hidden per-episode wind and fog.

Stages: collect -> vae -> encode -> mdn -> heldout, then controllers trained either in
the real environment or entirely inside M ("dream", as in the paper's Doom experiment),
always tested in the real environment. The controller follows the reference Doom code:
one tanh output without bias, mapped to left/stay/right at -1/3 and +1/3.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .carracing_pipeline import load_config, load_splits, paths, split_files, train_vae
from .data import frames_to_tensor
from .env import PixelCatcher, PixelCatcherConfig
from .models import VAE, MDNRNN, LinearController
from .pipeline import device_name

EVENTS = 3  # class 0 = miss, 1 = nothing, 2 = catch (env event + 1)
STEP_PENALTY = -0.005  # PixelCatcher's per-step reward when nothing happens


def env_config(config):
    e = config["environment"]
    return PixelCatcherConfig(horizon=e["horizon"], wind_range=e["wind_range"], fog_y=e.get("fog_y"))


# ---------------------------------------------------------------- collection

def _collect_episode(job):
    root, episode, seed, cfg, persistence = job
    path = root / f"episode_{episode:05d}.npz"
    if path.exists(): return episode
    env = PixelCatcher(cfg, seed); obs, _ = env.reset(seed); rng = np.random.default_rng(seed + 1)
    observations, actions, events = [obs], [], []; done = False; action = 1; left = 0
    while not done:
        if left <= 0:  # sticky random actions, so the paddle visits the whole width
            action = int(rng.integers(3)); left = int(rng.integers(1, persistence + 1))
        left -= 1
        obs, _, done, _, info = env.step(action)
        observations.append(obs); actions.append(action); events.append(info["event"])
    partial = path.with_name("tmp_" + path.name)
    np.savez_compressed(partial, observations=np.asarray(observations, np.uint8),
                        actions=np.asarray(actions, np.int64), events=np.asarray(events, np.int64),
                        wind=env.wind, seed=seed)
    partial.replace(path); return episode


def collect(config, workers):
    _, record, _ = paths(config); record.mkdir(parents=True, exist_ok=True)
    e = config["environment"]; cfg = env_config(config)
    jobs = [(record, i, config["experiment"]["seed"] + i, cfg, e.get("random_persistence", 8))
            for i in range(e["episodes"])]
    with _pool(workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(_collect_episode, jobs) if pool else map(_collect_episode, jobs), 1):
            if done % 100 == 0 or done == len(jobs): print(f"collect {done}/{len(jobs)}")


class _pool:
    def __init__(self, workers, initializer=None, initargs=()):
        self.workers, self.initializer, self.initargs = workers, initializer, initargs

    def __enter__(self):
        if self.workers <= 1:
            if self.initializer: self.initializer(*self.initargs)
            self.pool = None; return None
        import multiprocessing as mp
        self.pool = mp.get_context("spawn").Pool(self.workers, self.initializer, self.initargs)
        return self.pool

    def __exit__(self, *exc):
        if self.pool: self.pool.close(); self.pool.join()


# ---------------------------------------------------------------- V and M

@torch.no_grad()
def encode(config, device):
    root, record, series = paths(config); series.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(root / "vae.pt", map_location=device)
    vae = VAE(ckpt["latent"]).to(device); vae.load_state_dict(ckpt["state_dict"]); vae.eval()
    vae_time = (root / "vae.pt").stat().st_mtime
    for path in sorted(record.glob("episode_*.npz")):
        target = series / path.name
        if target.exists() and target.stat().st_mtime > vae_time: continue
        data = np.load(path); mu, logvar = vae.encode(frames_to_tensor(data["observations"], device))
        partial = series / ("tmp_" + path.name)
        np.savez_compressed(partial, mu=mu.cpu().numpy(), logvar=logvar.cpu().numpy(), actions=data["actions"],
                            events=data["events"], wind=data["wind"])
        partial.replace(target)
    print(f"encode episodes={len(list(series.glob('episode_*.npz')))}")


def load_series(files):
    data = [np.load(f) for f in files]
    stack = lambda key: torch.from_numpy(np.stack([d[key] for d in data]))
    return {"mu": stack("mu").float(), "logvar": stack("logvar").float(), "actions": stack("actions").long(),
            "events": stack("events").long() + 1, "wind": torch.tensor([float(d["wind"]) for d in data])}


def _sample_z(series, idx, generator=None):
    mu, logvar = series["mu"][idx], series["logvar"][idx]
    return mu + torch.randn(mu.shape, generator=generator) * torch.exp(.5 * logvar)


def event_weights(settings):
    """[miss, none, catch] loss weights; a scalar weights both rare classes equally."""
    w = settings.get("event_weights") or [settings["event_weight"], 1., settings["event_weight"]]
    return torch.tensor(w, dtype=torch.float32)


def mdn_losses(model, series, idx, device, weights, generator=None):
    z = _sample_z(series, idx, generator).to(device)
    actions, events = series["actions"][idx].to(device), series["events"][idx].to(device)
    logits, mu, log_sigma, _, event_logits = model(z[:, :-1], actions, return_events=True)
    nll = MDNRNN.mixture_nll(logits, mu, log_sigma, z[:, 1:])
    ce = F.cross_entropy(event_logits.reshape(-1, EVENTS), events.reshape(-1), weight=weights.to(device))
    return nll, ce, event_logits, events


@torch.no_grad()
def mdn_metrics(model, series, device, weights):
    """Deterministic z sample. Besides argmax recall, report the mean predicted catch
    probability on steps that were really catches vs misses: the dream samples events,
    so this gap is what turns into a reward signal for catching."""
    model.eval(); idx = torch.arange(len(series["mu"]))
    nll, ce, logits, true = mdn_losses(model, series, idx, device, weights, torch.Generator().manual_seed(0))
    pred, prob = logits.argmax(-1), torch.softmax(logits, -1)
    recall = {name: float((pred[true == c] == c).float().mean()) for c, name in enumerate(["miss", "none", "catch"])}
    return {"nll": float(nll), "event_ce": float(ce), **{f"recall_{k}": v for k, v in recall.items()},
            "p_catch_on_catch": float(prob[..., 2][true == 2].mean()), "p_catch_on_miss": float(prob[..., 2][true == 0].mean())}


def train_mdn(config, device):
    root, _, series_dir = paths(config); s = config["mdn"]; splits = load_splits(config)
    train = load_series(split_files(series_dir, splits["train"]))
    validation = load_series(split_files(series_dir, splits["validation"]))
    torch.manual_seed(config["experiment"]["seed"])
    model = MDNRNN(config["vae"]["latent_size"], 3, s["hidden_size"], s["mixtures"], True, EVENTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), s["learning_rate"])
    for epoch in range(s["epochs"]):
        model.train(); order = torch.randperm(len(train["mu"])); totals = np.zeros(2)
        for start in range(0, len(order), s["batch_size"]):
            idx = order[start:start + s["batch_size"]]
            nll, ce, _, _ = mdn_losses(model, train, idx, device, event_weights(s))
            optimizer.zero_grad(); (nll + ce).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            totals += np.array([nll.item(), ce.item()]) * len(idx)
        totals /= len(order); val = mdn_metrics(model, validation, device, event_weights(s))
        print(f"mdn epoch={epoch+1} nll={totals[0]:.4f} event_ce={totals[1]:.4f} "
              + " ".join(f"val_{k}={v:.4f}" for k, v in val.items()))
    torch.save({"state_dict": model.state_dict(), "latent": config["vae"]["latent_size"],
                "hidden": s["hidden_size"], "mixtures": s["mixtures"], "validation": val}, root / "mdn.pt")


def heldout(config, device):
    root, record, series_dir = paths(config); splits = load_splits(config)
    from .carracing_data import FrameDataset
    from .carracing_pipeline import vae_metrics
    vae, mdn = load_models(config, device)
    test = load_series(split_files(series_dir, splits["test"]))
    train = load_series(split_files(series_dir, splits["train"][:200]))
    var = (_sample_z(train, slice(None), torch.Generator().manual_seed(0)).diff(dim=1)).reshape(-1, train["mu"].shape[-1]).var(0)
    zt = _sample_z(test, slice(None), torch.Generator().manual_seed(0))
    baseline = float((.5 * (zt.diff(dim=1).square() / var + var.log() + math.log(2 * math.pi))).mean())
    result = {"episodes": {k: len(v) for k, v in splits.items()},
              "vae_test": vae_metrics(vae, FrameDataset(split_files(record, splits["test"])), device),
              "mdn_test": mdn_metrics(mdn, test, device, event_weights(config["mdn"])),
              "baseline_random_walk_test_nll": baseline}
    (root / "heldout.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


def load_models(config, device):
    root, _, _ = paths(config)
    vc = torch.load(root / "vae.pt", map_location=device); mc = torch.load(root / "mdn.pt", map_location=device)
    vae = VAE(vc["latent"]).to(device); vae.load_state_dict(vc["state_dict"]); vae.eval()
    mdn = MDNRNN(mc["latent"], 3, mc["hidden"], mc["mixtures"], True, EVENTS).to(device)
    mdn.load_state_dict(mc["state_dict"]); mdn.eval()
    return vae, mdn


# ---------------------------------------------------------------- controllers

def make_controller(config, use_memory):
    return LinearController(config["vae"]["latent_size"], config["mdn"]["hidden_size"], 1, use_memory, bias=False)


def to_action(output):
    """Reference Doom mapping: tanh output below -1/3 -> left, above 1/3 -> right."""
    out = torch.tanh(output).squeeze(-1)
    return 1 + (out > 1 / 3).long() - (out < -1 / 3).long()


_AGENT = None


def _init_real(config, use_memory):
    global _AGENT
    torch.set_num_threads(1); _AGENT = (config, *load_models(config, "cpu"), make_controller(config, use_memory))


@torch.no_grad()
def real_rollout(job):
    """Returns (total reward, catches, balls) for one real episode."""
    params, seed = job; config, vae, mdn, controller = _AGENT
    torch.manual_seed(seed); controller.set_flat_parameters(params)
    env = PixelCatcher(env_config(config), seed); obs, _ = env.reset(seed)
    hidden = None; total = 0.; catches = balls = 0; done = False
    while not done:
        mu, logvar = vae.encode(frames_to_tensor(obs[None]))
        z = vae.reparameterize(mu, logvar)
        h = hidden[0][-1] if hidden else torch.zeros(1, mdn.hidden_size)
        action = int(to_action(controller(z, h)))
        obs, reward, done, _, info = env.step(action); total += reward
        catches += info["event"] == 1; balls += info["event"] != 0
        hidden, _ = mdn.step_hidden(z, action, hidden)
    return total, catches, balls


class Dream:
    """Batched rollouts inside M with common random numbers across candidates."""

    def __init__(self, config, use_memory, temperature):
        _, _, series_dir = paths(config); splits = load_splits(config)
        _, self.mdn = load_models(config, "cpu"); self.use_memory = use_memory
        self.temperature = temperature; self.steps = config["environment"]["horizon"]
        initial = load_series(split_files(series_dir, splits["train"]))
        self.initial_mu, self.initial_logvar = initial["mu"][:, 0], initial["logvar"][:, 0]

    @torch.no_grad()
    def __call__(self, params, episodes, seed):
        """params: (candidates, n). Every candidate meets the same ``episodes`` dreams."""
        torch.set_num_threads(4)
        candidates = len(params); g = torch.Generator().manual_seed(int(seed)); m = self.mdn
        weights = torch.as_tensor(np.asarray(params), dtype=torch.float32).repeat_interleave(episodes, 0)
        pick = torch.randint(len(self.initial_mu), (episodes,), generator=g)
        z = (self.initial_mu[pick] + torch.randn(episodes, m.latent_size, generator=g)
             * torch.exp(.5 * self.initial_logvar[pick])).repeat(candidates, 1)
        hidden = None; total = torch.zeros(len(z)); catches = torch.zeros(len(z)); balls = torch.zeros(len(z))
        tau = self.temperature
        for _ in range(self.steps):
            h = hidden[0][-1] if hidden else torch.zeros(len(z), m.hidden_size)
            x = torch.cat((z, h), -1) if self.use_memory else z
            action = to_action((weights * x).sum(-1, keepdim=True))
            logits, mu, log_sigma, hidden, event_logits = m(z[:, None], action[:, None], hidden, return_events=True)
            # Shared noise: one draw per (episode, step), broadcast over candidates.
            u = torch.rand(episodes, m.latent_size, 1, generator=g).repeat(candidates, 1, 1)
            eps = torch.randn(episodes, m.latent_size, generator=g).repeat(candidates, 1)
            ue = torch.rand(episodes, 1, generator=g).repeat(candidates, 1)
            cdf = torch.softmax(logits[:, 0] / tau, -1).cumsum(-1)
            k = (u > cdf).sum(-1, keepdim=True).clamp(max=m.mixtures - 1)
            z = mu[:, 0].gather(-1, k).squeeze(-1) + log_sigma[:, 0].gather(-1, k).squeeze(-1).exp() * eps * math.sqrt(tau)
            event = (ue > torch.softmax(event_logits[:, 0], -1).cumsum(-1)).sum(-1).clamp(max=EVENTS - 1)
            total += STEP_PENALTY + (event == 2).float() * (1 - STEP_PENALTY) + (event == 0).float() * (-1 - STEP_PENALTY)
            catches += (event == 2).float(); balls += (event != 1).float()
        return total.view(candidates, episodes).mean(1).numpy()


def train_controller(config, mode, use_memory, temperature, workers):
    """CMA-ES with the reference settings; ``mode`` is 'real' or 'dream'."""
    import cma
    root, _, _ = paths(config); s = config["controller"]; seed = config["experiment"]["seed"]
    name = run_name(mode, use_memory, temperature); n = make_controller(config, use_memory).parameter_count
    population, episodes = s["population"], s["episodes_per_candidate"]
    decay, eval_every = s.get("weight_decay", .01), s.get("eval_every", 25)
    state_path, log_path = root / f"controller_{name}_cma.pkl", root / f"controller_{name}_log.json"
    if state_path.exists():
        with open(state_path, "rb") as stream: state = pickle.load(stream)
    else:
        state = {"es": cma.CMAEvolutionStrategy(np.zeros(n), s.get("sigma_init", .1), {"popsize": population, "seed": seed + 1}),
                 "generation": 0, "seeder": np.random.RandomState(seed), "best": None, "history": [], "evaluations": []}
    es = state["es"]; dream = Dream(config, use_memory, temperature) if mode == "dream" else None
    with _pool(workers if mode == "real" else 1, _init_real, (config, use_memory)) as pool:
        run = (lambda jobs: pool.map(real_rollout, jobs, chunksize=1)) if pool else (lambda jobs: list(map(real_rollout, jobs)))
        for generation in range(state["generation"], s["generations"]):
            started = time.time(); candidates = np.asarray(es.ask())
            if mode == "real":
                base = state["seeder"].randint(10_000_000, 2**31 - 1 - episodes, size=population // 2)
                base = np.concatenate([base, base])  # antithetic pairs share tracks
                jobs = [(p, int(b) + k) for p, b in zip(candidates, base) for k in range(episodes)]
                rewards = np.asarray([r[0] for r in run(jobs)]).reshape(population, episodes).mean(1)
            else:
                rewards = dream(candidates, episodes, state["seeder"].randint(2**31 - 1))
            es.tell(list(candidates), (-(rewards - decay * np.mean(candidates**2, 1))).tolist())
            g = generation + 1
            if g % eval_every == 0 or g == s["generations"]:
                mean = np.asarray(es.mean)
                # Selection never touches the real environment in dream mode.
                val = (float(np.mean([r[0] for r in run([(mean, 2_000_000 + i) for i in range(s["eval_rollouts"])])]))
                       if mode == "real" else float(dream(mean[None], s["eval_rollouts"], 2_000_000)[0]))
                state["evaluations"].append([g, val])
                if state["best"] is None or val > state["best"][1]:
                    state["best"] = (mean, val); np.save(root / f"controller_{name}.npy", mean)
                print(f"controller {name} generation={g} validation_reward={val:.3f} best={state['best'][1]:.3f}")
            state["history"].append([g, round(time.time() - started, 1), float(rewards.mean()), float(rewards.max()), float(es.sigma)])
            state["generation"] = g
            with open(state_path, "wb") as stream: pickle.dump(state, stream)
            log_path.write_text(json.dumps({"columns": ["generation", "seconds", "mean", "max", "sigma"],
                                            "history": state["history"], "evaluations": state["evaluations"]}) + "\n")
            print(f"controller {name} generation={g} mean_reward={rewards.mean():.3f} max_reward={rewards.max():.3f} "
                  f"sigma={es.sigma:.4f} seconds={state['history'][-1][1]}")


def run_name(mode, use_memory, temperature):
    return f"{mode}_{'vm' if use_memory else 'v'}" + (f"_t{temperature:g}" if mode == "dream" else "")


def evaluate(config, mode, use_memory, temperature, workers):
    """Real-environment test on fixed seeds, whatever the controller was trained in."""
    root, _, _ = paths(config); name = run_name(mode, use_memory, temperature)
    params = np.load(root / f"controller_{name}.npy"); count = config["controller"]["evaluation_episodes"]
    with _pool(workers, _init_real, (config, use_memory)) as pool:
        jobs = [(params, 1_000_000 + i) for i in range(count)]
        results = np.asarray(pool.map(real_rollout, jobs) if pool else list(map(real_rollout, jobs)))
    result = {"controller": name, "episodes": count, "reward_mean": float(results[:, 0].mean()),
              "reward_std": float(results[:, 0].std()), "catch_rate": float(results[:, 1].sum() / results[:, 2].sum())}
    (root / f"evaluation_{name}.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


def scripted_baselines(config):
    """Privileged scripted players on the same test seeds: reference lines for the table."""
    root, _, _ = paths(config); cfg = env_config(config); count = config["controller"]["evaluation_episodes"]

    def landing(e, x, vx, y, wind):
        while y < e.cfg.height - 7:
            vx = float(np.clip(vx + wind, -e.cfg.max_ball_vx, e.cfg.max_ball_vx)); x += vx; y += e.cfg.ball_speed
            if x < 2 or x > e.cfg.width - 3: vx *= -1; x = float(np.clip(x, 2, e.cfg.width - 3))
        return x

    def move(e, target): return 1 if abs(target - e.paddle_x) < 1.5 else (2 if target > e.paddle_x else 0)

    results = {}
    for kind in ["random", "memoryless", "velocity", "velocity_and_wind"]:
        totals, catches, balls = [], 0, 0
        for i in range(count):
            seed = 1_000_000 + i; e = PixelCatcher(cfg, seed); e.reset(seed); rng = np.random.default_rng(seed)
            prev = last = None; done = False; total = 0.
            while not done:
                fog = cfg.fog_y if cfg.fog_y is not None else e.cfg.height
                if e.ball_y == 5.0: prev = None
                if kind == "random": action = int(rng.integers(3))
                else:
                    visible = e.ball_y < fog
                    if visible:
                        v = None if prev is None else e.ball_x - prev; prev = e.ball_x; last = (e.ball_x, v, e.ball_y)
                    if kind == "memoryless": action = move(e, e.ball_x) if visible else 1
                    elif last[1] is None: action = move(e, e.ball_x) if visible else 1
                    else: action = move(e, landing(e, last[0], last[1], last[2], e.wind if kind == "velocity_and_wind" else 0.))
                _, r, done, _, info = e.step(action); total += r; catches += info["event"] == 1; balls += info["event"] != 0
            totals.append(total)
        results[kind] = {"reward_mean": float(np.mean(totals)), "reward_std": float(np.std(totals)), "catch_rate": catches / balls}
    (root / "scripted_baselines.json").write_text(json.dumps(results, indent=2) + "\n"); print(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["collect", "vae", "encode", "mdn", "heldout", "baselines", "controller", "evaluate"])
    p.add_argument("--config", default="configs/catcher.toml"); p.add_argument("--device", default="auto")
    p.add_argument("--mode", choices=["real", "dream"], default="real"); p.add_argument("--no-memory", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--workers", type=int, default=None)
    args = p.parse_args(); config = load_config(args.config); device = device_name(args.device)
    workers = args.workers or config["controller"].get("workers", 1)
    root, _, _ = paths(config); root.mkdir(parents=True, exist_ok=True)
    if args.command == "collect": collect(config, workers)
    elif args.command == "vae": train_vae(config, device, resume=True)
    elif args.command == "encode": encode(config, device)
    elif args.command == "mdn": train_mdn(config, device)
    elif args.command == "heldout": heldout(config, device)
    elif args.command == "baselines": scripted_baselines(config)
    elif args.command == "controller": train_controller(config, args.mode, not args.no_memory, args.temperature, workers)
    else: evaluate(config, args.mode, not args.no_memory, args.temperature, workers)


if __name__ == "__main__": main()
