"""Command-line experiment: collect, train V, train M, train/evaluate C."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import collect_random_rollouts, frames_to_tensor
from .env import PixelCatcher
from .models import VAE, MDNRNN, LinearController


def device_name(requested):
    if requested != "auto": return requested
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def train_vae(data_path, out, latent=16, epochs=10, batch_size=128, device="cpu"):
    data = np.load(data_path)
    frames = data["observations"].reshape(-1, 64, 64, 3)
    loader = DataLoader(TensorDataset(frames_to_tensor(frames)), batch_size=batch_size,
                        shuffle=True)
    model = VAE(latent).to(device); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for epoch in range(epochs):
        total = 0.0
        for (x,) in loader:
            x = x.to(device); recon, mu, logvar = model(x)
            loss, _ = model.loss(recon, x, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(x)
        print(f"vae epoch={epoch+1} loss={total/len(frames):.4f}")
    torch.save({"state_dict": model.state_dict(), "latent": latent}, out)


@torch.no_grad()
def encode_rollouts(data_path, vae_path, out, device="cpu", batch_size=512):
    data = np.load(data_path); ckpt = torch.load(vae_path, map_location=device)
    model = VAE(ckpt["latent"]).to(device); model.load_state_dict(ckpt["state_dict"]); model.eval()
    obs = data["observations"]; flat = obs.reshape(-1, 64, 64, 3); chunks = []
    for i in range(0, len(flat), batch_size):
        mu, _ = model.encode(frames_to_tensor(flat[i:i+batch_size], device))
        chunks.append(mu.cpu().numpy())
    latents = np.concatenate(chunks).reshape(*obs.shape[:2], -1)
    np.savez_compressed(out, latents=latents, actions=data["actions"], rewards=data["rewards"])


def train_mdn(data_path, out, latent=16, hidden=128, mixtures=5, epochs=20,
              batch_size=32, device="cpu"):
    data = np.load(data_path)
    z = torch.from_numpy(data["latents"]).float()
    a = torch.from_numpy(data["actions"]).long()
    loader = DataLoader(TensorDataset(z[:, :-1], a, z[:, 1:]), batch_size=batch_size, shuffle=True)
    model = MDNRNN(latent, hidden_size=hidden, mixtures=mixtures, discrete_actions=True).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for epoch in range(epochs):
        total = 0.0
        for z0, actions, z1 in loader:
            loss = model.nll(z0.to(device), actions.to(device), z1.to(device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * len(z0)
        print(f"mdn epoch={epoch+1} nll={total/len(z):.4f}")
    torch.save({"state_dict": model.state_dict(), "latent": latent, "hidden": hidden,
                "mixtures": mixtures}, out)


def load_agent(vae_path, mdn_path, use_memory=True, device="cpu"):
    vc = torch.load(vae_path, map_location=device); mc = torch.load(mdn_path, map_location=device)
    vae = VAE(vc["latent"]).to(device); vae.load_state_dict(vc["state_dict"]); vae.eval()
    mdn = MDNRNN(mc["latent"], hidden_size=mc["hidden"], mixtures=mc["mixtures"],
                 discrete_actions=True).to(device)
    mdn.load_state_dict(mc["state_dict"]); mdn.eval()
    controller = LinearController(vc["latent"], mc["hidden"], use_memory=use_memory).to(device)
    return vae, mdn, controller


@torch.no_grad()
def rollout(params, vae, mdn, controller, seed):
    controller.set_flat_parameters(params); env = PixelCatcher(seed=seed)
    obs, _ = env.reset(seed); hidden = None; total = 0.0; done = False
    while not done:
        z, _ = vae.encode(frames_to_tensor(obs[None], next(vae.parameters()).device))
        h = (hidden[0][-1] if hidden else torch.zeros(1, mdn.hidden_size, device=z.device))
        action = int(controller(z, h).argmax(-1).item())
        obs, reward, done, _, _ = env.step(action); total += reward
        hidden, _ = mdn.step_hidden(z, action, hidden)
    return total


def train_controller(vae_path, mdn_path, out, generations=60, population=24,
                     episodes=4, use_memory=True, seed=0, device="cpu"):
    import cma
    vae, mdn, controller = load_agent(vae_path, mdn_path, use_memory, device)
    es = cma.CMAEvolutionStrategy(np.zeros(controller.parameter_count), .5,
                                  {"popsize": population, "seed": seed})
    for generation in range(generations):
        candidates = es.ask(); fitness = []
        for j, params in enumerate(candidates):
            scores = [rollout(params, vae, mdn, controller, seed + generation*10000+j*100+k)
                      for k in range(episodes)]
            fitness.append(-float(np.mean(scores)))
        es.tell(candidates, fitness)
        print(f"controller generation={generation+1} reward={-min(fitness):.3f}")
    best = np.asarray(es.result.xbest); np.save(out, best)


def evaluate(vae_path, mdn_path, controller_path, episodes=100, use_memory=True, device="cpu"):
    vae, mdn, controller = load_agent(vae_path, mdn_path, use_memory, device)
    params = np.load(controller_path)
    scores = np.asarray([rollout(params, vae, mdn, controller, 100000+i) for i in range(episodes)])
    result = {"episodes": episodes, "mean": float(scores.mean()), "std": float(scores.std()),
              "min": float(scores.min()), "max": float(scores.max()), "use_memory": use_memory}
    print(json.dumps(result, indent=2)); return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("command", choices=["collect", "vae", "encode", "mdn", "controller", "evaluate"])
    p.add_argument("--workdir", default="runs/default"); p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None); p.add_argument("--generations", type=int, default=60)
    p.add_argument("--population", type=int, default=24); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto"); p.add_argument("--no-memory", action="store_true")
    args = p.parse_args(); d = Path(args.workdir); d.mkdir(parents=True, exist_ok=True); dev = device_name(args.device)
    if args.command == "collect": collect_random_rollouts(d/"rollouts.npz", args.episodes or 200, args.seed)
    elif args.command == "vae": train_vae(d/"rollouts.npz", d/"vae.pt", epochs=args.epochs or 10, device=dev)
    elif args.command == "encode": encode_rollouts(d/"rollouts.npz", d/"vae.pt", d/"latents.npz", device=dev)
    elif args.command == "mdn": train_mdn(d/"latents.npz", d/"mdn.pt", epochs=args.epochs or 20, device=dev)
    elif args.command == "controller": train_controller(d/"vae.pt", d/"mdn.pt", d/("controller_v.npy" if args.no_memory else "controller_vm.npy"), generations=args.generations, population=args.population, episodes=args.episodes or 4, use_memory=not args.no_memory, seed=args.seed, device=dev)
    else: evaluate(d/"vae.pt", d/"mdn.pt", d/("controller_v.npy" if args.no_memory else "controller_vm.npy"), episodes=args.episodes or 100, use_memory=not args.no_memory, device=dev)


if __name__ == "__main__": main()
