"""Measure rollout throughput and VAE training speed on this machine.

Uses randomly initialised V/M (speed does not depend on the weights), so it can run
before any training. Usage:  python scripts/benchmark.py --workers 1 6 10
"""
import argparse
import tempfile
import time

import numpy as np
import torch

from world_models.carracing_pipeline import RolloutPool
from world_models.models import VAE, MDNRNN, LinearController


def fake_config(workdir):
    vae, mdn = VAE(32), MDNRNN(32, 3, 256, 5, False)
    torch.save({"state_dict": vae.state_dict(), "latent": 32}, f"{workdir}/vae.pt")
    torch.save({"state_dict": mdn.state_dict(), "latent": 32, "hidden": 256, "mixtures": 5},
               f"{workdir}/mdn.pt")
    return {"experiment": {"workdir": workdir, "seed": 0},
            "environment": {"max_steps": 1000, "invert_colors": True},
            "vae": {"latent_size": 32}, "mdn": {"hidden_size": 256}, "controller": {}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 6, 10])
    parser.add_argument("--rollouts", type=int, default=None, help="per setting (default 2 x workers)")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as workdir:
        config = fake_config(workdir)
        n = LinearController(32, 256, 3, True).parameter_count
        for workers in args.workers:
            count = args.rollouts or 2 * workers
            jobs = [(np.random.default_rng(i).normal(0, .1, n), 10_000 + i) for i in range(count)]
            with RolloutPool(config, "cpu", True, workers) as pool:
                pool.map(jobs[:workers])  # warm up worker processes
                start = time.time(); pool.map(jobs); elapsed = time.time() - start
            print(f"rollouts workers={workers}: {count} x 1000 steps in {elapsed:.1f}s "
                  f"-> {count / elapsed * 3600:.0f} rollouts/hour")
    if torch.cuda.is_available():
        model = VAE(32).cuda(); optimizer = torch.optim.Adam(model.parameters(), 1e-4)
        x = torch.rand(100, 3, 64, 64, device="cuda")
        for step in range(60):
            if step == 10: torch.cuda.synchronize(); start = time.time()
            recon, mu, logvar = model(x); loss, _ = model.loss(recon, x, mu, logvar)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        torch.cuda.synchronize()
        print(f"vae cuda: {(time.time() - start) / 50 * 1000:.1f} ms per batch of 100")


if __name__ == "__main__": main()
