# World Models reproduction

A modern PyTorch reproduction of **Ha & Schmidhuber (2018),
[World Models](https://arxiv.org/abs/1803.10122)**. The repository now contains
the paper's CarRacing pipeline and retains PixelCatcher as a fast custom-environment
prototype.

## CarRacing local smoke test

Install the optional Box2D environment and run the deliberately tiny profile:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,carracing]'
python -m world_models.carracing_pipeline all \
  --config configs/carracing-smoke.toml --device auto
```

This performs collection, one VAE epoch, latent encoding, one MDN-RNN epoch,
one CMA-ES generation, and a three-episode evaluation. It checks plumbing only.
The paper-scale settings are recorded in `configs/carracing-paper.toml`; do not
launch that shared-cluster workload without reviewing storage and worker limits.

Run the V-only ablation after V and M exist:

```bash
python -m world_models.carracing_pipeline controller --no-memory
python -m world_models.carracing_pipeline evaluate --no-memory
```

Checkpoints are written after every VAE/MDN epoch and CMA-ES generation. Add
`--resume` to continue an interrupted stage.

### Local pilot (50 episodes) and device choice

`configs/carracing-pilot.toml` uses 50 full random-policy episodes. On an Apple M3:

- Collection: ~5–7 s per episode in one process (~1 MB compressed per episode).
- VAE / MDN-RNN training: use `--device mps` (about 6× faster than CPU per batch).
- Controller and evaluation rollouts: use `--device cpu`; single-frame inference is
  dominated by MPS dispatch overhead (~4.6 s vs ~8.3 s per 1000-step rollout).

The VAE samples frames with `EpisodeChunkSampler`: episodes are shuffled into chunks
of `vae.shuffle_chunk_episodes` and frames are shuffled within each chunk. A global
shuffle would decompress a whole episode for almost every frame, which makes an
epoch I/O-bound regardless of device.

### Splits and held-out metrics

On first use, episodes are split 80/10/10 into train/validation/test by whole
episode (`[data]` in the config) and frozen in `<workdir>/splits.json`, so
correlated neighbouring frames never cross splits. V and M train on `train` and
report `val_*` each epoch. After model choices are final, run once:

```bash
python -m world_models.carracing_pipeline heldout --config configs/carracing-pilot.toml
```

This writes `heldout.json` with the VAE test loss (reconstruction from the
posterior mean), the MDN-RNN test NLL (nats per latent dimension, z sampled with a
fixed per-window seed), and a random-walk baseline `z[t+1] ~ N(z[t], sigma^2)` with
sigma fitted on training windows. The MDN is only learning dynamics if it beats it.

### Parallel controller rollouts

CMA-ES fitness and evaluation rollouts run on `--workers N` CPU processes (default
`controller.workers`). Each rollout seeds torch from its environment seed, so the
results are identical to a serial run.

## MLDS server (mlds-deepdish1)

One-time setup creates a private conda env in `~/envs/wm` (nothing system-wide):

```bash
bash scripts/setup_server.sh
python scripts/benchmark.py --workers 1 6 10 12   # optional: measure throughput
```

Then run inside `tmux`; every stage resumes if rerun after an interruption:

```bash
bash scripts/run_server.sh world           # collect, V, M, held-out metrics
bash scripts/run_server.sh controller vm   # then the V-only ablation: controller v
```

`configs/carracing-deepdish.toml` uses the reference settings but starts with 50
CMA-ES generations: one generation is 1,024 rollouts, about 0.6 h on the server's
6 physical cores (~1,700 rollouts/hour), so the reference's 1,800 would take ~44
days per mode. Extend by raising `generations` and rerunning (it resumes). Jobs run under `nice` on one GPU
(`GPU=1` selects another).

## Alignment with the reference implementation

Checked against `hardmaru/WorldModelsExperiments/carracing` (cloned to `references/`, not
committed). `configs/carracing-paper.toml` is the full reference; `carracing-deepdish.toml`
differs only in CMA-ES generations.

| Component | Reference | This repo |
|---|---|---|
| Frames | crop 84 rows, resize 64x64, invert | same (PIL bilinear) |
| Data | 10,000 episodes, termination ignored, 1000 frames | same |
| Random policy | random V/M/C with Cauchy x U(0, 0.01) weights | **smooth random controls** (see below) |
| V | conv VAE, z=32, 10 epochs, batch 100, lr 1e-4, KL tolerance 0.5 | same; shuffle in 200-episode chunks instead of globally |
| M | LSTM 256, 5 mixtures per dim, seq 999, batch 100, lr 1e-3, clip 1.0, 4000 steps | same (50 epochs x 80 steps); 80/10/10 episode split added |
| C | linear on [z, h], 867 params, tanh / (x+1)/2 / clip | same |
| CMA-ES | pop 64 x 16 episodes, sigma 0.1, weight decay 0.01, antithetic seeds | same |
| Model selection | CMA mean scored on 1024 tracks every 25 generations, keep best | same |
| Generations | 1800 (validation 822.9 @25, 867.5 @50, 880.4 @200, 900.5 @1800) | 1800 in paper config; 50+ on deepdish |
| Test | mean over 100 random tracks | 100 fixed seeds |
| Environment | gym 0.9.4 CarRacing | Gymnasium CarRacing-v3 |

**Random policy deviation.** Ported faithfully, the reference random policy overflows: the
Cauchy-weighted random VAE gives log-variances of 1e3-1e5, so `exp(logvar/2)` is infinite in
the reference's float32 and the actions become NaN. How gym 0.9.4 handled NaN actions cannot be
checked now. Saturating instead of propagating NaN (`collection_policy = "reference"`) overflowed
in 11 of 12 real episodes and braked about half the time, so the car barely moves. We therefore
collect with `CorrelatedRandomPolicy` (piecewise-constant random steering/gas, rare braking).

## Methodology

1. Collect pixel observations using a random policy.
2. Train a convolutional VAE (**V**) without rewards to compress each 64×64 RGB
   frame to a latent vector `z`.
3. Train an LSTM mixture-density network (**M**) without rewards to model
   `p(z[t+1] | z[t], action[t], h[t])`.
4. Optimize a single linear controller (**C**) with CMA-ES. Only C sees reward;
   its input is `[z, h]`.
5. Evaluate over held-out environment seeds and ablate memory by comparing
   `[z, h]` against `z` alone.

PixelCatcher is deliberately partially observable from one frame: the ball's
horizontal velocity is invisible without temporal context. This makes the V-only
versus V+M comparison scientifically meaningful.

## PixelCatcher prototype

Python 3.10–3.12 is recommended (PyTorch wheels may lag brand-new Python releases).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

python -m world_models.pipeline collect --episodes 200
python -m world_models.pipeline vae --epochs 10
python -m world_models.pipeline encode
python -m world_models.pipeline mdn --epochs 20
python -m world_models.pipeline controller --generations 60
python -m world_models.pipeline evaluate --episodes 100
```

Run the required ablation with the same data/model and a separate controller:

```bash
python -m world_models.pipeline controller --no-memory --generations 60
python -m world_models.pipeline evaluate --no-memory --episodes 100
```

For a quick smoke run, use 10 collection episodes, one epoch, two generations,
and a population of four. Those settings validate plumbing, not performance.

## Expected experiment table

Report mean ± standard deviation across 100 fixed held-out seeds:

| Controller input | Mean reward | Std. dev. |
|---|---:|---:|
| VAE latent `z` | fill after run | fill after run |
| VAE latent + MDN-RNN state `[z,h]` | fill after run | fill after run |

The primary hypothesis is that `[z,h]` outperforms `z`. Also retain VAE
reconstructions and held-out MDN negative log likelihood as representation/model
diagnostics; reward alone cannot establish that the learned world model is sound.

## Scope and differences from the paper

This reproduces the paper's architecture and staged methodology on a small new
environment. It does **not** claim numerical replication of the published
CarRacing score (906 ± 21), which used 10,000 rollouts, a 32-D latent, and the
historical `CarRacing-v0` stack. PixelCatcher uses a 16-D latent, 128-unit LSTM,
five Gaussian mixture components, and 200 rollouts by default so the experiment
is practical on a laptop.
