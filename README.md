# World Models reproduction

A modern PyTorch reproduction of **Ha & Schmidhuber (2018),
[World Models](https://arxiv.org/abs/1803.10122)**: the paper's CarRacing experiment,
plus a new environment (PixelCatcher with a hidden per-episode wind and fog) on which
the controller is trained both in the real game and inside the learned world model.

## Results

### CarRacing (replication)

Test score over 100 unseen tracks (Gymnasium CarRacing-v3), mean ± std. Paper numbers
are from Table 1 of the paper (CarRacing-v0, 100 random trials).

| Controller input | Ours | Paper |
|---|---:|---:|
| V only, `z` | **658.6 ± 186.5** | 632 ± 251 |
| Full world model, `[z, h]` | **841.7 ± 94.6** | 906 ± 21 |

- Settings follow the authors' code (see [alignment table](#alignment-with-the-reference-implementation)),
  except that CMA-ES ran **50 generations instead of 1,800** (36.7 h per mode on 6 cores;
  the reference took ~46 days on 64).
- At equal generations our full model tracks the reference: validation score of the
  CMA mean on 1,024 tracks was 727.8 vs 822.9 at generation 25 and **828.1 vs 867.5 at
  generation 50** (95%). V only: 480.4 and 637.7.
- Memory adds **+183** points and halves the spread. Tracks below 600: 27 (V only) vs 5
  (full model); tracks at 900+: 2 vs 20. The remaining gap to 906 comes mostly from the
  5 failed tracks (median 870, best 914.7), which the reference's extra 1,750
  generations would be expected to remove.
- Held-out diagnostics (1,000 test episodes): VAE reconstruction 4.29; MDN-RNN NLL 0.967
  nats/dim vs 1.393 for a random-walk baseline (train ≈ validation ≈ test).

### PixelCatcher with hidden wind and fog (new environment)

Real-game test over the same 200 episodes (~2,000 balls) for every row; scripted rows
read the true game state and serve as reference lines.

| Policy | Trained in | Catch rate | Reward / episode |
|---|---|---:|---:|
| Random | – | 24.7% | −6.51 |
| **V only** | real game, 50 gen | **38.1%** | −3.84 |
| Scripted: no memory | – | 39.6% | −3.53 |
| V+M, τ = 1.0 | **dream only**, 50 gen (~1 min) | 58.8% | 0.30 |
| V+M, τ = 1.25 | **dream only**, 50 gen (~1 min) | 60.7% | 0.69 |
| **V+M** | real game, 50 gen (~2.4 h) | **63.1%** | 1.17 |
| Scripted: remembers velocity | – | 73.9% | 3.32 |
| Scripted: velocity + true wind | – | 99.6% | 8.46 |

- **Memory is necessary and is used**: V only stops at the memoryless ceiling (38.1% vs
  39.6%); V+M reaches 63.1%. The 25-point gap is far larger than the ~1.5-point standard
  error.
- **Dream training transfers**: a controller that never played the real game reaches
  60.7%, close to 63.1% from real-game training, in ~1 minute instead of ~2.4 hours.
  τ = 1.25 is slightly better than τ = 1.0, as in the paper, but the difference is
  within noise.
- **Not yet learned: inferring the wind.** The best agent is below the
  velocity-only scripted player (73.9%) and far from the wind-aware one (99.6%).
- M learned to track the invisible ball: on test data its predicted catch probability
  is 0.75 on real catches vs 0.21 on real misses (needs class-balanced event weights;
  with equal weights it always predicted "miss").

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


## New environment: PixelCatcher with hidden wind and fog

A 64×64 pixel game: a ball falls from the top and a paddle at the bottom catches it
(actions: left / stay / right; +1 catch, −1 miss, −0.005 per step; 300 steps ≈ 10 balls).

**Why the original PixelCatcher was not enough.** Its paddle is faster than the ball,
so simply following the ball's current position catches 100% of balls; memory is
useless and V only would equal V+M. Two additions make memory matter
(`world_models/env.py`, both off by default):

- **Fog** (`fog_y = 25`): the ball is invisible in the lower part of the screen, so its
  landing point must be extrapolated from the trajectory seen before.
- **Hidden wind** (`wind_range = 0.08`): each episode draws a horizontal acceleration
  that bends every ball's path and is never shown; it can only be inferred from history.

Scripted players calibrate the design (privileged state, same test seeds): no memory
39.6%, velocity 73.9%, velocity + wind 99.6%.

**Model and training** (`world_models/catcher_pipeline.py`, `configs/catcher.toml`):
1,000 random-policy episodes (sticky random actions), 80/10/10 episode split; V and M
as in CarRacing (z = 32, LSTM 256, 5 mixtures). M additionally predicts the step's event
(miss / nothing / catch) so the dream can hand out rewards, like the done head of the
paper's Doom model; event classes are weighted [10, 1, 34] to balance catches against
the 3.4× more frequent misses. M was trained 160 epochs (the reference's 4,000 steps);
320 epochs overfit the event head and were discarded on validation. The controller
follows the paper's Doom code: one tanh output without bias, mapped to left/stay/right
at ±1/3. CMA-ES uses the reference settings (population 64 × 16 episodes, σ 0.1, weight
decay 0.01). Dream rollouts are batched inside M with temperature τ (mixture logits / τ,
noise × √τ) and common random numbers across candidates; dream-trained controllers are
selected on dream validation only and never see the real game before the test.

```bash
bash scripts/run_catcher.sh world        # collect, V, M, held-out metrics, scripted baselines
bash scripts/run_catcher.sh controllers  # dream (τ 1.0, 1.25), real V+M, real V only
```

On an Apple M3 the whole experiment takes ~6 hours, most of it the two real-game
controllers. Do not edit `scripts/run_catcher.sh` while it runs (bash reads it lazily).

**Next steps.** Iterative training (re-collect data with a trained controller, which
catches more balls, then retrain V and M), probing whether `h` encodes the wind, more
temperatures with repeated seeds, and a harder Breakout-style environment.

## Scope and differences from the paper

- CarRacing reproduces the architecture, data volume and optimizer settings of the
  authors' code but runs 50 of 1,800 CMA-ES generations, on Gymnasium CarRacing-v3
  rather than gym 0.9.4, with a smooth random collection policy (see above).
- The paper's VizDoom experiment was not reproduced; its dream-training method is
  applied to PixelCatcher instead.
- `world_models/pipeline.py` is the earlier small PixelCatcher prototype (16-D latent,
  128-unit LSTM, no fog or wind) and is superseded by `catcher_pipeline.py`.
