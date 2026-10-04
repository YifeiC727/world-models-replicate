"""Disk-backed trajectory collection and datasets for CarRacing."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .carracing_env import CarRacingAdapter, CorrelatedRandomPolicy
from .data import frames_to_tensor


def _collect_episode(job):
    root, episode, episode_seed, max_steps, invert = job
    path = root / f"episode_{episode:05d}.npz"
    if path.exists():  # finished earlier; files appear atomically, so they are complete
        with np.load(path) as data: steps = len(data["actions"])
        return {"file": path.name, "seed": episode_seed, "steps": steps}, None
    env = CarRacingAdapter(seed=episode_seed, invert=invert, max_steps=max_steps)
    policy = CorrelatedRandomPolicy(episode_seed)
    obs, _ = env.reset(episode_seed)
    observations, actions, rewards = [obs], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs); actions.append(action); rewards.append(reward)
    env.close()
    partial = path.with_name("tmp_" + path.name)  # not matched by episode_*.npz
    np.savez_compressed(partial, observations=np.asarray(observations, dtype=np.uint8),
                        actions=np.asarray(actions, dtype=np.float32),
                        rewards=np.asarray(rewards, dtype=np.float32), seed=episode_seed)
    partial.replace(path)
    return {"file": path.name, "seed": episode_seed, "steps": len(actions)}, float(sum(rewards))


def collect_rollouts(root, episodes=10, seed=0, max_steps=1000, invert=True, workers=1):
    """Collect random-policy episodes; reruns skip finished files, so it is resumable.

    Episode ``i`` always uses seed ``seed + i``, so the data do not depend on ``workers``.
    """
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    jobs = [(root, episode, seed + episode, max_steps, invert) for episode in range(episodes)]
    if workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(workers)
        results = pool.imap_unordered(_collect_episode, jobs, chunksize=1)
    else:
        pool = None; results = map(_collect_episode, jobs)
    manifest = []
    for done, (entry, reward) in enumerate(results, 1):
        manifest.append(entry)
        status = "skipped" if reward is None else f"reward={reward:.2f}"
        print(f"collect {done}/{episodes} {entry['file']} steps={entry['steps']} {status}")
    if pool: pool.close(); pool.join()
    manifest.sort(key=lambda entry: entry["file"])
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def episode_files(source):
    """Accept a directory of episodes or an explicit list of episode paths."""
    if isinstance(source, (str, Path)): return sorted(Path(source).glob("episode_*.npz"))
    return [Path(path) for path in source]


def split_episodes(names, validation_fraction=.1, test_fraction=.1, seed=0):
    """Deterministic episode-level split, so correlated frames never cross splits."""
    names = sorted(names); order = np.random.default_rng(seed).permutation(len(names))
    if len(names) < 3: raise ValueError("need at least 3 episodes for train/validation/test")
    n_test = max(1, int(round(len(names) * test_fraction)))
    n_val = max(1, int(round(len(names) * validation_fraction)))
    pick = lambda idx: sorted(names[i] for i in idx)
    return {"test": pick(order[:n_test]), "validation": pick(order[n_test:n_test+n_val]),
            "train": pick(order[n_test+n_val:])}


class FrameDataset(Dataset):
    def __init__(self, source, cache_size=2):
        self.files = episode_files(source)
        if not self.files: raise ValueError(f"no episodes found in {source}")
        self.lengths = [len(np.load(path, mmap_mode="r")["observations"]) for path in self.files]
        self.cumulative = np.cumsum([0] + self.lengths)
        self.cache_size = cache_size; self.cache = OrderedDict()

    def __len__(self): return int(self.cumulative[-1])

    def _episode(self, index):
        if index not in self.cache:
            self.cache[index] = np.load(self.files[index])["observations"]
            if len(self.cache) > self.cache_size: self.cache.popitem(last=False)
        return self.cache[index]

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.cumulative, index, side="right") - 1)
        offset = index - self.cumulative[episode]
        return frames_to_tensor(self._episode(episode)[offset:offset+1])[0]


class EpisodeChunkSampler(Sampler):
    """Shuffle episodes into chunks of ``chunk_size`` and shuffle frames within each chunk.

    A global shuffle over compressed episodes re-decompresses a whole episode for almost
    every frame. Chunking keeps each episode resident while its frames are consumed, so
    ``FrameDataset.cache_size`` must be at least ``chunk_size``.
    """
    def __init__(self, dataset, chunk_size=16, seed=0):
        self.dataset = dataset; self.chunk_size = chunk_size
        self.seed = seed; self.epoch = 0

    def __len__(self): return len(self.dataset)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch); self.epoch += 1
        bounds = self.dataset.cumulative; order = rng.permutation(len(self.dataset.files))
        for start in range(0, len(order), self.chunk_size):
            chunk = np.concatenate([np.arange(bounds[e], bounds[e+1])
                                    for e in order[start:start+self.chunk_size]])
            yield from rng.permutation(chunk).tolist()


class LatentSequenceDataset(Dataset):
    """Windows of (z[t], a[t], z[t+1]); ``deterministic`` fixes the z sample per window."""
    def __init__(self, source, sequence_length=128, deterministic=False):
        self.files = episode_files(source)
        self.sequence_length = sequence_length; self.deterministic = deterministic
        self.index = []
        for file_index, path in enumerate(self.files):
            n = len(np.load(path, mmap_mode="r")["actions"])
            for start in range(0, n - sequence_length + 1, sequence_length):
                self.index.append((file_index, start))
        if not self.index: raise ValueError("no latent sequence windows available")

    def __len__(self): return len(self.index)

    def __getitem__(self, index):
        file_index, start = self.index[index]
        data = np.load(self.files[file_index])
        stop = start + self.sequence_length
        mu = torch.from_numpy(data["mu"][start:stop+1].astype(np.float32))
        logvar = torch.from_numpy(data["logvar"][start:stop+1].astype(np.float32))
        generator = torch.Generator().manual_seed(index) if self.deterministic else None
        noise = torch.randn(mu.shape, generator=generator)
        z = mu + noise * torch.exp(.5 * logvar)
        actions = torch.from_numpy(data["actions"][start:stop].astype(np.float32))
        return z[:-1], actions, z[1:]
