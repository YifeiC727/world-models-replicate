"""Rollout collection and simple NPZ dataset helpers."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from .env import PixelCatcher


def collect_random_rollouts(path: str | Path, episodes=200, seed=0):
    env = PixelCatcher(seed=seed)
    rng = np.random.default_rng(seed)
    observations, actions, rewards = [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed + episode)
        ep_obs, ep_actions, ep_rewards = [obs], [], []
        done = False
        while not done:
            action = int(rng.integers(env.action_size))
            obs, reward, done, _, _ = env.step(action)
            ep_obs.append(obs); ep_actions.append(action); ep_rewards.append(reward)
        observations.append(np.stack(ep_obs))
        actions.append(np.asarray(ep_actions, dtype=np.int64))
        rewards.append(np.asarray(ep_rewards, dtype=np.float32))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, observations=np.stack(observations),
                        actions=np.stack(actions), rewards=np.stack(rewards))
    return path


def frames_to_tensor(frames, device="cpu"):
    import torch
    x = torch.as_tensor(frames, dtype=torch.float32, device=device) / 255.0
    return x.permute(0, 3, 1, 2)
