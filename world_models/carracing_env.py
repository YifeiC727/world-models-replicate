"""Modern Gymnasium adapter matching the 2018 CarRacing preprocessing."""
from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess_frame(frame: np.ndarray, invert: bool = True) -> np.ndarray:
    """Crop the 12-pixel HUD, resize to 64x64, optionally invert colors."""
    if frame.shape != (96, 96, 3):
        raise ValueError(f"expected a 96x96 RGB frame, got {frame.shape}")
    cropped = frame[:84]
    resized = np.asarray(Image.fromarray(cropped).resize((64, 64), Image.Resampling.BILINEAR))
    return (255 - resized) if invert else resized


class CarRacingAdapter:
    action_size = 3
    continuous_actions = True

    def __init__(self, seed=0, render_mode="rgb_array", invert=True, max_steps=1000):
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise RuntimeError("Install the CarRacing extra: pip install '.[carracing]'") from exc
        self.env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True,
                            max_episode_steps=max_steps)
        self.invert = invert
        self.seed = seed

    def reset(self, seed=None):
        frame, info = self.env.reset(seed=self.seed if seed is None else seed)
        return preprocess_frame(frame, self.invert), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, [-1., 0., 0.], [1., 1., 1.])
        frame, reward, terminated, truncated, info = self.env.step(action)
        return preprocess_frame(frame, self.invert), reward, terminated, truncated, info

    def close(self):
        self.env.close()


class CorrelatedRandomPolicy:
    """Smooth exploratory controls, closer to the reference random controller."""

    def __init__(self, seed=0, persistence=20):
        self.rng = np.random.default_rng(seed)
        self.persistence = persistence
        self.action = np.zeros(3, dtype=np.float32)
        self.steps_left = 0

    def __call__(self, _obs):
        if self.steps_left <= 0:
            steering = self.rng.uniform(-1, 1)
            gas = self.rng.uniform(.25, 1.)
            brake = self.rng.uniform(0, .4) if self.rng.random() < .15 else 0.
            self.action = np.asarray([steering, gas, brake], dtype=np.float32)
            self.steps_left = int(self.rng.integers(max(2, self.persistence // 2), self.persistence + 1))
        self.steps_left -= 1
        return self.action.copy()
