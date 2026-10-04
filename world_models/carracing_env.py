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

    def __init__(self, seed=0, render_mode="rgb_array", invert=True, max_steps=1000,
                 ignore_termination=False):
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise RuntimeError("Install the CarRacing extra: pip install '.[carracing]'") from exc
        self.env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True,
                            max_episode_steps=max_steps)
        self.invert = invert
        self.seed = seed
        # The reference extract.py used full_episode=True: always run max_steps frames.
        self.ignore_termination = ignore_termination

    def reset(self, seed=None):
        frame, info = self.env.reset(seed=self.seed if seed is None else seed)
        return preprocess_frame(frame, self.invert), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, [-1., 0., 0.], [1., 1., 1.])
        frame, reward, terminated, truncated, info = self.env.step(action)
        if self.ignore_termination: terminated = False
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


class ReferenceRandomPolicy:
    """The random policy of the reference extract.py.

    Every weight of a fresh V, M and linear C is drawn as Cauchy * stdev, with
    stdev ~ U(0, 0.01) per episode, and C acts on [z, h] exactly like the trained
    agent. The result is a mostly steady, occasionally extreme, per-episode policy.
    """

    def __init__(self, seed=0, latent_size=32, hidden_size=256):
        import torch
        from .models import VAE, MDNRNN
        self.torch = torch
        self.rng = np.random.default_rng(seed); stdev = self.rng.random() * .01
        self.vae, self.mdn = VAE(latent_size), MDNRNN(latent_size, 3, hidden_size, 5, False)
        with torch.no_grad():
            for module in (self.vae, self.mdn):
                for param in module.parameters():
                    param.copy_(torch.from_numpy(self.rng.standard_cauchy(param.shape) * stdev))
            self.mdn.rnn.bias_hh_l0.zero_()  # TF's LSTM has one bias; keep the redundant one at 0
        self.bias = self.rng.standard_cauchy(3) * stdev
        self.weight = self.rng.standard_cauchy((latent_size + hidden_size, 3)) * stdev
        self.hidden = None; self.hidden_size = hidden_size; self.overflowed = False

    def __call__(self, obs):
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(obs).permute(2, 0, 1)[None].float() / 255.
            mu, logvar = (t[0].double().numpy() for t in self.vae.encode(x))
            with np.errstate(over="ignore", invalid="ignore"):
                z = mu + np.exp(logvar / 2) * self.rng.standard_normal(mu.shape)
            # Heavy-tailed weights can overflow; saturate instead of propagating NaN.
            z = np.clip(np.nan_to_num(z, nan=0.), -1e6, 1e6); self.overflowed |= bool((abs(z) >= 1e6).any())
            h = self.hidden[0][-1, 0].double().numpy() if self.hidden else np.zeros(self.hidden_size)
            action = np.tanh(np.concatenate([z, h]) @ self.weight + self.bias)
            action[1] = (action[1] + 1.) / 2.; action[2] = np.clip(action[2], 0., 1.)
            z_t = torch.as_tensor(z, dtype=torch.float32).view(1, -1)
            self.hidden, _ = self.mdn.step_hidden(z_t, action.astype(np.float32), self.hidden)
        return action.astype(np.float32)
