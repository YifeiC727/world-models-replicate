"""A tiny visual control environment with no Gym dependency.

With the defaults, the paddle is faster than the ball, so simply following the ball
catches everything and memory is unnecessary. Two options make memory matter:

- ``fog_y``: the ball is invisible once its centre is at or below this row, so the
  landing point must be extrapolated from the trajectory seen before.
- ``wind_range``: each episode draws a hidden horizontal acceleration
  ``wind ~ U(-wind_range, wind_range)`` that bends every ball's path; it is never
  shown and must be inferred from how balls have moved so far.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PixelCatcherConfig:
    width: int = 64
    height: int = 64
    horizon: int = 160
    paddle_width: int = 13
    paddle_speed: float = 3.0
    ball_speed: float = 1.8
    wind_range: float = 0.0
    fog_y: int | None = None
    max_ball_vx: float = 2.0


class PixelCatcher:
    """Catch bouncing balls using actions 0=left, 1=stay, 2=right."""

    action_size = 3

    def __init__(self, config: PixelCatcherConfig | None = None, seed: int = 0):
        self.cfg = config or PixelCatcherConfig()
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.score = 0.0
        self.paddle_x = self.cfg.width / 2
        # Draw only when enabled, so the default environment keeps its random stream.
        self.wind = float(self.rng.uniform(-self.cfg.wind_range, self.cfg.wind_range)) if self.cfg.wind_range else 0.0
        self._spawn_ball()
        return self._render(), {}

    def _spawn_ball(self):
        self.ball_x = float(self.rng.uniform(6, self.cfg.width - 6))
        self.ball_y = 5.0
        self.ball_vx = float(self.rng.choice([-1.0, 1.0]) * self.rng.uniform(.45, 1.25))

    def step(self, action: int):
        if action not in (0, 1, 2):
            raise ValueError("action must be 0, 1, or 2")
        self.paddle_x += (action - 1) * self.cfg.paddle_speed
        half = self.cfg.paddle_width / 2
        self.paddle_x = float(np.clip(self.paddle_x, half, self.cfg.width - 1 - half))
        self.ball_vx = float(np.clip(self.ball_vx + self.wind, -self.cfg.max_ball_vx, self.cfg.max_ball_vx))
        self.ball_x += self.ball_vx
        self.ball_y += self.cfg.ball_speed
        if self.ball_x < 2 or self.ball_x > self.cfg.width - 3:
            self.ball_vx *= -1
            self.ball_x = float(np.clip(self.ball_x, 2, self.cfg.width - 3))
        reward = -0.005; event = 0
        if self.ball_y >= self.cfg.height - 7:
            caught = abs(self.ball_x - self.paddle_x) <= half + 1
            reward = 1.0 if caught else -1.0; event = 1 if caught else -1
            self._spawn_ball()
        self.t += 1
        self.score += reward
        terminated = self.t >= self.cfg.horizon
        return self._render(), reward, terminated, False, {"score": self.score, "event": event, "wind": self.wind}

    def _render(self) -> np.ndarray:
        frame = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        # Background gradient prevents the VAE task from being completely trivial.
        frame[..., 2] = np.linspace(5, 25, self.cfg.height, dtype=np.uint8)[:, None]
        x, y = int(round(self.ball_x)), int(round(self.ball_y))
        if self.cfg.fog_y is None or y < self.cfg.fog_y:
            frame[max(0, y-2):y+3, max(0, x-2):x+3] = (245, 210, 45)
        if self.cfg.fog_y is not None:
            frame[self.cfg.fog_y:, :, :2] = 55  # grey band; the paddle is drawn on top
        half = self.cfg.paddle_width // 2
        px = int(round(self.paddle_x))
        frame[-5:-2, max(0, px-half):min(self.cfg.width, px+half+1)] = (50, 220, 120)
        return frame

