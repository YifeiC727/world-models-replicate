"""Videos of trained agents, in the real game and inside the learned world model.

    python -m world_models.visualize carracing-real --mode vm --seed 1000004
    python -m world_models.visualize carracing-compare --seed 1000008
    python -m world_models.visualize carracing-dream --seed 1000004 --temperature 1.0
    python -m world_models.visualize catcher --seed 1000003

Outputs go to ``media/`` as MP4 (and a short GIF with ``--gif``). Scores are replayed
on this machine; floating-point differences between CPUs change individual tracks,
so they can differ from the server's test scores (the distributions agree).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .carracing_env import CarRacingAdapter
from .carracing_pipeline import action_from_logits, load_agent, load_config
from .data import frames_to_tensor

MEDIA = Path("media")
FONT = ImageFont.load_default(size=16)
SMALL = ImageFont.load_default(size=13)
BG = (24, 24, 28)
INK = (235, 235, 235)
MUTED = (150, 150, 160)


def panel(image, size, title, note=None):
    """Upscale an RGB array to ``size`` with a title strip above it."""
    img = Image.fromarray(image).resize(size, Image.NEAREST)
    out = Image.new("RGB", (size[0], size[1] + 26), BG); out.paste(img, (0, 26))
    d = ImageDraw.Draw(out); d.text((6, 5), title, font=FONT, fill=INK)
    if note: d.text((size[0] - 6, 7), note, font=SMALL, fill=MUTED, anchor="ra")
    return out


def action_bars(action, width, discrete=False):
    """Steering / gas / brake bars (CarRacing) or left / stay / right (PixelCatcher)."""
    out = Image.new("RGB", (width, 34), BG); d = ImageDraw.Draw(out)
    if discrete:
        for i, name in enumerate(["left", "stay", "right"]):
            x = 8 + i * 70; on = i == action
            d.rectangle((x, 8, x + 60, 26), fill=(80, 170, 255) if on else (55, 55, 62))
            d.text((x + 30, 17), name, font=SMALL, fill=INK if on else MUTED, anchor="mm")
        return out
    labels = [("steer", action[0], True), ("gas", action[1], False), ("brake", action[2], False)]
    w = (width - 40) // 3
    for i, (name, value, signed) in enumerate(labels):
        x0 = 8 + i * (w + 12); d.text((x0, 9), name, font=SMALL, fill=MUTED)
        bx0, bx1 = x0 + 46, x0 + w; d.rectangle((bx0, 11, bx1, 23), fill=(55, 55, 62))
        if signed:
            mid = (bx0 + bx1) // 2; end = mid + value * (bx1 - bx0) / 2
            d.rectangle((min(mid, end), 11, max(mid, end), 23), fill=(80, 170, 255))
        else:
            d.rectangle((bx0, 11, bx0 + value * (bx1 - bx0), 23), fill=(80, 170, 255) if name == "gas" else (240, 110, 90))
    return out


def stack(rows, header=None, width=None):
    width = width or max(r.width for r in rows)
    height = sum(r.height for r in rows) + (34 if header else 0)
    out = Image.new("RGB", (width, height), BG); y = 0
    if header:
        ImageDraw.Draw(out).text((8, 8), header, font=FONT, fill=INK); y = 34
    for r in rows: out.paste(r, (0, y)); y += r.height
    return out


def hstack(images, gap=8):
    out = Image.new("RGB", (sum(i.width for i in images) + gap * (len(images) - 1), max(i.height for i in images)), BG)
    x = 0
    for i in images: out.paste(i, (x, 0)); x += i.width + gap
    return out


def save(frames, name, fps, gif=False):
    MEDIA.mkdir(exist_ok=True)
    frames = [np.asarray(f.convert("RGB")) for f in frames]
    h, w = frames[0].shape[:2]
    pad = lambda f: np.pad(f, ((0, h % 2), (0, w % 2), (0, 0)), constant_values=BG[0])  # H.264 needs even sizes
    imageio.mimsave(MEDIA / f"{name}.mp4", [pad(f) for f in frames], fps=fps, codec="libx264", quality=8,
                    macro_block_size=1)
    print(f"wrote {MEDIA / f'{name}.mp4'} ({len(frames)} frames)")
    if gif:  # short, small excerpt for README / slides
        small = [Image.fromarray(f).resize((w // 2, h // 2)) for f in frames[:600:3]]
        small[0].save(MEDIA / f"{name}.gif", save_all=True, append_images=small[1:], duration=1000 // (fps // 3), loop=0)
        print(f"wrote {MEDIA / f'{name}.gif'}")


def view(obs, invert):
    """The VAE sees colour-inverted frames (as in the reference); invert back for display."""
    return 255 - obs if invert else obs


# ---------------------------------------------------------------- CarRacing

@torch.no_grad()
def carracing_episode(config, use_memory, seed, record=True):
    """Yields per-step dicts for one real episode, mirroring carracing_pipeline.rollout."""
    vae, mdn, controller = load_agent(config, "cpu", use_memory)
    suffix = "vm" if use_memory else "v"
    controller.set_flat_parameters(np.load(Path(config["experiment"]["workdir"]) / f"controller_{suffix}.npy"))
    torch.manual_seed(seed); e = config["environment"]
    env = CarRacingAdapter(seed, invert=e["invert_colors"], max_steps=e["max_steps"])
    obs, _ = env.reset(seed); hidden = None; total = 0.; done = False; t = 0
    while not done:
        mu, logvar = vae.encode(frames_to_tensor(obs[None])); z = vae.reparameterize(mu, logvar)
        h = hidden[0][-1] if hidden else torch.zeros(1, mdn.hidden_size)
        action = action_from_logits(controller(z, h))[0].numpy()
        step = {"t": t, "obs": obs, "action": action, "total": total}
        if record:
            step["game"] = env.env.render()[:350]  # drop the HUD strip with its unreadable score
            step["recon"] = (vae.decode(mu)[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        obs, reward, terminated, truncated, _ = env.step(action); total += reward; done = terminated or truncated
        hidden, _ = mdn.step_hidden(z, action, hidden); t += 1
        yield step
    env.close()


def carracing_real(config, mode, seed, fps, gif):
    invert = config["environment"]["invert_colors"]; frames = []
    label = "full model (V+M)" if mode == "vm" else "V only"
    for s in carracing_episode(config, mode == "vm", seed):
        game = panel(s["game"], (525, 306), "CarRacing (real game)", f"step {s['t']}")
        seen = panel(view(s["obs"], invert), (306, 306), "V input 64x64")
        recon = panel(view(s["recon"], invert), (306, 306), "VAE reconstruction")
        row = hstack([game, seen, recon])
        frames.append(stack([row, action_bars(s["action"], row.width)],
                            header=f"{label} | track seed {seed} | reward {s['total']:.0f}"))
    save(frames, f"carracing_real_{mode}_{seed}", fps, gif)


def carracing_compare(config, seed, fps, gif):
    runs = {m: list(carracing_episode(config, m == "vm", seed)) for m in ("vm", "v")}
    n = max(len(r) for r in runs.values()); frames = []
    for i in range(n):
        cols = []
        for m, title in (("v", "V only (z)"), ("vm", "full model (z + h)")):
            s = runs[m][min(i, len(runs[m]) - 1)]; final = i >= len(runs[m])
            p = panel(s["game"], (450, 262), title, "finished" if final else f"step {s['t']}")
            score = runs[m][-1]["total"] if final else s["total"]
            cols.append(stack([p, action_bars(s["action"], p.width)], header=f"reward {score:.0f}"))
        frames.append(stack([hstack(cols, gap=12)], header=f"Same unseen track (seed {seed}): does memory help?"))
    save(frames, f"carracing_compare_{seed}", fps, gif)


@torch.no_grad()
def sample_next_z(logits, mu, log_sigma, temperature, generator):
    """Reference sampling: mixture weights logits / tau, Gaussian noise * sqrt(tau)."""
    probs = torch.softmax(logits[0, 0] / temperature, -1)  # (latent, mixtures)
    k = torch.multinomial(probs, 1, generator=generator)
    mean = mu[0, 0].gather(-1, k).squeeze(-1); std = log_sigma[0, 0].gather(-1, k).squeeze(-1).exp()
    return (mean + std * torch.randn(mean.shape, generator=generator) * math.sqrt(temperature))[None]


@torch.no_grad()
def carracing_dream(config, seed, temperature, steps, fps, gif):
    """C drives inside M: no game engine, every frame is M's sampled z decoded by V."""
    vae, mdn, controller = load_agent(config, "cpu", True)
    controller.set_flat_parameters(np.load(Path(config["experiment"]["workdir"]) / "controller_vm.npy"))
    invert = config["environment"]["invert_colors"]; g = torch.Generator().manual_seed(seed)
    env = CarRacingAdapter(seed, invert=invert); obs, _ = env.reset(seed); env.close()
    mu, logvar = vae.encode(frames_to_tensor(obs[None])); z = mu + torch.randn(mu.shape, generator=g) * (.5 * logvar).exp()
    hidden = None; frames = []
    for t in range(steps):
        h = hidden[0][-1] if hidden else torch.zeros(1, mdn.hidden_size)
        action = action_from_logits(controller(z, h))
        image = (vae.decode(z)[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        dream = panel(view(image, invert), (448, 448), "dream: M imagines, V decodes", f"step {t}")
        frames.append(stack([dream, action_bars(action[0].numpy(), dream.width)],
                            header=f"No game engine | temperature {temperature:g}"))
        logits, m, log_sigma, hidden = mdn(z[:, None], action[:, None], hidden)
        z = sample_next_z(logits, m, log_sigma, temperature, g)
    save(frames, f"carracing_dream_t{temperature:g}_{seed}", fps, gif)


# ---------------------------------------------------------------- PixelCatcher

@torch.no_grad()
def catcher(config_path, seed, temperature, fps, gif):
    """Left: the real game (fog hides the ball) played by the real-trained V+M controller.
    Right: a dream episode imagined by M, played by the dream-trained controller."""
    from .catcher_pipeline import EVENTS, env_config, load_models, make_controller, to_action
    from .env import PixelCatcher
    config = load_config(config_path); root = Path(config["experiment"]["workdir"])
    vae, mdn = load_models(config, "cpu")
    real_c, dream_c = make_controller(config, True), make_controller(config, True)
    real_c.set_flat_parameters(np.load(root / "controller_real_vm.npy"))
    dream_c.set_flat_parameters(np.load(root / f"controller_dream_vm_t{temperature:g}.npy"))
    torch.manual_seed(seed); env = PixelCatcher(env_config(config), seed); obs, _ = env.reset(seed)
    g = torch.Generator().manual_seed(seed)
    mu, logvar = vae.encode(frames_to_tensor(obs[None])); dz = mu + torch.randn(mu.shape, generator=g) * (.5 * logvar).exp()
    hidden = dream_hidden = None; caught = balls = d_caught = d_balls = 0; frames = []; flash = {"real": 0, "dream": 0}
    note = {"real": "", "dream": ""}
    for t in range(env.cfg.horizon):
        mu, logvar = vae.encode(frames_to_tensor(obs[None])); z = vae.reparameterize(mu, logvar)
        h = hidden[0][-1] if hidden else torch.zeros(1, mdn.hidden_size)
        action = int(to_action(real_c(z, h)))
        dh = dream_hidden[0][-1] if dream_hidden else torch.zeros(1, mdn.hidden_size)
        d_action = to_action(dream_c(dz, dh))
        d_image = (vae.decode(dz)[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        real = panel(obs, (320, 320), "real game (fog hides the ball)", f"t {t}")
        dream = panel(d_image, (320, 320), "dream (imagined by M)", f"tau {temperature:g}")
        for key, p in (("real", real), ("dream", dream)):
            if flash[key]:
                ImageDraw.Draw(p).text((p.width // 2, 60), note[key], font=FONT, anchor="mm",
                                       fill=(90, 230, 120) if note[key] == "catch" else (240, 110, 90))
        cols = [stack([real, action_bars(action, real.width, True)], header=f"caught {caught}/{balls} | wind {env.wind:+.3f} (hidden)"),
                stack([dream, action_bars(int(d_action), dream.width, True)], header=f"caught {d_caught}/{d_balls} in the dream")]
        frames.append(hstack(cols, gap=12))
        flash = {k: max(0, v - 1) for k, v in flash.items()}
        obs, _, _, _, info = env.step(action)
        if info["event"]: balls += 1; caught += info["event"] == 1; note["real"] = "catch" if info["event"] == 1 else "miss"; flash["real"] = 8
        hidden, _ = mdn.step_hidden(z, action, hidden)
        logits, m, log_sigma, dream_hidden, event_logits = mdn(dz[:, None], d_action.view(1, 1), dream_hidden, return_events=True)
        event = int(torch.multinomial(torch.softmax(event_logits[0, 0], -1), 1, generator=g)) - 1
        if event: d_balls += 1; d_caught += event == 1; note["dream"] = "catch" if event == 1 else "miss"; flash["dream"] = 8
        dz = sample_next_z(logits, m, log_sigma, temperature, g)
    save(frames, f"catcher_real_vs_dream_{seed}", fps, gif)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["carracing-real", "carracing-compare", "carracing-dream", "catcher"])
    p.add_argument("--config", default=None); p.add_argument("--mode", choices=["vm", "v"], default="vm")
    p.add_argument("--seed", type=int, default=1_000_004); p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=400); p.add_argument("--fps", type=int, default=30)
    p.add_argument("--gif", action="store_true")
    a = p.parse_args()
    if a.command == "catcher":
        return catcher(a.config or "configs/catcher.toml", a.seed, a.temperature, a.fps, a.gif)
    config = load_config(a.config or "configs/carracing-deepdish.toml")
    if a.command == "carracing-real": carracing_real(config, a.mode, a.seed, a.fps, a.gif)
    elif a.command == "carracing-compare": carracing_compare(config, a.seed, a.fps, a.gif)
    else: carracing_dream(config, a.seed, a.temperature, a.steps, a.fps, a.gif)


if __name__ == "__main__": main()
