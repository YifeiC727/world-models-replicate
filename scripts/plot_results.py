"""Static result figures for the README and reports (writes media/*.png).

    .venv/bin/python scripts/plot_results.py

Needs runs/carracing-deepdish/*_log.json (copied from the server), runs/catcher results,
and the reference repo's training logs in references/WorldModelsExperiments.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"
REF_LOG = ROOT / "references/WorldModelsExperiments/carracing/log/carracing.cma.16.64"

# Reference palette (validated: scripts/validate_palette.js, light mode).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a3"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})
LINE = dict(linewidth=2, solid_capstyle="round", solid_joinstyle="round")
DOT = dict(marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)


def end_label(ax, x, y, text, dy=0):
    """Direct label at the end of a line, in text ink (identity comes from the line)."""
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", color=INK, fontsize=10)


def carracing():
    ours = {m: json.loads((ROOT / f"runs/carracing-deepdish/controller_{m}_log.json").read_text()) for m in ("vm", "v")}
    ref_hist = json.loads(Path(f"{REF_LOG}.hist.json").read_text())[:50]
    ref_eval = {r[0]: r[5] for r in json.loads(Path(f"{REF_LOG}.hist_best.json").read_text()) if r[0] <= 50}

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    series = [("Original V+M", AQUA, [r[0] for r in ref_hist], [r[2] for r in ref_hist], 4),
              ("Ours V+M", BLUE, [r[0] for r in ours["vm"]["history"]], [r[2] for r in ours["vm"]["history"]], 0),
              ("Ours V only", ORANGE, [r[0] for r in ours["v"]["history"]], [r[2] for r in ours["v"]["history"]], 0)]
    for name, color, x, y, dy in series:
        a.plot(x, y, color=color, label=name, **LINE); end_label(a, x[-1], y[-1], name, dy)
    a.set(title="Population mean reward per CMA-ES generation", xlabel="generation", ylabel="reward", xlim=(0, 62))

    gens = [25, 50]
    points = [("Original V+M", AQUA, [ref_eval[g] for g in gens]),
              ("Ours V+M", BLUE, [dict(ours["vm"]["evaluations"])[g] for g in gens]),
              ("Ours V only", ORANGE, [dict(ours["v"]["evaluations"])[g] for g in gens])]
    for name, color, y in points:
        b.plot(gens, y, color=color, label=name, **LINE, **DOT)
        below = name == "Ours V+M"  # keeps 867 and 828 apart at generation 50
        for g, v in zip(gens, y):
            b.annotate(f"{v:.0f}", (g, v), xytext=(0, -16 if below else 9), textcoords="offset points", ha="center",
                       color=INK, fontsize=10)
    b.set(title="Validation (1,024 tracks)", xlabel="generation", ylabel="reward", xticks=gens, xlim=(18, 57),
          ylim=(400, 920))
    b.legend(loc="lower right", fontsize=10)
    fig.suptitle("CarRacing: our replication vs the authors' training log (first 50 of their 1,800 generations)",
                 x=0.01, ha="left", fontsize=12, color=INK_2)
    fig.tight_layout(); fig.savefig(MEDIA / "carracing_curves.png", dpi=160); plt.close(fig)
    print("wrote", MEDIA / "carracing_curves.png")


def catcher():
    rows = [("Random", "reference", "scripted_baselines.json", "random"),
            ("V only (real game)", "agent", "evaluation_real_v.json", None),
            ("No memory (scripted)", "reference", "scripted_baselines.json", "memoryless"),
            ("V+M, dream only, τ=1.0", "agent", "evaluation_dream_vm_t1.json", None),
            ("V+M, dream only, τ=1.25", "agent", "evaluation_dream_vm_t1.25.json", None),
            ("V+M (real game)", "agent", "evaluation_real_vm.json", None),
            ("Remembers velocity (scripted)", "reference", "scripted_baselines.json", "velocity"),
            ("Velocity + true wind (scripted)", "reference", "scripted_baselines.json", "velocity_and_wind")]
    names, values, colors = [], [], []
    for name, kind, file, key in rows:
        d = json.loads((ROOT / "runs/catcher" / file).read_text()); d = d[key] if key else d
        # round half up (0.3805 -> 38.1%), matching the README table
        names.append(name); values.append(round(100 * d["catch_rate"] + 1e-9, 1)); colors.append(BLUE if kind == "agent" else NEUTRAL)
    fig, ax = plt.subplots(figsize=(9, 4.8)); ax.set_axisbelow(True)
    y = range(len(names))[::-1]
    ax.barh(list(y), values, height=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, values):
        ax.annotate(f"{v:.1f}%", (v, yi), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.set_yticks(list(y), names); ax.set_xlim(0, 112); ax.set_xlabel("catch rate on 200 test episodes (%)")
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="World Models agent"), Patch(color=NEUTRAL, label="scripted reference (reads true state)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=10)
    ax.set_title("PixelCatcher with hidden wind and fog", pad=30)
    fig.tight_layout(); fig.savefig(MEDIA / "catcher_results.png", dpi=160); plt.close(fig)
    print("wrote", MEDIA / "catcher_results.png")


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True); carracing(); catcher()
