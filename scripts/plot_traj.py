"""Plot prophet's Elo-vs-games learning curve from the trajectory gauntlets.

Reads runs/moonshot10m/traj_<game>.json (256fw deployed) and the deep-sims
traj_<game>_{512,1024}fw.json if present, and saves results/trajectory.png.
Re-run after the deep-sims sweep to add the latent-strength curves.
"""
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "runs/moonshot10m"
GAMES = [10000, 15000, 20000, 25000, 30000]


def load(suffix=""):
    pts = []
    for g in GAMES:
        # 30k deep-sims may not be re-run yet under traj_ — fall back to the
        # earlier session's gauntlet_30000_*fw.json so the curve is complete.
        candidates = [f"{RUN}/traj_{g:06d}{suffix}.json"]
        if g == 30000:
            candidates.append(f"{RUN}/gauntlet_{g}{suffix}.json")
        for f in candidates:
            if os.path.exists(f):
                pts.append((g / 1000.0, json.load(open(f))["elo"]))
                break
    return pts


def line(ax, pts, **kw):
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", **kw)


fig, ax = plt.subplots(figsize=(8.5, 5.5))
line(ax, load(""), label="256 fw (deployed / standard)", color="#1f77b4", lw=2.5)
line(ax, load("_512fw"), label="512 fw (deep search)", color="#ff7f0e", lw=2, ls="--")
line(ax, load("_1024fw"), label="1024 fw (latent ceiling)", color="#2ca02c", lw=2, ls="--")

ax.axhline(1151, color="gray", ls=":", lw=1)
ax.text(30.1, 1151, "official 30k = 1151", va="center", fontsize=8, color="gray")
ax.set_xlabel("self-play games (thousands)")
ax.set_ylabel("Elo (vs Stockfish ladder, nodes-anchored MLE)")
ax.set_title("Prophet — learning curve from pure self-play (10M net, no human data)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
out = "results/trajectory.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
# also print the numbers
for label, suf in [("256fw", ""), ("512fw", "_512fw"), ("1024fw", "_1024fw")]:
    pts = load(suf)
    if pts:
        print(f"  {label}: " + "  ".join(f"{int(g)}k={e:.0f}" for g, e in pts))
