"""Plot every Elo measurement we have and assess the 2500 question.

Data grid: 2 checkpoints (26.7k, 30k) x 3 search depths (256/512/1024 fw)
x {overall, white, black}. Separates the *deployed* trajectory (256fw, flat)
from the *capability* trajectory (1024fw, rising) and tests both against the
10M network's estimated capacity ceiling.
"""

import torch  # noqa: I001 (before numpy)
import numpy as np

# (games_k, sims, elo) ----------------------------------------------------
OVERALL = {256: [(26.7, 1180), (30, 1151)],
           512: [(26.7, 1304), (30, 1268)],
           1024: [(26.7, 1304), (30, 1399)]}
WHITE = {256: [(26.7, 1234), (30, 1171)],
         512: [(26.7, 1318), (30, 1221)],
         1024: [(26.7, 1374), (30, 1479)]}
BLACK = {256: [(26.7, 1092), (30, 1129)],
         512: [(26.7, 1289), (30, 1310)],
         1024: [(26.7, 1210), (30, 1310)]}
V2 = [(32, 950), (50, 1055), (79, 1265), (100, 1255)]  # 2.77M reference run
CAP_10M = (1800, 2000)   # estimated 10M capacity band
CAP_277M = 1400          # measured v2 ceiling


def slope(pts):
    (x0, y0), (x1, y1) = pts
    return (y1 - y0) / (x1 - x0)  # Elo per 1k games


def report():
    print("\n=== ALL MEASURED ELO POINTS (10M net) ===")
    print(f"{'depth':>6} | {'26.7k → 30k overall':>22} | {'white':>14} | {'black':>14} | slope/1k")
    for s in (256, 512, 1024):
        o, w, b = OVERALL[s], WHITE[s], BLACK[s]
        print(f"{s:>5}fw | {o[0][1]:>6.0f} → {o[1][1]:<5.0f} ({slope(o):+5.0f}/1k) | "
              f"{w[0][1]:>4.0f}→{w[1][1]:<4.0f} | {b[0][1]:>4.0f}→{b[1][1]:<4.0f} | "
              f"O{slope(o):+.0f} W{slope(w):+.0f} B{slope(b):+.0f}")
    print("\n  deployed (256fw) trajectory: FLAT  (overall slope "
          f"{slope(OVERALL[256]):+.0f}/1k games)")
    print("  capability (1024fw) trajectory: RISING (overall slope "
          f"{slope(OVERALL[1024]):+.0f}/1k, White {slope(WHITE[1024]):+.0f}/1k)")

    # how far to 2500 from the best line we have (White @1024fw = 1479)
    gap = 2500 - WHITE[1024][1][1]
    rate = slope(WHITE[1024])  # +Elo/1k at current (early, pre-deceleration)
    print(f"\n=== 2500 CHECK ===")
    print(f"  best current line (White @1024fw): {WHITE[1024][1][1]:.0f}")
    print(f"  gap to 2500: {gap:.0f} Elo")
    print(f"  10M capacity ceiling (est): {CAP_10M[0]}-{CAP_10M[1]}  <-- 2500 is ABOVE this")
    print(f"  naive-linear games to close gap at {rate:+.0f}/1k: "
          f"{gap/rate:.0f}k more games (but Elo DECELERATES, so really far more)")


def plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[matplotlib unavailable: {e}] — text report only")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # -- panel 1: search-scaling at each checkpoint -----------------------
    sims = [256, 512, 1024]
    for ck, mk, c in [(0, "o", "#888"), (1, "s", "#1f77b4")]:
        ax1.plot(sims, [OVERALL[s][ck][1] for s in sims], mk + "-", color=c,
                 label=f"{'26.7k' if ck==0 else '30k'} overall", lw=2)
        ax1.plot(sims, [WHITE[s][ck][1] for s in sims], mk + "--", color="#2ca02c",
                 alpha=0.4 + 0.4*ck, label=f"{'26.7k' if ck==0 else '30k'} White")
    ax1.set_xscale("log", base=2); ax1.set_xticks(sims); ax1.set_xticklabels(sims)
    ax1.set_xlabel("search depth (forwards/move)"); ax1.set_ylabel("Elo")
    ax1.set_title("Search-scaling: deeper thinking → more Elo\n(30k curve sits ABOVE 26.7k at depth = ceiling rose)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # -- panel 2: training trajectory + 2500 question ---------------------
    ax2.plot([x for x, _ in V2], [y for _, y in V2], "^-", color="#aaa",
             label="v2 (2.77M) full run — flattened at capacity", lw=1.5)
    for s, c in [(256, "#d62728"), (1024, "#1f77b4")]:
        xs = [x for x, _ in OVERALL[s]]; ys = [y for _, y in OVERALL[s]]
        ax2.plot(xs, ys, "o-", color=c, lw=2.5, label=f"10M @ {s}fw (measured)")
    # extrapolation cone for deployed 256fw to 100k
    ax2.fill_between([30, 100], [1151, 1450], [1151, 1950], color="#d62728",
                     alpha=0.10, label="10M plausible range → 100k")
    ax2.axhspan(*CAP_10M, color="orange", alpha=0.15, label="10M capacity ceiling (est)")
    ax2.axhline(2000, color="green", ls=":", lw=1.5); ax2.text(101, 2000, "2000 (stretch)", va="center", fontsize=8)
    ax2.axhline(2500, color="red", ls="--", lw=2); ax2.text(101, 2500, "2500 GOAL", va="center", color="red", fontsize=9, weight="bold")
    ax2.axhline(CAP_277M, color="gray", ls=":", lw=1); ax2.text(101, CAP_277M, "2.77M wall", va="center", fontsize=7, color="gray")
    ax2.set_xlim(8, 122); ax2.set_ylim(500, 2650)
    ax2.set_xlabel("self-play games (thousands)"); ax2.set_ylabel("Elo (256fw deployed)")
    ax2.set_title("Trajectory vs the 2500 goal\n(2500 sits above the 10M net's capacity)")
    ax2.legend(fontsize=8, loc="upper left"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = "runs/moonshot10m/trajectory_2500.png"
    fig.savefig(out, dpi=110)
    print(f"\n  saved plot -> {out}")


if __name__ == "__main__":
    report()
    plot()
