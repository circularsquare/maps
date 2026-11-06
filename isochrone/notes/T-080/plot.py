"""T-080 box plots: fine/coarse time ratio by kind of land, now and with crossing speeds."""
import json
import os
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\anita\projects\maps\artifacts\isochrone"
pure = json.load(open(os.path.join(SCR, "results", "pure_walk.json")))
proto = json.load(open(os.path.join(SCR, "results", "proto.json")))
KINDS = ["farmland", "mountains", "lakes", "desert", "forest", "city", "coast/delta"]


def ratios(data, a, b, kind, minD=20):
    out = []
    for r in data:
        if r["kind"] != kind:
            continue
        for d in r["dests"]:
            ta, tb = d.get(a), d.get(b)
            if d["D"] < minD or ta is None or tb is None or not tb or math.isinf(ta) or math.isinf(tb):
                continue
            out.append(ta / tb)
    return out


fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, k in zip(axes, ["10", "3"]):
    pos, data, cols = [], [], []
    for i, kind in enumerate(KINDS):
        now = ratios(proto, "t1", "t" + k + "cur", kind)
        fix = ratios(proto, "t1", "t" + k + "xmin", kind)
        data += [now, fix]
        pos += [3 * i, 3 * i + 1]
        cols += ["#c0504d", "#4f81bd"]
    bp = ax.boxplot(data, positions=pos, widths=0.8, whis=(10, 90), showfliers=False, patch_artist=True)
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c)
        p.set_alpha(0.7)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks([3 * i + 0.5 for i in range(len(KINDS))])
    ax.set_xticklabels(KINDS, rotation=30)
    ax.set_title("L1 time / L%s time, trips 20-300 km" % k)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("ratio (below 1: L1 faster)")
axes[0].plot([], [], color="#c0504d", lw=8, alpha=0.7, label="coarse as built now (mean time per metre)")
axes[0].plot([], [], color="#4f81bd", lw=8, alpha=0.7, label="coarse from fine crossing times")
axes[0].legend(loc="upper left", fontsize=9)
fig.text(0.5, 0.005, "boxes: quartiles; whiskers: 10th-90th percentile", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "t080_ratios.png"), dpi=110)
print("wrote t080_ratios.png")
