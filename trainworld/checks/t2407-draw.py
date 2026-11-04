"""T-2407: a picture of one case t2407-blocks.mjs wrote (lines, pills, each city's box and the line's stops named).
   python trainworld/checks/t2407-draw.py <folder under artifacts/trainworld/t2407> <case> <out.png> [--title t]"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "artifacts", "trainworld", "t2407")
ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("case")
ap.add_argument("out")
ap.add_argument("--title", default="")
ap.add_argument("--width", type=float, default=12)
a = ap.parse_args()
D = json.load(open(os.path.join(ART, a.folder, a.case + ".json"), encoding="utf8"))
pu = D["pitch"]
BG, INK = "#f4f1ea", "#2b2b2b"
xs, ys = D["pos"]["x"], D["pos"]["y"]
x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
padx, pady = (x1 - x0) * 0.04 + 1, (y1 - y0) * 0.04 + 1
x0, x1, y0, y1 = x0 - padx, x1 + padx, y0 - pady, y1 + pady
h = max(2.5, min(a.width * (y1 - y0) / (x1 - x0), 14))
fig = plt.figure(figsize=(a.width, h), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
ax.axis("off")
ppu = a.width * 72 / (x1 - x0)
lw = max(0.6, D["stroke"] * ppu)
for b in D["boxes"]:
    bx, p = b["box"], 6 * pu
    ax.add_patch(FancyBboxPatch((bx[0] - p, bx[1] - p), bx[2] - bx[0] + 2 * p, bx[3] - bx[1] + 2 * p, boxstyle=f"round,pad=0,rounding_size={8*pu}", fc=INK, alpha=0.05, ec=INK, lw=0.8))
    ax.text(bx[0] - p + 4 * pu, bx[3] + p - 4 * pu, b["tag"], fontsize=8, color=INK, alpha=0.75, va="top", ha="left")
for r in D["ribbons"]:
    p = r["p"]
    ax.plot(p[0::2], p[1::2], color=r["c"], lw=lw, solid_capstyle="round", solid_joinstyle="round")
    if r["f"]:
        ax.plot(p[0::2], p[1::2], color=BG, lw=lw * 0.34, solid_capstyle="round")
for p in D["pills"]:
    w = max(0.8, 2 * p[5] * ppu)
    ax.plot([p[0], p[2]], [p[1], p[3]], color=INK, lw=w + max(0.4, 2 * p[6] * ppu), solid_capstyle="round")
    ax.plot([p[0], p[2]], [p[1], p[3]], color="white", lw=w, solid_capstyle="round")
for s in D["line"]:
    ax.text(s["x"], s["y"] + 6 * pu, s["name"], fontsize=7, color=INK, ha="center", va="bottom")
f = D["fig"]
title = a.title or a.case
ax.text(0.01, 0.99, f"{title}\nline {f['spanPitches']} pitches, New York {f['nyWidth']} wide; layout {f['ms'] / 1000:.1f} s", transform=ax.transAxes, fontsize=10, color=INK, va="top", ha="left")
fig.savefig(a.out, dpi=150, facecolor=BG)
print(a.out)
