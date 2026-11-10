import json, sys, math, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

scenes = json.load(open(sys.argv[1]))
out = sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 4
R = float(sys.argv[4]) if len(sys.argv) > 4 else 25
rows = math.ceil(len(scenes) / cols)
fig, axs = plt.subplots(rows, cols, figsize=(cols * (5.2 if cols <= 2 else 3.4), rows * (5.0 if cols <= 2 else 3.7)), squeeze=False)
for ax in axs.flat:
    ax.axis("off")
for ax, s in zip(axs.flat, scenes):
    ax.axis("on")
    for w in s["osm"]:
        xs = [p[0] for p in w["pts"]]; ys = [p[1] for p in w["pts"]]
        ax.plot(xs, ys, color="#9bbfe0", lw=5, solid_capstyle="round", zorder=1, alpha=0.55)
    for w in s["left"]:
        xs = [p[0] for p in w["pts"]]; ys = [p[1] for p in w["pts"]]
        ax.plot(xs, ys, color="#b0b0b0", lw=1.2, ls=(0, (3, 2)), zorder=2)
    for w in s["game"]:
        xs = [p[0] for p in w["pts"]]; ys = [p[1] for p in w["pts"]]
        col = "#222" if w["lev"] == 0 else "#a050c0"
        if w["w"] >= 2: col = "#c07000"
        ax.plot(xs, ys, color=col, lw=1.6, zorder=3)
        ax.plot(xs, ys, ".", color=col, ms=2.5, zorder=3)
    for j in s["junctions"]:
        ax.plot(j[0], j[1], "D", color="#1a9850", ms=6, zorder=4, mfc="none", mew=1.5)
    ax.plot(0, 0, "o", color="#e41a1c", ms=7, zorder=5, mfc="none", mew=2)
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
    ax.set_xticks([-R / 2, 0, R / 2]); ax.set_yticks([-R / 2, 0, R / 2]); ax.tick_params(labelsize=6)
    ax.set_title(chr(10).join(textwrap.wrap(s.get("label", ""), 52 if cols <= 2 else 40)), fontsize=8)
fig.suptitle(sys.argv[5] if len(sys.argv) > 5 else "", fontsize=10)
fig.text(0.5, 0.005, textwrap.fill("black: game graph after joinRoads (dots are vertices); purple: bridge/tunnel; orange: motorway; dashed grey: left out (alley, path, tagged service); blue band: OSM vehicle ways; green diamond: OSM junction; red ring: the unjoined end", 150 if cols <= 2 else 200), ha="center", fontsize=7, va="bottom")
plt.tight_layout(rect=(0, 0.05 if cols <= 2 else 0.02, 1, 0.96))
plt.savefig(out, dpi=110)
