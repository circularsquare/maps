"""Boston before/after from 2020 TIGER counties (Boston is a hand pack; the plan state has no US units). Read-only."""
import json, math, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import pyogrio, shapely, shapely.ops
import world_plan as wp
sys.path.insert(0, "C:/Users/anita/projects/maps/trainworld/notes/T-4364/kit")
import dry
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TW = r"C:\Users\anita\projects\maps\trainworld"
df = pyogrio.read_dataframe(f"/vsizip/{TW}/data/raw/cb_2020_us_county_500k.zip", columns=["GEOID", "NAME"])
geo = {g: s for g, s in zip(df.GEOID, df.geometry) if g[:2] in ("25", "33", "23", "50", "44", "09")}
POP = {"33015": 314176, "33017": 130889}  # Census 2020
pack = {k: json.load(open(f"{TW}/data/packs/{k}.json"))["totals"]["pop"] for k in ("bos", "his", "wor")}
BOS = ["25009", "25017", "25021", "25023", "25025", "33015", "33017"]
HIS = ["33011", "33013", "33009", "33005", "33001", "33003", "33019", "33007"]
WOR = ["25027"]
before = {"Boston (bos)": (BOS, pack["bos"]), "Manchester, NH (his)": (HIS, pack["his"]), "Worcester (wor)": (WOR, pack["wor"])}
mv = POP["33015"] + POP["33017"]
after = {"Boston (bos)": (BOS[:5], pack["bos"] - mv), "Manchester, NH (his)": (HIS + BOS[5:], pack["his"] + mv),
         "Worcester (wor)": (WOR, pack["wor"])}
col = {"Boston (bos)": "#fb8072", "Manchester, NH (his)": "#80b1d3", "Worcester (wor)": "#d9d9d9"}
view = (-72.7, 41.2, -69.8, 45.4)
fig, axes = plt.subplots(1, 2, figsize=(14, 10))
dens = dry.density(view)
for ax, rows, title in ((axes[0], before, "now"), (axes[1], after, "proposed")):
    if dens is not None:
        ax.imshow(dens, extent=(view[0], view[2], view[1], view[3]), origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.6, zorder=2, interpolation="nearest", aspect="auto")
    for g, s in geo.items():
        for p in getattr(s, "geoms", [s]):
            ax.fill(*p.exterior.xy, color="#f2f2f2", ec="#bbb", lw=0.4, zorder=0)
    for name, (cs, ppl) in rows.items():
        u = shapely.union_all([geo[c] for c in cs])
        m = wp.to_moll(u)
        main = max(getattr(m, "geoms", [m]), key=lambda q: q.area)
        h = main.area / main.convex_hull.area
        hot = name != "Worcester (wor)"
        for p in getattr(u, "geoms", [u]):
            ax.fill(*p.exterior.xy, color=col[name], lw=0, zorder=1)
            ax.plot(*p.exterior.xy, color="#222" if hot else "#888", lw=1.6 if hot else 0.4, zorder=4 if hot else 3)
        pt = shapely.ops.polylabel(max(getattr(u, "geoms", [u]), key=lambda q: q.area), 0.002)
        ax.text(pt.x, pt.y, f"{name}\n{ppl / 1e6:.2f}M" + (f"\nhull {h:.2f}" if hot else ""), ha="center", va="center",
                fontsize=8 if hot else 6, weight="bold" if hot else "normal", zorder=5)
        print(title, name, f"{ppl:,.0f}", f"hull {h:.2f}")
    ax.set_xlim(view[0], view[2]); ax.set_ylim(view[1], view[3])
    ax.set_aspect(1 / math.cos(math.radians(43.3)))
    ax.set_title(f"Boston: New Hampshire's counties to New Hampshire's zone: {title}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.text(0.5, 0.01, "Counties (TIGER 2020). People: the packs' figures; Rockingham and Strafford by the 2020 census. Bold outlines change; hull of the main piece. Grey: GHS-POP 2025 people a km2, log scale over the denser part of this view.",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(sys.argv[1], dpi=100)
print("->", sys.argv[1])
