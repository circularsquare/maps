"""T-3495: for the drawn passenger features t3495-snap-survey.mjs saved, the share of sample points with no
router graph chain within 25 m, by place and kind. Run from the maps root:
    venv\\Scripts\\python.exe -I trainworld/checks/t3495-graph-compare.py survey.json [graph dir] [all]
Reads the graphs the page's router reads (trainworld/data/packs/realgraph/)."""
import json, sys
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import realgraph

ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "data" / "packs" / "realgraph"
ALL = len(sys.argv) > 3 and sys.argv[3] == "all"  # the All setting's features instead of the snap setting's
idx = json.loads((ROOT / "index.json").read_text())["regions"]
M = 111320.0
data = json.load(open(sys.argv[1]))
graphs = {}
def lines_near(cx, cy):
    out = []
    for cc in idx:
        g = graphs.get(cc)
        if g is None:
            g = graphs[cc] = realgraph.read(ROOT / f"{cc}.rgr")
        lo, la = g["lon"], g["lat"]
        if lo.min() - 0.3 > cx or lo.max() + 0.3 < cx or la.min() - 0.3 > cy or la.max() + 0.3 < cy:
            continue
        for i in range(g["a"].size):
            a, b = g["off"][i], g["off"][i + 1]
            if abs(lo[a:b].mean() - cx) < 0.5 and abs(la[a:b].mean() - cy) < 0.4 and b - a > 1:
                out.append(LineString(np.column_stack([lo[a:b], la[a:b]])))
    return out
for place, d in data.items():
    pts = [(k["key"], p) for k in d["allFeats" if ALL else "feats"] for p in k["pts"]]
    if not pts:
        print(place, "no drawn passenger features"); continue
    cx, cy = np.mean([p[0] for _, p in pts]), np.mean([p[1] for _, p in pts])
    ls = lines_near(cx, cy)
    tree = STRtree(ls)
    from shapely.geometry import Point
    miss = {}
    for key, p in pts:
        pt = Point(p)
        j = tree.nearest(pt)
        dm = ls[j].distance(pt) * M * np.cos(np.radians(p[1])) if ls else 1e9
        m = miss.setdefault(key, [0, 0, []])
        m[1] += 1
        if dm > 25:
            m[0] += 1
            if len(m[2]) < 3: m[2].append([round(p[0], 5), round(p[1], 5), round(dm)])
    print(place, len(ls), "chains;", "; ".join(f"{k}: {m[0]}/{m[1]}" + (f" e.g. {m[2]}" if m[0] else "") for k, m in sorted(miss.items())))
