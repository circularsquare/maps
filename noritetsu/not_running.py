"""Register sections with no track on the map: lines the register still lists that no train runs.

    (called from build_model.main, after drop_unridden_sections)

WHY.  A register is a legal list, and it lags the railway.  N02-24 still has the 日田彦山線
from 添田 to 夜明, a BRT since August 2023, and the 留萌線 to 深川, closed in 2026; the 美祢線,
肥薩線 八代-吉松 and the far end of the 津軽線 are suspended after floods.  OpenStreetMap has
taken the rails off all of them, so the base map drew nothing there while the line's panel
listed stations, which looked like a bug (Anita, 2026-09-30).  She chose to keep such lines,
greyed as not running, rather than drop them: a rider who went before the closure wants to
record it.

THE TEST.  A section is not running when under half its length has drawn track within
BESIDE_M of it: any drawn track, whatever line it belongs to, since the question is only whether
there are rails there at all.  "Drawn" is build_tiles' own filter, so the answer matches what
the map shows.  On Japan every section that fails is a real closure or suspension, plus the
Nagoya guideway bus (legally a railway, a busway in OSM): 154 km in 45 sections on 6 lines.
Too near other track to be caught: 広島電鉄's old 猿猴橋町-的場町, rerouted in 2025, 0.4 km.

Each such line gets `closed`: the "a|b" keys of those sections.  The app draws them grey and
dashed, and leaves them out of completion.
"""
import math

import numpy as np

BESIDE_M = 150.0
# Registers whose geometry is close enough to the rails for the test to mean anything. N02 is
# (a section of the 青梅線 in the 奥多摩 tunnels is 120 m off, hence BESIDE_M). So is the Swiss
# network register: where it fails, OSM has no rail of any tag within 350 m to 1 km, as on
# Basel tram 11 at Reinach, closed for rebuilding and mapped railway=construction, which
# extract.py does not keep. Korea's and Taiwan's registers are OSM track, so they cannot fail.
# SNCF Réseau's RFN geometry (fr_register) is surveyed to about 10 m, so the test holds there.
SOURCES = {"n02", "schienennetz", "rfn"}
STEP_M = 100.0
MIN_SHARE = 0.5


def mark(region, lines, geoms, log):
    from shapely import STRtree, line_interpolate_point
    from shapely.geometry import LineString
    import build_tiles as bt

    ways, _stops, on_route, cid, cx, cy = bt.load(region, lambda m: None)
    feats = bt.geometries(ways, on_route, cid, cx, cy, lambda m: None)
    tree = STRtree([LineString(f["xy"]) for f in feats])
    earth = 40075016.0

    n_sec, km_sec, hit = 0, 0.0, []
    for l in lines:
        l.pop("closed", None)
        if l.get("suspended"):
            # The register reader knows no passenger train runs on the whole line
            # (rinf.py `suspended`, Slovakia): every section is not running.
            l["closed"] = [f"{s[0]}|{s[1]}" for s in l["sections"]]
            n_sec += len(l["closed"])
            km_sec += sum(s[2] for s in l["sections"])
            hit.append((l["name"], l.get("operator", ""), len(l["closed"]), len(l["sections"])))
            continue
        if l.get("src", "osm") not in SOURCES:
            continue
        g = geoms.get(l["id"], {})
        gone = []
        for sec in l["sections"]:
            key = f"{sec[0]}|{sec[1]}"
            pts = g.get(key)
            if not pts or len(pts) < 2:
                continue
            a = np.asarray(pts, dtype=np.float64)
            x, y = bt.merc(a[:, 0], a[:, 1])
            line = LineString(np.column_stack([x, y]))
            unit = 1.0 / (earth * math.cos(math.radians(float(a[:, 1].mean()))))
            n = max(2, int(sec[2] * 1000 / STEP_M))
            probe = line_interpolate_point(line, (np.arange(n) + 0.5) / n, normalized=True)
            near = tree.query_nearest(probe, max_distance=BESIDE_M * unit, all_matches=False)
            share = len(set(near[0].tolist())) / n
            if share < MIN_SHARE:
                gone.append(key)
                km_sec += sec[2]
        if gone:
            l["closed"] = gone
            n_sec += len(gone)
            hit.append((l["name"], l.get("operator", ""), len(gone), len(l["sections"])))
    log(f"not running: {n_sec} register sections ({km_sec:,.0f} km) on {len(hit)} lines have "
        f"no drawn track beside them")
    for name, op, k, total in sorted(hit):
        log(f"    {name} [{op}] {k} of {total} sections")
