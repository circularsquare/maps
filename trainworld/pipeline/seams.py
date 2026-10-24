"""T-1192 (notes/T-1178.md): zone lines that run through one continuous city, and the commute factor
across each, written to `borders.json`'s `seamless` list for the joint gravity.

    python pipeline/seams.py          the list, written into pipeline/borders.json
    python pipeline/seams.py --dry    printed only

A pair of registered packs of one country is seamless when at least `MIN_KM` of the line between them
runs through city: pairs of edge-neighbouring GHS-POP 2025 1 km pixels, one in each pack, both of
`URBAN` people or more and in one urban centre (pixels of `URBAN`+ joined edge to edge, clusters of
`CENTRE_MIN`+, GHSL's degree of urbanisation without its gap filling); each pixel pair is about 1 km
of line. The factor eases from the line's ordinary one (the country's admin-1 default, else
`pack_line_default`) at `SIDE_FLOOR` people of the shared centre on the smaller side up to 1 at
`SIDE_FULL`, linearly. National lines are left out. Rerun after a replan or a change of the default.
Read-only on data; about 3 minutes on one core.
"""
import collections
import json
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

import world_plan as wp  # noqa: E402

BORDERS = wp.ROOT / "pipeline/borders.json"
URBAN = 1500.0          # people a km2 (GHSL's urban centre)
CENTRE_MIN = 50000.0    # people of an urban centre
MIN_KM = 10             # km of line through one urban centre
SIDE_FLOOR = 50000.0    # the smaller side's people of that centre where the lift starts
SIDE_FULL = 200000.0    # ... and where the factor reaches 1


def boundary(pid):
    gj = json.loads((wp.PACKS / f"{pid}.boundary.geojson").read_text(encoding="utf-8"))
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    parts = [shapely.make_valid(shapely.geometry.shape(f.get("geometry", f))) for f in feats]
    parts = [q for g in parts for q in shapely.get_parts(g) if q.geom_type in ("Polygon", "MultiPolygon")]
    return wp.to_moll(shapely.union_all(parts) if len(parts) > 1 else parts[0])


def urban_pixels(reg, pop):
    """(pixel key, people, pack index) of every URBAN+ pixel inside a pack, the first pack winning."""
    keys, ppl, pk, meta = [], [], [], []
    for c in reg:
        pid = c["id"]
        if not (wp.PACKS / f"{pid}.boundary.geojson").exists():
            continue
        h = json.loads((wp.PACKS / f"{pid}.json").read_text(encoding="utf-8"))
        g = boundary(pid)
        x, y, p = pop.pixels(g.bounds)
        m = p >= URBAN
        x, y, p = x[m], y[m], p[m]
        shapely.prepare(g)
        m = shapely.contains_xy(g, x, y)
        col = np.round((x[m] - pop.X0) / pop.PIX - 0.5).astype(np.int64)
        row = np.round((pop.Y0 - y[m]) / pop.PIX - 0.5).astype(np.int64)
        keys.append(row * pop.W + col)
        ppl.append(p[m])
        pk.append(np.full(int(m.sum()), len(meta), np.int32))
        meta.append({"id": pid, "country": h.get("country") or c.get("country"), "regions": (h.get("regions") or {}).get("ids", [])})
    key, ppl, pk = np.concatenate(keys), np.concatenate(ppl), np.concatenate(pk)
    o = np.argsort(key, kind="stable")
    key, ppl, pk = key[o], ppl[o], pk[o]
    keep = np.r_[True, key[1:] != key[:-1]]
    return key[keep], ppl[keep], pk[keep], meta


def seams(key, ppl, pk, W):
    """Per pack pair: km of line through one urban centre, and that centre's people on each side."""
    n = len(key)
    ea, eb = [], []
    for off in (1, W):
        j = np.minimum(np.searchsorted(key, key + off), n - 1)
        ok = key[j] == key + off
        if off == 1:
            ok &= (key % W) != W - 1
        ea.append(np.nonzero(ok)[0])
        eb.append(j[ok])
    ea, eb = np.concatenate(ea), np.concatenate(eb)
    _, lab = connected_components(coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(n, n)), directed=False)
    cpop = np.bincount(lab, weights=ppl)
    side = collections.defaultdict(float)
    for l_, k, v in zip(lab.tolist(), pk.tolist(), ppl.tolist()):
        side[(l_, k)] += v
    out = collections.defaultdict(lambda: {"km": 0, "side": 0.0})
    x = (pk[ea] != pk[eb]) & (cpop[lab[ea]] >= CENTRE_MIN)
    for a, b, l_ in zip(pk[ea][x].tolist(), pk[eb][x].tolist(), lab[ea][x].tolist()):
        s = out[(min(a, b), max(a, b))]
        s["km"] += 1
        s["side"] = max(s["side"], min(side[(l_, a)], side[(l_, b)]))
    return out


def main():
    dry = "--dry" in sys.argv
    borders = json.loads(BORDERS.read_text(encoding="utf-8"))
    reg = json.loads(wp.REGISTRY.read_text(encoding="utf-8"))["cities"]
    pop = wp.Pop()
    key, ppl, pk, meta = urban_pixels(reg, pop)
    wp.log(f"{len(key):,} urban pixels in {len(meta):,} packs")
    pairs = []
    for (a, b), s in seams(key, ppl, pk, pop.W).items():
        A, B = meta[a], meta[b]
        if A["country"] != B["country"] or s["km"] < MIN_KM or s["side"] <= SIDE_FLOOR:
            continue
        # a pack holding several regions shares them with other packs, so a pack pair is no region pair
        assert A["regions"] == ["00"] and B["regions"] == ["00"], f"{A['id']}-{B['id']}: not tier 0 single-region packs"
        base = borders["admin1_default"].get(A["country"], borders["pack_line_default"])
        t = min(1.0, (s["side"] - SIDE_FLOOR) / (SIDE_FULL - SIDE_FLOOR))
        ids = sorted((A["id"], B["id"]))
        pairs.append({"a": ids[0], "b": ids[1], "factor": round(base + t * (1 - base), 3), "km": s["km"], "side": int(round(s["side"], -3))})
    pairs.sort(key=lambda p: (-p["factor"], -p["km"], p["a"]))
    for p in pairs:
        wp.log(f"  {p['a']}-{p['b']}: {p['km']} km of line through city, {p['side'] / 1e6:.2f}M on the smaller side, factor {p['factor']}")
    wp.log(f"{len(pairs)} seamless pairs, {sum(p['factor'] == 1 for p in pairs)} at 1")
    if dry:
        return
    borders["seamless"] = pairs
    # one key a line, one pair a line, as the file was written by hand
    lines = []
    for k, v in borders.items():
        if k == "seamless":
            body = ",\n".join("    " + json.dumps(p, ensure_ascii=False) for p in v)
            lines.append(f'  "seamless": [\n{body}\n  ]')
        else:
            lines.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}")
    tmp = BORDERS.with_suffix(".json.tmp")
    tmp.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")
    os.replace(tmp, BORDERS)
    wp.log(f"wrote {BORDERS}")


if __name__ == "__main__":
    main()
