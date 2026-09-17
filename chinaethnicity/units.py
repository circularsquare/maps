"""Roll the county counts up to admin units, for the viewer's pie charts and hover.

The dot layer answers "who lives here"; this answers "what is this county". Each unit gets
its nationality breakdown, its people-weighted centre, and a Latin name (names.py).

The breakdown is the same arithmetic the dots come from, not a count of the dots: a census
row covers one or more county polygons (join.py's `adcodes`), and scatter.py splits it
between them by ASPECT population, or by the explicit shares join.py sets for a development
zone. This repeats that split and stops at the county instead of going on to the cells, so
a county's pie and the dots inside it are the same numbers and the national totals match
the census exactly.

Three levels, because 2,848 county pies are a mesh at low zoom: county, prefecture (the
four municipalities count as one prefecture each) and province. Each unit sits at the mean
position of its people, not the middle of its polygon, so a pie in Gansu sits on the Hexi
corridor rather than in the desert north of it.

Writes:
    data/processed/units.json   three levels of unit, each with g/k breakdown arrays

Usage:
    python units.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")

import json
import sys
import time

import numpy as np
import pandas as pd

import names as name_source
from common import GEO, KEYS, PROC, PROVINCES, RELIGIONDOTS, WORK

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LEAVES = os.path.join(WORK, "leaves_2020.csv")
FALLBACK = os.path.join(WORK, "leaves_fallback.csv")
CELLS = os.path.join(GEO, "cells.npz")
INDEX = os.path.join(RELIGIONDOTS, "data", "raw", "cn", "datav", "county_index.json")
OUT = os.path.join(PROC, "units.json")

# The four municipalities have no prefecture tier: DataV files every district under itself,
# which would put 38 "prefectures" in Chongqing. They roll up as one unit.
MUNICIPALITIES = {"110000", "120000", "310000", "500000"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def county_shares(leaves, cells_of, cty, wgt, is_fb):
    """(leaf, county, share) for every census row: the split scatter.py gives the dots."""
    def spread(idx, what):
        real = idx[~is_fb[idx]]
        if len(real) and wgt[real].sum() > 0:
            return real, wgt[real]
        idx = idx[is_fb[idx]]
        if not len(idx):
            raise SystemExit(f"row {what} has no cells at all")
        return idx, np.ones(len(idx))

    out = []
    for li, (spec, nm) in enumerate(zip(leaves["adcodes"], leaves["name"])):
        parts = [p.split(":") for p in spec.split(";")]
        if all(len(p) == 2 for p in parts):
            # join.py set each county's share of a development zone explicitly, printed to
            # four decimals, so a row's shares can miss 1.0 by a few 1e-4.
            s = [float(p[1]) for p in parts]
            for (code, _), share in zip(parts, s):
                out.append((li, code, share / sum(s)))
        else:
            idx, w = spread(np.concatenate([cells_of(p[0]) for p in parts]), nm)
            w = w / w.sum()
            c = cty[idx]
            o = np.argsort(c, kind="stable")
            c, w = c[o], w[o]
            edge = np.flatnonzero(np.diff(c)) + 1
            for a, b in zip(np.r_[0, edge], np.r_[edge, len(c)]):
                out.append((li, int(c[a]), float(w[a:b].sum())))
    return out


def main():
    with open(INDEX, encoding="utf-8") as fh:
        index = json.load(fh)
    idx_by_code = {str(r["code"]): r for r in index}
    pairs = dict([(str(r["city_code"]), r["city"]) for r in index]
                 + [(str(r["code"]), r["name"]) for r in index])
    labels, from_wd = name_source.resolve(pairs.items())
    log(f"{len(labels):,} unit names, {from_wd:,} from Wikidata, "
        f"{len(labels) - from_wd:,} romanised")

    leaves = pd.read_csv(LEAVES, dtype={"prov": str, "adcodes": str})
    measured = set(leaves["prov"])
    if os.path.exists(FALLBACK):
        fb = pd.read_csv(FALLBACK, dtype={"prov": str, "adcodes": str})
        leaves = pd.concat([leaves, fb[~fb["prov"].isin(measured)]], ignore_index=True)
    else:
        log("no leaves_fallback.csv: only the measured provinces get units")

    C = np.load(CELLS)
    n_pop = len(C["row"])
    row = np.concatenate([C["row"], C["fb_row"]]).astype(np.int64)
    col = np.concatenate([C["col"], C["fb_col"]]).astype(np.int64)
    wgt = np.concatenate([C["pop"].astype(np.float64), np.zeros(len(C["fb_row"]))])
    cty = np.concatenate([C["county"], C["fb_county"]]).astype(np.int64)
    is_fb = np.arange(len(row)) >= n_pop
    adcodes = C["adcodes"].astype(str)
    code_idx = {a: i for i, a in enumerate(adcodes)}
    order = np.argsort(cty, kind="stable")
    bounds = np.searchsorted(cty[order], np.arange(len(adcodes) + 1))

    def cells_of(code):
        i = code_idx[code]
        return order[bounds[i]:bounds[i + 1]]

    a, _, c0, _, e, f = C["transform"]
    lon = c0 + (col + 0.5) * a
    lat = f + (row + 0.5) * e

    # People-weighted centre of each county, from the ASPECT cells inside it.
    cx = np.zeros(len(adcodes))
    cy = np.zeros(len(adcodes))
    for i in range(len(adcodes)):
        sl = order[bounds[i]:bounds[i + 1]]
        real = sl[~is_fb[sl]]
        if len(real) and wgt[real].sum() > 0:
            w = wgt[real]
        else:
            real, w = sl, np.ones(len(sl))
        if not len(real):
            raise SystemExit(f"county {adcodes[i]} has no cells")
        cx[i] = float((lon[real] * w).sum() / w.sum())
        cy[i] = float((lat[real] * w).sum() / w.sum())

    counts = leaves[KEYS].to_numpy(dtype=np.float64)
    county = np.zeros((len(adcodes), len(KEYS)))
    seen = np.zeros(len(leaves))
    for li, c, s in county_shares(leaves, cells_of, cty, wgt, is_fb):
        ci = code_idx[c] if isinstance(c, str) else c
        county[ci] += counts[li] * s
        seen[li] += s
    bad = np.abs(seen - 1) > 1e-6
    if bad.any():
        first = leaves["name"].iloc[int(np.flatnonzero(bad)[0])]
        raise SystemExit(f"{int(bad.sum())} census rows do not split into whole counties, "
                         f"first: {first}")
    gap = abs(county.sum() - counts.sum())
    if gap > 1:
        raise SystemExit(f"county totals are {gap:,.0f} off the census rows")
    log(f"{len(leaves):,} census rows -> {int((county.sum(axis=1) > 0).sum()):,} "
        f"counties with people, {county.sum():,.0f} people")

    drawn = set(leaves["prov"])
    rounded = np.rint(county).astype(np.int64)

    def record(code, en, cn, parent, vec, x, y, estimated):
        g = np.flatnonzero(vec > 0)
        g = g[np.argsort(-vec[g], kind="stable")]
        rec = {"c": code, "n": en, "z": cn, "t": int(vec.sum()),
               "x": round(float(x), 4), "y": round(float(y), 4),
               "g": [int(i) for i in g], "k": [int(vec[i]) for i in g]}
        if parent:
            rec["in"] = parent
        if estimated:
            rec["e"] = 1
        return rec

    def label(code, chinese):
        return labels.get(code) or name_source.romanise(chinese)

    counties, roll_pref, roll_prov = [], {}, {}
    for i, code in enumerate(adcodes):
        meta = idx_by_code[code]
        pcode = code[:2]
        if pcode not in drawn or rounded[i].sum() == 0:
            continue
        prov_code = str(meta["prov_code"])
        city_code = prov_code if prov_code in MUNICIPALITIES else str(meta["city_code"])
        est = pcode not in measured
        parent = (PROVINCES[pcode][1] if city_code in (prov_code, code)
                  else f"{label(city_code, meta['city'])}, {PROVINCES[pcode][1]}")
        counties.append(record(code, label(code, meta["name"]), meta["name"], parent,
                               rounded[i], cx[i], cy[i], est))
        for bucket, key, cn in ((roll_pref, city_code, meta["city"]),
                                (roll_prov, pcode, PROVINCES[pcode][2])):
            b = bucket.setdefault(key, {"v": np.zeros(len(KEYS), dtype=np.int64), "cn": cn,
                                        "prov": pcode, "est": est, "wx": 0.0, "wy": 0.0,
                                        "w": 0.0})
            b["v"] += rounded[i]
            w = float(rounded[i].sum())
            b["wx"] += cx[i] * w
            b["wy"] += cy[i] * w
            b["w"] += w

    def rolled(bucket, parented):
        out = []
        for key, b in bucket.items():
            if not parented:
                en, parent = PROVINCES[key][1], None
            else:
                en = (PROVINCES[b["prov"]][1] if key.endswith("0000")
                      else label(key, b["cn"]))
                parent = PROVINCES[b["prov"]][1]
                if parent == en:
                    parent = None
            out.append(record(key, en, b["cn"], parent, b["v"], b["wx"] / b["w"],
                              b["wy"] / b["w"], b["est"]))
        return sorted(out, key=lambda r: -r["t"])

    levels = {
        "province": rolled(roll_prov, False),
        "prefecture": rolled(roll_pref, True),
        "county": sorted(counties, key=lambda r: -r["t"]),
    }
    os.makedirs(PROC, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"year": 2020, "levels": levels}, fh, ensure_ascii=False,
                  separators=(",", ":"))
    for k, v in levels.items():
        log(f"  {k:<11s} {len(v):>5,} units, {sum(u['t'] for u in v):>15,} people")
    log(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
