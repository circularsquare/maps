"""Audit helper1m units against GHSL 2025 (1 km, independent of our sources).

Per country and level: our 2025 estimate (linear from the last two years, as the
viewer does) vs the GHSL sum inside each polygon, GHSL scaled so the level's
totals agree. Writes audit_<country>.csv and prints summaries:
  - units far off GHSL (people misplaced), and neighbour pairs off in opposite
    directions (the Bengaluru signature: a boundary in the wrong place)
  - overlaps between units of one level, units with no population
  - children vs parent totals

Usage: C:\Python39\python.exe helper1m/scripts/audit_ghsl.py pakistan iran ...
CSVs go to helper1m/data/audit/.

GHSL is a suspect list, not a verdict: it has its own boundary errors (2026-11-02:
it puts Perm's million in the Permsky district ring, Butwal's people in Rohini,
1.8M in rural Sarvestan beside Shiraz; ours were right in all three, checked by
which unit holds the real city centre). A flag counts when a second signal agrees:
the census's own area for the unit, a density no city has, a city centre outside
its unit, or growth far from the parent's.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.windows import from_bounds

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HELPER = Path(__file__).resolve().parents[1] / "countries"
GHSL = (Path(__file__).resolve().parents[2] / "trainworld" / "data" / "raw" / "worldplan"
        / "GHS_POP_E2025_GLOBE_R2023A_54009_1000_V1_0.tif")
OUT = Path(__file__).resolve().parents[1] / "data" / "audit"
MOLL = "ESRI:54009"


def load_level(country, lvl):
    p = HELPER / country / f"adm{lvl}.geojson"
    files = [p] if p.exists() else sorted((HELPER / country / f"adm{lvl}").glob("*.geojson"))
    if not files:
        return None
    g = pd.concat([gpd.read_file(f) for f in files], ignore_index=True)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
    g["geometry"] = shapely.make_valid(g.geometry.values)
    return g


def est(pops, year=2025):
    if not isinstance(pops, dict):
        pops = json.loads(pops) if isinstance(pops, str) else {}
    ys = sorted((int(k), v) for k, v in pops.items() if v is not None)
    if not ys:
        return np.nan
    if len(ys) == 1:
        return ys[0][1]
    (y0, p0), (y1, p1) = ys[-2], ys[-1]
    return p1 + (p1 - p0) * (year - y1) / (y1 - y0)


def ghsl_sums(g):
    m = g.to_crs(MOLL)
    with rasterio.open(GHSL) as src:
        b = m.total_bounds
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win, boundless=True, fill_value=0).astype("float64")
        tr = src.window_transform(win)
    arr[arr < 0] = 0
    ids = rasterize(((geom, i + 1) for i, geom in enumerate(m.geometry) if geom is not None),
                    out_shape=arr.shape, transform=tr, fill=0, dtype="int32")
    s = np.bincount(ids.ravel(), weights=arr.ravel(), minlength=len(m) + 1)[1:]
    return s, m.area.values / 1e6


def audit(country):
    rows = []
    levels = {}
    for lvl in range(1, 6):
        g = load_level(country, lvl)
        if g is None:
            continue
        levels[lvl] = g
        g = g.copy()
        g["ours"] = [est(p) for p in g["populations"]]
        s, area = ghsl_sums(g)
        tot = np.nansum(g.ours)
        g["ghsl"] = s * tot / s.sum() if s.sum() else s
        g["area"] = area
        g["level"] = lvl
        g["ratio"] = g.ours / g.ghsl.replace(0, np.nan)
        g["off"] = g.ours - g.ghsl
        print(f"\n== {country} adm{lvl}: {len(g)} units, ours {tot/1e6:.1f}M, GHSL raw {s.sum()/1e6:.1f}M")
        nopop = g[g.ours.isna()]
        if len(nopop):
            print(f"   no population: {len(nopop)}: {list(nopop.name)[:8]}")
        # overlaps
        m = g.to_crs(MOLL)
        m["geometry"] = shapely.make_valid(m.geometry.values)
        tree = shapely.STRtree(m.geometry.values)
        pairs = tree.query(m.geometry.values, predicate="intersects")
        ov = []
        for a, bb in zip(*pairs):
            if a < bb:
                x = m.geometry.iloc[a].intersection(m.geometry.iloc[bb]).area / 1e6
                if x > 2:
                    ov.append((x, g.name.iloc[a], g.name.iloc[bb]))
        if ov:
            ov.sort(reverse=True)
            print(f"   overlaps >2 km2: {len(ov)}; biggest: " +
                  "; ".join(f"{n1} / {n2} {x:.0f} km2" for x, n1, n2 in ov[:5]))
        # far off GHSL
        big = g[(np.maximum(g.ours, g.ghsl) > 100_000) & ((g.ratio > 2) | (g.ratio < 0.5))]
        big = big.reindex(big.off.abs().sort_values(ascending=False).index)
        for _, r in big.head(15).iterrows():
            print(f"   {r['name'][:60]:60s} ours {r.ours/1e3:8.0f}k  ghsl {r.ghsl/1e3:8.0f}k  x{r.ratio:5.2f}  {r.area:7.0f} km2")
        if len(big) > 15:
            print(f"   ... {len(big)} units over the bar in all")
        # opposite-direction neighbours (boundary in the wrong place)
        hi = set(g.index[(g.ratio > 1.6) & (g.off > 50_000)])
        lo = set(g.index[(g.ratio < 0.6) & (g.off < -50_000)])
        pr = []
        for a, bb in zip(*pairs):
            if a in hi and bb in lo:
                pr.append((min(g.off[a], -g.off[bb]), g.name[a], g.name[bb]))
        if pr:
            pr.sort(reverse=True)
            print("   high/low neighbour pairs: " +
                  "; ".join(f"{h} (high) / {l} (low) ~{x/1e3:.0f}k" for x, h, l in pr[:8]))
        rows.append(g.drop(columns="geometry"))
    # nesting
    for lvl, g in levels.items():
        if lvl - 1 not in levels or "parent_code" not in g:
            continue
        par = levels[lvl - 1]
        kids = g.assign(ours=[est(p) for p in g["populations"]]).groupby("parent_code").ours.sum()
        pp = pd.Series([est(p) for p in par["populations"]], index=par.code)
        d = (kids.reindex(pp.index) - pp).abs()
        bad = d[d > 1000]
        orphan = set(g.parent_code) - set(par.code)
        if len(bad) or orphan:
            print(f"   nesting adm{lvl} -> adm{lvl-1}: {len(bad)} parents off by >1000 "
                  f"(max {d.max():,.0f}); orphan parent codes {len(orphan)}")
    if rows:
        OUT.mkdir(parents=True, exist_ok=True)
        pd.concat(rows).to_csv(OUT / f"audit_{country}.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    for c in sys.argv[1:]:
        audit(c)
