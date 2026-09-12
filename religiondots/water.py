"""
Take the sea out of the placement polygons, so no dot is drawn on water — spec §8.2.

THE PROBLEM. Placement polygons are administrative units, and administrative units own
water. A US census tract on the Manhattan shore reaches into the middle of the Hudson; the
tract is where the people are counted and the river is inside it, so a dot placed uniformly
in that tract has a real chance of landing mid-river. Measured before this existed: **370 of
the 12,399 US dots in the New York bbox, 3.0%, were in open water**, and they are the most
visible dots on the map because nothing else is drawn there. San Francisco Bay was 1.4%,
Puget Sound 0.4%.

It is not a counting error — every dot is still in the right unit and every total is
untouched. It is a placement error, and §8.2 is exactly the section that says placement is
allowed to use more information than the count does.

THE FIX IS TO THE POLYGON, NOT THE DOT. The obvious alternative is rejection sampling:
place a dot, test it against water, try again if it is wet. That pays the test for every
dot forever, and `tools/scan_congregations.py` timings already have point-in-polygon as the
hot path of a scatter. Subtracting water from the polygon once is paid per BUILD instead of
per dot, is cached, and makes sampling slightly *faster* afterwards because the polygon it
samples is smaller and every candidate point is a hit.

WHAT COUNTS AS WATER. OpenStreetMap's `water-polygons-split-4326` — the global ocean and
tidal-water layer derived from `natural=coastline`, already in the repo at
`../data/water-polygons-split-4326/` for other maps. It is the right layer for this problem
rather than a lucky one: OSM runs the coastline up an estuary to the tidal limit, so the
Hudson, the East River, the Thames and the Río de la Plata are all in it, and those are the
cases that produce visible dots in a city.

**IT IS NOT INLAND WATER.** Lakes and non-tidal rivers are a different OSM layer and are not
subtracted here. In practice that matters much less than it sounds, because the statistical
agency has usually done it already — Lake Lanao is a hole in the Philippine barangays, and
the Great Lakes are absent from the US tract file. Where an agency has NOT done it the lake
will still take dots, and that is a known gap rather than a solved problem.

A UNIT THAT IS ENTIRELY WATER KEEPS ITS ORIGINAL SHAPE. Two US tracts are all water. Their
people are real and have to be drawn somewhere, and a polygon that clips to nothing would
either crash the sampler or silently drop them — §4.1 says the dot count is the one thing
that may not move. So the clip is skipped for them and reported.

**AND "ENTIRELY WATER" IS THE WRONG LINE, SO THE RULE IS A THRESHOLD (spec §8.2c-i).**
"Nobody lives on water" is false in the Sulu Archipelago. Ten populated Philippine barangays
lose more than 90% of their area to this clip: Port Holland Zone III in Basilan (4,904
people, 98.0% gone), Tungbangkaw in Tawi-Tawi and four more in Sulu are Sama-Bajau villages
built on stilts over water, with no land under them in OSM, so clipping crams them onto
whatever shore sliver survives. Three others — Nasingin, Batasan and Ubay Island in Bohol —
are real coral islets where clipping is exactly right, because it moves the dots off the sea
and onto the islet.

**Nothing in a boundary file separates those two cases**, and Anita's read (2026-09-05) is
that no single number ever will: the honest threshold is different in Manila and in Maine.
`KEEP_WHOLE_ABOVE` is that number anyway, at 0.95 — high enough that ordinary coastal units
still get clipped, low enough that a village standing over open water keeps the polygon it
actually occupies. It rescues six of the ten above and costs the three Bohol islets, which
is a known and accepted trade rather than a solved problem.

IT IS APPLIED WHEN THE CLIP IS READ, NOT WHEN IT IS COMPUTED, so the cache does not depend
on it. Moving the threshold is then free — no country is re-clipped — which is the property
you want on a number nobody can derive.

Usage: called from scatter.py. `--no-water` there turns it off and is the before/after.
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

HERE = Path(__file__).parent
WATER = HERE.parent / "data" / "water-polygons-split-4326" / "water_polygons.shp"
CACHE = HERE / "data" / "geo" / "_waterclip"

# A unit that loses MORE than this share of its area to the sea is left unclipped, on the
# grounds that a 2% sliver is not where 4,904 people live — see the stilt-village note above.
# Applied on read, so changing it re-clips nothing.
KEEP_WHOLE_ABOVE = 0.95


def _cache_paths(cc, src):
    CACHE.mkdir(parents=True, exist_ok=True)
    stem = f"{cc}_{Path(src).stem}"
    return CACHE / f"{stem}.gpkg", CACHE / f"{stem}.json"


def _stamp(src):
    return {"place": str(src), "place_mtime": os.path.getmtime(src),
            "water_mtime": os.path.getmtime(WATER) if WATER.exists() else None,
            "version": 1}


def _keep_whole(place, clipped, verbose):
    """Undo the clip wherever it removed more than KEEP_WHOLE_ABOVE of a unit.

    Read-time, and deliberately so: the cache holds the raw clip, so this threshold can move
    without re-clipping a single country. A unit that was entirely water was already left
    whole upstream, so it measures as 0% lost here and is not touched twice.
    """
    a0 = shapely.area(place.geometry.values)
    a1 = shapely.area(clipped)
    with np.errstate(divide="ignore", invalid="ignore"):
        lost = np.where(a0 > 0, 1.0 - a1 / np.where(a0 > 0, a0, 1.0), 0.0)
    back = lost > KEEP_WHOLE_ABOVE
    if back.any():
        clipped = clipped.copy()
        clipped[back] = place.geometry.values[back]
        if verbose:
            print(f"  water: {int(back.sum()):,} unit(s) lost over "
                  f"{KEEP_WHOLE_ABOVE * 100:.0f}% to the sea and are left UNCLIPPED — "
                  "stilt villages and the like (water.py)")
    return clipped


def clip(place, cc, src, verbose=True):
    """`place` clipped to land. Same rows, same order, same columns — geometry only.

    Row identity is load-bearing: scatter.py builds `by_unit` from positional indices and
    the place_weight hooks read columns off the same frame, so this must never add, drop or
    reorder a row. It replaces `geometry` in place and nothing else.
    """
    if not WATER.exists():
        if verbose:
            print(f"  !! no water layer at {WATER} — dots may land in the sea; "
                  "see water.py")
        return place

    cache, meta = _cache_paths(cc, src)
    want = _stamp(src)
    if cache.exists() and meta.exists():
        try:
            if json.loads(meta.read_text()) == want:
                g = gpd.read_file(cache)
                if len(g) == len(place):
                    if verbose:
                        print(f"  water: reusing {cache.name}")
                    out = place.copy()
                    out["geometry"] = _keep_whole(place, g.geometry.values, verbose)
                    return out
                if verbose:
                    print(f"  !! {cache.name} has {len(g):,} rows for {len(place):,} "
                          "placement polygons — rebuilding")
        except Exception as e:                       # a half-written cache is not fatal
            if verbose:
                print(f"  !! could not reuse {cache.name} ({e}) — rebuilding")

    t0 = time.time()
    g = place.geometry.values
    bounds = tuple(place.total_bounds)
    w = gpd.read_file(WATER, bbox=bounds)
    if not len(w):
        if verbose:
            print("  water: no ocean polygons over this country, nothing to clip")
        return place
    wg = w.geometry.values
    if w.crs is not None and w.crs.to_epsg() != 4326:
        raise SystemExit(f"water layer is {w.crs}, expected EPSG:4326")

    pairs = shapely.STRtree(wg).query(g, predicate="intersects")
    by = defaultdict(list)
    for a, b in zip(pairs[0], pairs[1]):
        by[int(a)].append(int(b))

    out = g.copy()
    n_clip = n_gone = n_fixed = 0
    for a, bs in by.items():
        x0, y0, x1, y1 = shapely.bounds(g[a])
        # Cut each ocean polygon down to this unit's own envelope FIRST. A rectangle clip is
        # cheap and an Atlantic polygon is millions of vertices; without it the difference
        # below runs 2x slower for exactly the same answer (measured on the US: 211s -> 107s,
        # identical output).
        parts = []
        for b in bs:
            try:
                c = shapely.clip_by_rect(wg[b], x0, y0, x1, y1)
            except Exception:                        # GEOSClipByRect dislikes some inputs
                c = shapely.intersection(wg[b], shapely.box(x0, y0, x1, y1))
            if shapely.is_empty(c):
                continue
            if not shapely.is_valid(c):              # clip_by_rect may return invalid rings
                c = shapely.make_valid(c)
                n_fixed += 1
            parts.append(c)
        if not parts:
            continue
        u = parts[0] if len(parts) == 1 else shapely.union_all(np.array(parts))
        d = shapely.difference(g[a], u)
        if shapely.is_empty(d) or shapely.area(d) <= 0:
            n_gone += 1                              # all water: keep it, §4.1
        else:
            out[a] = d
            n_clip += 1

    a0, a1 = float(shapely.area(g).sum()), float(shapely.area(out).sum())
    if verbose:
        print(f"  water: {n_clip:,} of {len(place):,} placement polygons clipped, "
              f"{(1 - a1 / a0) * 100:.2f}% of their area was sea"
              + (f", {n_gone:,} entirely water and left whole" if n_gone else "")
              + (f", {n_fixed:,} repaired" if n_fixed else "")
              + f"  [{time.time() - t0:.0f}s]")

    # Cache the clip WITHOUT the threshold, then apply the threshold to what we return. The
    # two must not be the same array, or the cache would bake in a number meant to stay
    # adjustable. (A unit that clipped to nothing is already the original here rather than an
    # empty geometry — an empty one cannot be stored or sampled — so it measures 0% lost on
    # read and the threshold leaves it alone. The two rules do not double up.)
    res = place.copy()
    res["geometry"] = _keep_whole(place, out, verbose)
    try:
        gpd.GeoDataFrame(geometry=out, crs=place.crs).to_file(cache, driver="GPKG")
        _cache_paths(cc, src)[1].write_text(json.dumps(want))
    except Exception as e:
        print(f"  !! could not write {cache.name} ({e}) — will re-clip next run")
    return res


if __name__ == "__main__":
    sys.exit("water.py is called from scatter.py; there is nothing to run here.")
