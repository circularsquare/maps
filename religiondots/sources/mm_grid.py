"""Myanmar — the placement layer: Kontur 400 m population hexagons, keyed to State/Region.

Writes data/geo/mm/mm_hexes.gpkg.

15 units for 51.5M people is ~3.4M each — **the second coarsest counting geography on this
map after Zimbabwe's ten provinces** — and they are wildly uneven: Yangon is 7.36M in 9,867
km², Kachin 1.64M in 88,978 km², Chin 479k in 36,018 km² of mountain. An equal share per
polygon would wash the empty north in evenly spaced dots and squash a seventh of the country
into one speck, which is §8.2's Kenya case again.

**AND THE RATIO BAND HERE MEASURES SOMETHING REAL, WHICH IS NOT WHAT IT IS NORMALLY FOR.**
Kontur's grid is 2023 and the census is 2014, so the national ratio reads 1.050 and fourteen
of the fifteen states sit between **0.80 and 1.15** — a tight band for a nine-year gap.

**Rakhine reads 0.58, and that is the expulsion.** Against the enumerated count alone it is
0.89, in line with everywhere else; against the count *including* the 1,090,000 non-enumerated
it is 0.58, because roughly three quarters of a million Rohingya fled to Bangladesh in 2017
and Kontur's 2023 surface does not contain them. **The diagnostic that normally checks a join
is, for this one state, measuring the event the country is being drawn to show.**

**WHAT THAT COSTS THE MAP, STATED PLAINLY.** Placement inside Rakhine is weighted by where
people lived in 2023, and the non-enumerated of 2014 lived disproportionately in the northern
townships — Maungdaw, Buthidaung, Rathedaung — which are precisely the places the 2023 surface
finds emptiest. So Rakhine's `unenumerated` dots sit south of where those people actually
were. Nothing here can fix that: no source publishes the non-enumerated below state level, a
uniform spread would put them in the Arakan mountains instead, and inventing a northern
concentration would be §14.4. **The map is drawn at state level and its own note says to read
it as composition and never as location; this is the sharpest case of why.**

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two states.

Usage:
    python sources/mm_grid.py --fetch    one 15.3 MB gzipped gpkg from Kontur
    python sources/mm_grid.py            rebuild from data/raw/mm/
"""

import gzip
import os
import re
import shutil
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mm")
GEO = os.path.join(ROOT, "data", "geo", "mm")
STATES = os.path.join(GEO, "mm_states.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "mm.csv")
LOOKUP = os.path.join(GEO, "mm_lookup.csv")
BOUNDARIES = os.path.join(RAW, "mmr_admin_boundaries.shp.zip")
OUT = os.path.join(GEO, "mm_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_MM_20231101.gpkg.gz")
GZ_NAME = "kontur_population_MM_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_MM_20231101.gpkg"

EXPECTED_STATES = 15
UNIVERSE = 51_486_253          # enumerated 50,279,900 + non-enumerated 1,206,353

# A 2023 modelled grid against a 2014 census: Myanmar grew from ~51.5M to ~54M over the
# period, so the national ratio should read a little above 1.0.
NATIONAL_TOLERANCE = 0.30

# Measured, not chosen: fourteen states fall in 0.80-1.15 and Rakhine reads 0.58 for the
# reason in the module docstring. 1.8 admits all fifteen with real headroom, and the shuffle
# control below shows it still discriminates hard at that width.
UNIT_BAND = 1.8


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 20_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import math
    import random

    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(STATES):
        raise SystemExit(f"missing {STATES} -- run sources/mm_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    st = gpd.read_file(STATES)
    if len(st) != EXPECTED_STATES:
        raise SystemExit(f"{STATES} has {len(st)} states, expected {EXPECTED_STATES}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(st.crs)
    hexes = hexes.to_crs(st.crs)

    joined = gpd.sjoin(pts, st[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every state: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")

    # **IS ANY OF THAT A HOLE IN THE DISSOLVE RATHER THAN BORDER OVERRUN?** Myanmar has
    # 5,700 km of land border and Kontur's extract crosses it, which is ordinary. What would
    # NOT be ordinary is a hex inside the country and inside no state — that would mean
    # sources/mm_geo.py's fifteen do not cover Myanmar, and it would empty real places with
    # no error anywhere. ADM0 is the independent test.
    names = [i.filename for i in zipfile.ZipFile(BOUNDARIES).infolist()]
    shp0 = [n for n in names if re.search(r"adm(?:in)?0\.shp$", n, re.I)][0]
    adm0 = gpd.read_file(f"zip://{BOUNDARIES}!{shp0}", engine="fiona").to_crs(st.crs)
    in0 = gpd.sjoin(pts[outside.to_numpy()], adm0[["geometry"]], how="inner",
                    predicate="within")
    hole = float(in0[popcol].sum()) if len(in0) else 0.0
    print(f"  {'OK ' if len(in0) == 0 else 'BAD'} of those, inside Myanmar's ADM0 outline: "
          f"{len(in0):,} hexes ({hole:,.0f} people)")
    print("     A hex inside the country and inside no state would mean the 15 dissolved "
          "polygons\n     do not cover Myanmar. The rest is ordinary border overrun into "
          "Bangladesh, India,\n     China, Thailand and Laos; dropped.")
    if len(in0):
        raise SystemExit(f"{len(in0)} populated hexes are inside Myanmar and inside no "
                         "state -- sources/mm_geo.py's dissolve has a hole")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=st.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(st["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"states with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"states whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_STATES} states has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / UNIVERSE
    print(f"\n  Kontur {tot:,.0f} vs the drawn universe {UNIVERSE:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")
    print("     a 2023 modelled grid against a 2014 census count, nine years apart, so a "
          "ratio a\n     little above 1.0 is what this should read.")

    # ---- per state, against BOTH denominators, because the difference is the finding ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(LOOKUP, dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    enum, nonenum = {}, {}
    for gid, sub in df[df["geo_level"] == "state_region"].groupby("geo_id"):
        unit = unit_of[gid]
        enum[unit] = int(sub.loc[sub["source_category"] == "Total", "count"].iloc[0])
        m = sub["source_category"] == "Estimated Non-enumerated population"
        nonenum[unit] = int(sub.loc[m, "count"].iloc[0]) if m.any() else 0

    name_of = dict(zip(st["unit"], st["name"]))
    rows = []
    for u in enum:
        drawn = enum[u] + nonenum[u]
        k = float(per.loc[u, "sum"])
        rows.append((u, name_of[u], enum[u], nonenum[u], drawn, k,
                     k / drawn / ratio, k / enum[u] / ratio))
    rows.sort(key=lambda r: r[6])
    print("\n  per state, Kontur/census normalised by the national ratio — against the "
          "DRAWN\n  universe and against the ENUMERATED count, because for one state they "
          "differ:")
    print(f"    {'':<14} {'drawn':>10} {'kontur':>10} {'vs drawn':>9} {'vs enum':>8}")
    for u, nm, e, ne, d, k, r1, r2 in rows:
        flag = "   <-- the expulsion" if ne and abs(r1 - r2) > 0.15 else ""
        print(f"    {nm:<14} {d:>10,} {k:>10,.0f} {r1:>9.2f} {r2:>8.2f}{flag}")
    print("\n     Rakhine is 0.58 against the universe it is drawn on and 0.89 against the "
          "enumerated\n     count alone — in line with every other state. The gap is the "
          "~750,000 Rohingya who\n     fled to Bangladesh in 2017 and are not in a 2023 "
          "population grid. See the docstring\n     for what it costs the placement, which "
          "is real and is not fixable from any source.")

    worst = [r for r in rows if r[6] < 1 / UNIT_BAND or r[6] > UNIT_BAND]
    if worst:
        raise SystemExit(f"{len(worst)} states outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[6], 2)) for w in worst]}")

    # ---- both nulls, measured (§12 — Benin against Zimbabwe) ----
    rng = random.Random(0)
    ks = [r[5] for r in rows]
    ds = [r[4] for r in rows]
    fails = []
    for _ in range(2000):
        sh = list(ks)
        rng.shuffle(sh)
        fails.append(sum(1 for d, k2 in zip(ds, sh)
                         if not (1 / UNIT_BAND <= k2 / d / ratio <= UNIT_BAND)))
    med = sorted(fails)[len(fails) // 2]
    clean = sum(1 for f in fails if f == 0)
    print(f"\n  BAND control: shuffling the populations across the polygons puts a median "
          f"{med} of\n  {len(rows)} states outside the factor-of-{UNIT_BAND:g} band, and "
          f"{clean} of 2,000 shuffles pass cleanly.")
    if med < 4:
        raise SystemExit(f"a shuffled join puts only a median {med} states outside the "
                         "band -- it no longer discriminates and must not be reported as "
                         "a check")

    def pearson(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return num / den

    lc = [math.log(d) for d in ds]
    lk = [math.log(k) for k in ks]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(2000):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(abs(pearson(lc, sh)))
    perm.sort()
    beat = sum(1 for x in perm if x >= r_true)
    print(f"  CORRELATION control: r = {r_true:.4f} against a best of {perm[-1]:.4f} over "
          f"2,000 shuffles\n  ({beat} reach it).")
    if beat > 100 or r_true < 0.80:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 2,000 random pairings reach -- the join in "
                         "mm_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
