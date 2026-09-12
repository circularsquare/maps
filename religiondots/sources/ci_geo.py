"""Côte d'Ivoire — région boundaries and the placement grid.

Writes:
    data/geo/ci/ci_regions.gpkg     the 33 counted units (`units`)
    data/geo/ci/ci_hexes.gpkg       Kontur H3 hexes with `unit` and `pop` (`place`)
    data/geo/ci/ci_lookup.csv       unit -> census population, Kontur population

Usage:
    python sources/ci_geo.py --fetch    one 11.7 MB gzipped gpkg from Kontur
    python sources/ci_geo.py            rebuild from data/raw/ci/

THE BOUNDARIES ARE geoBoundaries AND THE COUNT MATCHES EXACTLY. `gbOpen/CIV/ADM2` is **33
polygons** against the census's 33 units — 31 régions plus the two autonomous districts —
and ADM1 is the 14 districts, which is the tier this map deliberately does not draw. That is
a better start than most: §9m had to rebuild the Philippines' BARMM, §9r disqualified
geoBoundaries CHN outright, §9ah found it missing an entire Korean county. Here it fits.

**THE JOIN IS BY NAME AND THREE OF THE 33 DO NOT FOLD.** Accent- and punctuation-folding
matches 30. The other three are:

    District D'Abidjan          <->  District Autonome D'Abidjan
    District De Yamoussoukro    <->  District Autonome De Yamoussoukro
    La Mé                       <->  Me

None is ambiguous — there is exactly one Abidjan, one Yamoussoukro and one Mé, and no other
candidate is within edit distance of any of them — so an explicit alias table is safe here in
a way it was not for Mauritius (§9af, where 183 French names needed a SPATIAL check because
two of them genuinely split the same ground differently). The aliases are asserted to be
exhaustive: any unmatched name at all raises.

KONTUR IS NEEDED, AND FOR THE NORTHERN SAVANNAH. Côte d'Ivoire is 322,463 km² over 33 units,
and the split is very uneven: the autonomous district of Abidjan is 2,153 km² holding 6.32
million people (2,936/km²) while Bounkani is 21,800 km² holding 427,037 (19.6/km²). An equal
share per polygon would spread the north's dots evenly over country that is mostly empty
Comoé National Park, and Bounkani is the one région where `Animiste` is 24.7% — so the
category this map is most interested in showing is precisely the one a uniform fill would
smear across the largest and emptiest polygon in the country.

AND THE §9av CHECK IS RUN HERE TOO. The Central African Republic (§9av) turned out to have a
Kontur extract that is close to a flat rescale of the census being drawn, because CAR has had
no census since. Côte d'Ivoire's census is 2021 and Kontur is 2023, so the two are two years
apart and Kontur was NOT built on this table; the per-unit spread should therefore be wide.
`main()` measures it rather than assuming, and prints the CAR comparison.
"""

import csv
import gzip
import json
import os
import re
import shutil
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ci")
GEO = os.path.join(ROOT, "data", "geo", "ci")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "ci.csv")

GB_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/"
          "gbOpen/CIV/ADM2/geoBoundaries-CIV-ADM2.geojson")
GB = os.path.join(RAW, "geoBoundaries-CIV-ADM2.geojson")

OUT_UNITS = os.path.join(GEO, "ci_regions.gpkg")
OUT_HEXES = os.path.join(GEO, "ci_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "ci_lookup.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_CI_20231101.gpkg.gz")
GZ_NAME = "kontur_population_CI_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_CI_20231101.gpkg"

UNITS = 33

# The three that do not fold. Census name -> geoBoundaries shapeName, both folded.
ALIASES = {
    "districtdabidjan": "districtautonomedabidjan",
    "districtdeyamoussoukro": "districtautonomedeyamoussoukro",
    "lame": "me",
}

# The census is 2021 and Kontur is 2023 — two years, so a ratio near 1.1 is expected.
KONTUR_RATIO_MIN = 0.85
KONTUR_RATIO_MAX = 1.45


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0"}

    if not os.path.exists(GB) or os.path.getsize(GB) < 500_000:
        r = requests.get(GB_URL, timeout=600, headers=ua)
        r.raise_for_status()
        # §5a: a 200 is not a download. GitHub serves an HTML error page happily.
        if r.content.lstrip()[:1] != b"{":
            raise SystemExit(f"{GB}: starts {r.content[:16]!r}, expected JSON")
        with open(GB, "wb") as fh:
            fh.write(r.content)
        print(f"wrote {GB} ({len(r.content):,} bytes)")

    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 20_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz) or os.path.getsize(gz) < 8_000_000:
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers=ua)
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    for p in (GB, NORM):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run sources/ci.py --fetch and "
                             f"sources/ci_geo.py --fetch")
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run: python sources/ci_geo.py --fetch")
    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the counted units, from the normalised file
    df = pd.read_csv(NORM, dtype={"geo_id": str})
    census = df.groupby("geo_name")["count"].sum()
    if len(census) != UNITS:
        raise SystemExit(f"{len(census)} units in ci.csv, expected {UNITS}")

    g = gpd.read_file(GB)
    if len(g) != UNITS:
        raise SystemExit(f"geoBoundaries ADM2 has {len(g)} features, "
                         f"expected {UNITS}")

    g["key"] = g["shapeName"].map(norm)
    lut = {}
    for name in census.index:
        k = ALIASES.get(norm(name), norm(name))
        hits = g.index[g["key"] == k].tolist()
        if len(hits) != 1:
            raise SystemExit(
                f"census unit {name!r} (folded {k!r}) matched {len(hits)} "
                f"polygons — the alias table is not exhaustive")
        lut[name] = hits[0]
    used = sorted(lut.values())
    if len(set(used)) != UNITS:
        raise SystemExit("two census units matched the same polygon")
    print(f"name join: {UNITS}/{UNITS} matched, "
          f"{UNITS - len(ALIASES)} directly and {len(ALIASES)} through the "
          f"alias table, 0 polygons unused")

    units = g.loc[used].copy()
    inv = {v: k for k, v in lut.items()}
    units["unit"] = [inv[i] for i in used]
    units["census_pop"] = units["unit"].map(census).astype(int)
    units[["unit", "shapeName", "census_pop", "geometry"]].to_file(
        OUT_UNITS, layer="regions", driver="GPKG")
    print(f"wrote {OUT_UNITS} ({len(units)} régions)")

    eq = units.to_crs(3857)
    km2 = eq.geometry.area / 1e6
    order = km2.sort_values(ascending=False).index
    top10 = order[:10]
    print(f"\n  the 10 largest régions are "
          f"{100 * km2[top10].sum() / km2.sum():.1f}% of the land and "
          f"{100 * units.loc[top10, 'census_pop'].sum() / units['census_pop'].sum():.1f}% "
          f"of the people — the reason a population grid is used.")

    # ---- 2. Kontur, joined on hex CENTROIDS
    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"\nKontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left",
                       predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"  hexes whose centroid is in no région: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%) — border "
          f"overrun\n     between Kontur's extract and geoBoundaries; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"régions with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"régions whose hexes sum to zero population: {zero}")
    print(f"  every one of the {UNITS} régions has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    # §8.2e: the grid must be finer than the tier it weights.
    print(f"  the smallest région is {km2.min():,.0f} km² against a 0.67 km² hex; "
          f"the grid is\n     far finer than the tier everywhere "
          f"(min hexes per région {per['size'].min():,}).")

    tot = float(out["pop"].sum())
    census_total = int(units["census_pop"].sum())
    ratio = tot / census_total
    print(f"\n  Kontur 2023 {tot:,.0f} vs the 2021 drawn population "
          f"{census_total:,} — ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(
            f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, "
            f"{KONTUR_RATIO_MAX}] — check the download")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")

    # ---- 3. the lookup, and §9av's independence check
    per = per.rename(columns={"size": "hexes", "sum": "kontur_pop"})
    lk = units[["unit", "census_pop"]].merge(
        per, left_on="unit", right_index=True, how="left")
    lk["kontur_over_census"] = lk["kontur_pop"] / lk["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "census_pop_2021", "kontur_pop_2023", "hexes",
                    "kontur_over_census"])
        for r in lk.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, int(r.census_pop), round(r.kontur_pop, 1),
                        int(r.hexes), round(r.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")

    q = lk["kontur_over_census"].quantile([.05, .25, .5, .75, .95])
    print(f"\n  per-région Kontur/census ratio: median {q[.5]:.2f}, "
          f"quartiles {q[.25]:.2f}–{q[.75]:.2f}, 5–95% {q[.05]:.2f}–{q[.95]:.2f}")

    # §9av: is Kontur independent of this census, or downstream of it?
    near = float(((lk["kontur_over_census"] / q[.5] - 1).abs() <= 0.05).mean())
    print(f"  {100 * near:.1f}% of régions are within ±5% of the median ratio "
          f"(CAR: 78.5%, Ethiopia: 34.0%)")
    if near > 0.60:
        print("     TOO TIGHT TO BE INDEPENDENT — see sources.md §9av. Côte d'Ivoire's "
              "census\n     is 2021 and Kontur is 2023, so this was NOT expected; "
              "investigate before\n     treating the ratio as any kind of check.")
    else:
        print("     Wide enough to be independent modelling, which is what a 2021 census "
              "against\n     a 2023 grid should look like — unlike CAR (§9av), whose "
              "grid had no newer\n     census to be built from and is a rescale of the "
              "table being drawn.")

    print("\n  the 6 régions where Kontur and the census disagree most:")
    for r in lk.reindex(lk["kontur_over_census"].sort_values(
            ascending=False).index).head(6).itertuples(index=False):
        print(f"     {r.kontur_over_census:6.2f}x  {str(r.unit)[:30]:30s} "
              f"census {int(r.census_pop):>10,}")


if __name__ == "__main__":
    main()
