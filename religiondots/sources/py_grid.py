"""Paraguay — the placement grid: Kontur 400 m population hexagons, clipped to the districts.

Writes data/geo/py/py_hexes.gpkg and data/geo/py/py_kontur_by_unit.csv. `countries.py` uses
it to weight where a district's dots land, never to change how many there are.

**PARAGUAY NEEDS THIS MORE THAN A COMPACT COUNTRY WOULD, BECAUSE HALF OF IT IS EMPTY.** The
Chaco — departments 15, 16 and 17, Presidente Hayes, Boqueron and Alto Paraguay — is 60% of
the national territory and held about 2% of the people in 2002. Boqueron was a **single
district** covering 91,000 km² with 30,896 people aged 10 and over. An equal share of dots
per polygon would smear those across an area the size of Portugal, and it would put the
Mennonite colonies, which are the reason anyone looks at Boqueron on this map, in the wrong
place by 200 km. §8.2's emptiness case, at the largest scale it occurs anywhere here.

**THE VINTAGE GAP IS TWENTY-ONE YEARS**, counts 2002 and grid 2023, which is the widest on
this map (Nicaragua's eighteen was the previous). It moves dots *within* a district and never
between districts, so no count is affected, but it is a real weakness where a district has
grown unevenly since 2002 and it is stated in `countries.py`.

**AND THE CORRELATION IS THE ONLY INDEPENDENT CHECK THE JOIN GETS.** A modelled 2023
population grid built from building footprints shares no lineage with DGEEC's 2002 census or
with OCHA's boundary file, so it has to agree about how many people are in each of 224
districts. `check()` reports it against random re-pairings, and prints the per-district growth
ratio that `sources/py_geo.py`'s docstring uses to flag Pto. Pinasco.

Usage:
    python sources/py_grid.py --fetch    one ~6.8 MB gz from Kontur
    python sources/py_grid.py            rebuild from data/raw/py/
"""

import argparse
import csv
import gzip
import os
import random
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

import geopandas as gpd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "py")
GEO = os.path.join(ROOT, "data", "geo", "py")
NORM = os.path.join(ROOT, "data", "normalized", "py.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_PY_20231101.gpkg.gz")
GZ_NAME = "kontur_population_PY_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_PY_20231101.gpkg"

ASUNCION = [f"001{i}" for i in range(6)]
ASUNCION_CODE = "0000"


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings()
    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    if not (os.path.exists(gz) and os.path.getsize(gz) > 1_000_000):
        r = requests.get(GZ_URL, headers={"User-Agent": "religiondots"},
                         timeout=900, verify=False, stream=True)
        r.raise_for_status()
        with open(gz, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    print(f"  {os.path.getsize(gz):,} bytes  {gz}")
    out = os.path.join(RAW, GPKG_NAME)
    with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    print(f"  ungzipped -> {out}")


def build():
    src = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src}; run with --fetch")

    hexes = gpd.read_file(src)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    print(f"  hexes: {len(hexes):,}  crs={hexes.crs}  "
          f"population={hexes['population'].sum():,.0f}")

    units = gpd.read_file(os.path.join(GEO, "py_distritos.gpkg"), layer="distritos")
    units = units.to_crs(hexes.crs)

    pts = hexes.copy()
    pts["geometry"] = pts.geometry.centroid
    j = gpd.sjoin(pts, units[["code", "geometry"]], how="inner", predicate="within")
    print(f"  hexes inside a district: {len(j):,} of {len(hexes):,} "
          f"({j['population'].sum():,.0f} people)")

    hexes = hexes.loc[j.index].copy()
    hexes["unit"] = j["code"].values
    # `pop` is the column name scatter.py looks for; a layer without it falls back to
    # equal shares per polygon SILENTLY, with only a `!!` line to say so.
    hexes = hexes.rename(columns={"population": "pop"})
    hexes = hexes[["unit", "pop", "geometry"]].to_crs(4326)
    os.makedirs(GEO, exist_ok=True)
    out = os.path.join(GEO, "py_hexes.gpkg")
    hexes.to_file(out, driver="GPKG", layer="hexes")
    print(f"  wrote {out}")

    per = hexes.groupby("unit")["pop"].sum()
    with open(os.path.join(GEO, "py_kontur_by_unit.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "kontur_2023"])
        for c, v in per.items():
            w.writerow([c, int(round(v))])

    check(per)


def check(per):
    """Kontur 2023 against the census's own 2002 district sizes."""
    pop = {}
    with open(NORM, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = ASUNCION_CODE if row["geo_id"] in ASUNCION else row["geo_id"]
            pop[c] = pop.get(c, 0) + int(row["count"])

    common = sorted(set(pop) & set(per.index))
    a = [pop[c] for c in common]
    b = [float(per[c]) for c in common]
    n = len(common)
    print(f"  {n} districts matched; census 2002 (10+) {sum(a):,}, "
          f"kontur 2023 {sum(b):,.0f}, ratio {sum(b)/sum(a):.2f}")

    def corr(x, y):
        mx, my = sum(x) / len(x), sum(y) / len(y)
        sx = sum((v - mx) ** 2 for v in x) ** 0.5
        sy = sum((v - my) ** 2 for v in y) ** 0.5
        return sum((p - mx) * (q - my) for p, q in zip(x, y)) / (sx * sy)

    r = corr(a, b)
    rng = random.Random(20260908)
    best = 0.0
    for _ in range(500):
        sh = b[:]
        rng.shuffle(sh)
        best = max(best, abs(corr(a, sh)))
    print(f"  correlation r={r:.3f} against a best of {best:.3f} over 500 shuffles")
    if r < 0.80:
        raise SystemExit(f"kontur/census correlation {r:.3f} is too low -- the join "
                         "between census districts and polygons is suspect")

    ratios = sorted(((b[i] / a[i], common[i]) for i in range(n)), reverse=True)
    print("  highest growth ratios (a bad post-2002 parent would show here):")
    for ratio, c in ratios[:5]:
        print(f"    {c}  {ratio:6.1f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    build()


if __name__ == "__main__":
    main()
