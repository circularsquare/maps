"""The microstate tier — nine countries built from UNSD table 28 and Kontur, no office visited.

    python sources/micro.py --fetch      nine Kontur extracts, ~230 KB in total
    python sources/micro.py              normalise all nine + build their placement layers
    python sources/micro.py pw ck        just those

Writes, per country:
    data/normalized/<cc>.csv        one row per religion, geo_level `country`
    data/geo/<cc>/<cc>_hexes.gpkg   Kontur 400 m population hexagons, unit = <cc>

**WHY NINE COUNTRIES SHARE ONE MODULE.** Every other source here is bespoke because offices
agree about nothing. These nine agree about everything: they come from **one instrument**,
the UNSD Demographic Yearbook's religion table, in **one shape**, a national partition with
no geography, for countries small enough that the whole map is 20 to 80 dots. A per-country
module would be nine copies of forty lines.

**THE PERMISSION THAT MAKES THEM BUILDABLE IS ANITA'S, 2026-09-08**: *"for the really small
island countries we might not even need any divisions. like for instance if we do palau, it
has 17000 people so itll just be 17 dots."* At 1 dot = 1,000 people the placement inside a
country of 20,000 carries no claim at all, so a national-only source is complete rather than
coarse. That retires spec §3.9b's unit-count floor and §3.9c's variety floor for this tier,
which is what queue.md had parked these behind.

**AND THE ORACLE HAD THE DATA ALL ALONG.** sources.md §11r documented a column set that
returns an index with the values stripped, so the table was only ever asked *"has this office
tabulated religion at all"*. `tools/oracle.py` has the set that carries the counts; 33 of the
37 countries then in queue.md turned out to be in it. These nine are the ones where it is a
SOURCE rather than a check.

**THE PARTITION IS CHECKED PER COUNTRY AND ONE OF THEM FAILS.** Eight of the nine sum to
their own stated total to the person. **Antigua and Barbuda's 21 categories sum to 76,889
against a stated 76,886**, three people over, which is in the Yearbook rather than in this
code. It is 0.004% and the country draws 77 dots either way; `EXPECTED` records the tolerance
per country so the discrepancy is declared rather than absorbed silently.

**WHAT THIS TIER MAY NOT BE USED FOR.** Every row is national. Nothing here says where inside
a country anyone lives, and the Kontur layer places dots by population only. Bermuda is the
one country in the tier where the DYB also publishes urban and rural, and even that is not
read: a two-way split is not a geography.

**STALENESS IS REAL AND IS DECLARED PER COUNTRY.** The census years run 1999 (Marshall
Islands) to 2017 (Tuvalu, Niue). Four are 2001 or older. `how=` on each country entry says
which year, per spec §7c, and the Marshall Islands' four categories are the thinnest table
drawn anywhere on this map.
"""

import csv
import gzip
import os
import shutil
import ssl
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

RAW = os.path.join(ROOT, "data", "raw", "micro")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
              "kontur_population_{CC}_20231101.gpkg.gz")
KONTUR_VINTAGE = "2023-11-01"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# cc -> (UNSD country name, census year, stated total, categories, partition slack allowed)
#
# The year is pinned rather than taken as "the latest": the oracle carries several censuses
# for most of these and a silent move to a new one would change every count on the map
# without changing a line of code. `check()` asserts the total, so a re-publication that
# revises a figure fails the build.
COUNTRIES = {
    "pw": ("Palau",               2005, 19_907,  9, 0),
    "ck": ("Cook Islands",        2011, 14_974,  9, 0),
    "tv": ("Tuvalu",              2017, 10_507, 11, 0),
    "nu": ("Niue",                2017,  1_591,  8, 0),
    "ms": ("Montserrat",          2001,  4_303, 11, 0),
    "bm": ("Bermuda",             2010, 64_237, 23, 0),
    "ag": ("Antigua and Barbuda", 2001, 76_886, 21, 3),   # see the docstring: DYB is +3
    "dm": ("Dominica",            2001, 68_635, 14, 0),
    "mh": ("Marshall Islands",    1999, 50_848,  4, 0),
}

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _kontur_paths(cc):
    gz = os.path.join(RAW, f"kontur_population_{cc.upper()}_20231101.gpkg.gz")
    return gz, gz[:-3]


def fetch(ccs):
    os.makedirs(RAW, exist_ok=True)
    import oracle
    oracle.fetch()
    for cc in ccs:
        gz, gpkg = _kontur_paths(cc)
        if os.path.exists(gpkg) and os.path.getsize(gpkg) > 0:
            print(f"  have {os.path.basename(gpkg)}")
            continue
        if not os.path.exists(gz):
            url = KONTUR_URL.format(CC=cc.upper())
            body = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                          timeout=180, context=_ctx()).read()
            tmp = gz + ".part"
            with open(tmp, "wb") as fh:
                fh.write(body)
            os.replace(tmp, gz)
            print(f"  got {os.path.basename(gz)} ({len(body):,} bytes)")
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)


def normalise(cc):
    """Oracle -> data/normalized/<cc>.csv, with the partition asserted."""
    import oracle

    name, year, total, ncats, slack = COUNTRIES[cc]
    got = oracle.oracle(name, year)
    if not got:
        raise SystemExit(f"{cc}: {name} {year} is not in the oracle — "
                         "run `python tools/oracle.py --fetch`")
    counts = got.get(oracle.TOTAL)
    if not counts:
        raise SystemExit(f"{cc}: no `Total` area for {name} {year}")

    cats, stated, exact = oracle.partition(counts)
    if stated != total:
        raise SystemExit(f"{cc}: the oracle now says {stated:,} for {name} {year}, "
                         f"this build expects {total:,}")
    if len(cats) != ncats:
        raise SystemExit(f"{cc}: {len(cats)} categories, expected {ncats}")
    drift = sum(cats.values()) - stated
    if abs(drift) > slack:
        raise SystemExit(f"{cc}: categories sum to {sum(cats.values()):,} against a stated "
                         f"{stated:,}, a difference of {drift:+,} and only {slack} allowed")
    flag = "" if drift == 0 else f"  [DYB is {drift:+,} against its own total]"
    print(f"  {cc}  {name:<22}{year}  {len(cats):>3} cats  {stated:>8,}{flag}")

    rows = [dict(geo_id=cc.upper(), geo_level="country", geo_name=name,
                 source_category=cat, count=n, basis="self_id", year=year,
                 source_id=f"unsd_dyb28_{cc}_{year}",
                 note=f"UNSD Demographic Yearbook table 28, {name} {year}, area Total")
            for cat, n in sorted(cats.items(), key=lambda kv: -kv[1])]

    out = os.path.join(ROOT, "data", "normalized", f"{cc}.csv")
    tmp = out + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    return stated


def geometry(cc, census_total):
    """Kontur hexes -> data/geo/<cc>/<cc>_hexes.gpkg, every hex on the one unit."""
    import geopandas as gpd

    _gz, gpkg = _kontur_paths(cc)
    if not os.path.exists(gpkg):
        raise SystemExit(f"{cc}: missing {gpkg} — run with --fetch")
    hexes = gpd.read_file(gpkg)
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"{cc}: no population column in {list(hexes.columns)}")
    hexes = hexes.rename(columns={popcol: "pop"})
    hexes = hexes[hexes["pop"] > 0].copy()
    if hexes.empty:
        raise SystemExit(f"{cc}: Kontur has no populated hexes")
    # ONE UNIT. The whole country is the geography the counts are on, so there is no join to
    # get wrong here and §12's first two shapes of failure cannot arise anywhere in this tier.
    hexes["unit"] = cc.upper()
    hexes["cellcode"] = cc.upper() + ":" + hexes.index.astype(str)
    hexes = hexes.to_crs("EPSG:4326")

    geo = os.path.join(ROOT, "data", "geo", cc)
    os.makedirs(geo, exist_ok=True)
    out = os.path.join(geo, f"{cc}_hexes.gpkg")
    hexes[["cellcode", "unit", "pop", "geometry"]].to_file(out, layer="hexes", driver="GPKG")

    k = float(hexes["pop"].sum())
    ratio = k / census_total
    # Kontur is a MODEL of the present population and the censuses here run 1999-2017, so a
    # wide band is expected and only a wild one means the wrong country's file.
    flag = "" if 0.5 <= ratio <= 2.0 else "   <-- CHECK"
    print(f"      {len(hexes):>6,} hexes, Kontur {k:>9,.0f} vs census {census_total:>9,} "
          f"({KONTUR_VINTAGE}, ratio {ratio:.2f}){flag}")
    return ratio


def main():
    args = [a.lower() for a in sys.argv[1:] if not a.startswith("--")]
    ccs = args or list(COUNTRIES)
    bad = [c for c in ccs if c not in COUNTRIES]
    if bad:
        raise SystemExit(f"not in this tier: {bad}; have {sorted(COUNTRIES)}")
    if "--fetch" in sys.argv:
        fetch(ccs)
    print(f"  normalising {len(ccs)} countries")
    for cc in ccs:
        total = normalise(cc)
        geometry(cc, total)


if __name__ == "__main__":
    main()
