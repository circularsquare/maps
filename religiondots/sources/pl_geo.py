"""Poland — gmina boundaries for the NSP 2021 religion data.

Writes data/geo/pl/pl_gminy.gpkg, the single layer countries.py reads, and prints the
join against data/normalized/pl.csv both ways.

SOURCE: Eurostat GISCO, LAU 2021, 1:1,000,000, EPSG:4326.  One 98MB zip covers 34
European countries, so it is also the boundary source for Slovakia, Hungary and Romania
if those ever land (sources.md §11).  Chosen over geoBoundaries POL ADM3, which is 2017
vintage — exactly the spec §8.1 hazard, since Polish gminy have been created and merged
since — and over GUGiK's PRG, which is authoritative but ships as a much larger national
download with no European reuse.

The GISCO vintage is 2021, the same year as the census, and it matches the census unit
for unit: 2,477 both sides, nothing unmatched either way.

THE JOIN IS NOT THE OBVIOUS ONE.  GISCO gives Poland a 13-digit LAU_ID and GUS gives a
7-digit TERYT code, and neither contains the other:

    LAU_ID   1006061110802
    TERYT       0608022

The LAU_ID embeds TERYT at fixed offsets and DROPS THE TYPE DIGIT — the trailing 1/2/3
that distinguishes an urban gmina from a rural one from a mixed one:

    voivodeship = LAU_ID[4:6]     powiat = LAU_ID[9:11]     gmina = LAU_ID[11:13]

so `LAU_ID[4:6] + LAU_ID[9:11] + LAU_ID[11:13]` reproduces TERYT's first six digits, and
those six are already unique per gmina.  Joining on the ids as delivered matches ZERO of
2,477 rows — and because both sides have exactly 2,477 units, a unit-count check passes
while the join fails completely, which is the §5c shape of mistake.

The independent evidence that the offsets are right rather than lucky is the names: 2,476
of 2,477 joined pairs agree on the gmina name.  The single disagreement is real and is not
an error — GUS calls 260417 `gm. w. Nowiny` and GISCO calls it `Sitkówka-Nowiny`, which is
the name it had before the 2021 rename.

PLACEMENT (spec §8.2): gminy are the placement layer as well as the count layer, because
GUS publishes religion at no finer unit.  Median gmina population is about 7,500, which is
finer than a Mexican municipio and about twice a US census tract, so an equal share per
polygon is a reasonable population weighting nearly everywhere.

WHERE IT IS NOT REASONABLE, and this is Czechia's Prague problem one size smaller:
Warszawa is a single gmina holding 1.79M people, 4.7% of the country, spread over 517 km².
Kraków, Łódź, Wrocław and Poznań are each one gmina too.  Czechia had a fix — ČSÚ
publishes the 142 city districts — and GUS does not: TABL.7 is gminy and stops.  Warsaw's
18 dzielnice exist as boundaries but there are no religion counts for them, so subdividing
would be an allocation inventing structure the source does not have.  Left as one polygon
and recorded here.

Usage:
    python sources/pl_geo.py            # needs the GISCO zip, see sources/pl_geo.md
    python sources/pl_geo.py --fetch    # download it first (98MB)
"""

import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GEO = os.path.join(ROOT, "data", "geo", "lau2021")
OUTER_ZIP = os.path.join(GEO, "ref-lau-2021-01m.shp.zip")
INNER_NAME = "LAU_RG_01M_2021_4326.shp.zip"
SHP_DIR = os.path.join(GEO, "shp4326")

OUT_DIR = os.path.join(ROOT, "data", "geo", "pl")
OUT = os.path.join(OUT_DIR, "pl_gminy.gpkg")

NORMALIZED = os.path.join(ROOT, "data", "normalized", "pl.csv")

LAU_URL = ("https://gisco-services.ec.europa.eu/distribution/v2/lau/download/"
           "ref-lau-2021-01m.shp.zip")
MIN_BYTES = 90_000_000

EXPECTED_UNITS = 2477


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(GEO, exist_ok=True)
    if os.path.exists(OUTER_ZIP) and os.path.getsize(OUTER_ZIP) >= MIN_BYTES:
        print("already have", OUTER_ZIP)
        return
    print("downloading", LAU_URL, "(98MB)")
    r = requests.get(LAU_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=900,
                     verify=False, stream=True)
    r.raise_for_status()
    with open(OUTER_ZIP, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    size = os.path.getsize(OUTER_ZIP)
    if size < MIN_BYTES:
        raise SystemExit(f"{OUTER_ZIP} is {size:,} bytes, expected >= {MIN_BYTES:,}")
    print(f"  {size:,} bytes")


def unpack():
    """GISCO ships a zip of zips — one per projection. Take the 4326 one."""
    if not os.path.exists(OUTER_ZIP):
        raise SystemExit(f"missing {OUTER_ZIP} -- run with --fetch first")
    inner = os.path.join(GEO, INNER_NAME)
    if not os.path.exists(inner):
        with zipfile.ZipFile(OUTER_ZIP) as z:
            z.extract(INNER_NAME, GEO)
        print("extracted", INNER_NAME)
    if not os.path.isdir(SHP_DIR):
        with zipfile.ZipFile(inner) as z:
            z.extractall(SHP_DIR)
        print("extracted", SHP_DIR)
    shps = [f for f in os.listdir(SHP_DIR) if f.endswith(".shp")]
    if not shps:
        raise SystemExit(f"no .shp under {SHP_DIR}")
    return os.path.join(SHP_DIR, shps[0])


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    shp = unpack()

    print("reading", shp)
    gdf = gpd.read_file(shp)
    print(f"  {len(gdf):,} LAU features across {gdf['CNTR_CODE'].nunique()} countries")

    pl = gdf[gdf["CNTR_CODE"] == "PL"].copy()
    pl["LAU_ID"] = pl["LAU_ID"].astype(str).str.strip()
    if len(pl) != EXPECTED_UNITS:
        raise SystemExit(f"got {len(pl)} PL features, expected {EXPECTED_UNITS} -- "
                         "GISCO reissued LAU 2021")

    # The offsets are asserted rather than discovered, so an id that stops being 13 digits
    # fails here rather than producing a silently wrong six-digit key.
    bad_len = pl[pl["LAU_ID"].str.len() != 13]
    if len(bad_len):
        raise SystemExit(f"{len(bad_len)} PL LAU_IDs are not 13 digits, e.g. "
                         f"{list(bad_len['LAU_ID'][:5])} -- the slice rule no longer holds")

    pl["kod"] = (pl["LAU_ID"].str[4:6] + pl["LAU_ID"].str[9:11] + pl["LAU_ID"].str[11:13])
    if pl["kod"].nunique() != len(pl):
        raise SystemExit("derived TERYT-6 keys are not unique")

    out = pl[["kod", "LAU_ID", "LAU_NAME", "POP_2021", "AREA_KM2", "geometry"]].rename(
        columns={"LAU_ID": "lau_id", "LAU_NAME": "name",
                 "POP_2021": "pop_2021", "AREA_KM2": "area_km2"})

    # ---- the join, both ways, against the data this layer exists to carry
    df = pd.read_csv(NORMALIZED, dtype={"geo_id": str}, low_memory=False)
    cen = df[df["geo_level"] == "gmina"][["geo_id", "geo_name"]].drop_duplicates("geo_id")
    cen["kod"] = cen["geo_id"].str[:6]

    geo_keys, cen_keys = set(out["kod"]), set(cen["kod"])
    missing_geo = cen_keys - geo_keys
    extra_geo = geo_keys - cen_keys
    print(f"\n  census gminy {len(cen_keys):,}  |  polygons {len(geo_keys):,}")
    print(f"  {'OK ' if not missing_geo else 'BAD'} census units with no polygon: "
          f"{len(missing_geo)}")
    print(f"  {'OK ' if not extra_geo else 'BAD'} polygons with no census unit : "
          f"{len(extra_geo)}")
    for k in sorted(missing_geo)[:10]:
        print("      no polygon:", k)
    for k in sorted(extra_geo)[:10]:
        print("      no data   :", k)
    if missing_geo or extra_geo:
        raise SystemExit("join FAILED")

    # Names are the independent check that the offset rule is right and not a coincidence
    # that happens to produce 2,477 unique keys.
    import re
    def bare(n):
        return re.sub(r"^gm\.\s*(m-w\.|m\.|w\.)\s*", "", str(n)).strip().casefold()

    merged = cen.merge(out[["kod", "name"]], on="kod")
    agree = (merged["geo_name"].map(bare) == merged["name"].map(
        lambda s: str(s).strip().casefold()))
    print(f"\n  name agreement on joined pairs: {agree.sum():,} of {len(merged):,}")
    for _, r in merged[~agree].iterrows():
        print(f"      {r['kod']}  GUS={r['geo_name']!r}  GISCO={r['name']!r}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_file(OUT, layer="gminy", driver="GPKG")
    print("\nwrote", OUT, f"({len(out):,} polygons)")

    big = out.nlargest(5, "pop_2021")
    tot = out["pop_2021"].sum()
    print("\n  largest single polygons (the spec §8.2 caveat for Poland):")
    for _, r in big.iterrows():
        print(f"    {r['name']:<14} {r['pop_2021']:>10,.0f}  {r['area_km2']:>7.1f} km2  "
              f"{100 * r['pop_2021'] / tot:5.2f}% of PL in one unit")


if __name__ == "__main__":
    main()
