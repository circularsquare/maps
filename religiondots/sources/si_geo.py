"""Slovenia — the drawn units: GISCO *Communes 2001*, which is the census's own občine.

Writes data/geo/si/si_units.gpkg.

**THE BOUNDARY VINTAGE IS FREE HERE, AND IT USUALLY IS NOT.** Slovenia had 192 municipalities
from 1998 until 2006, so the set the 2002 census enumerated stood unchanged for eight years;
it is 212 today. Every one of the twenty added since was carved out of an existing
municipality, so joining the census to a current boundary file would not lose territory, but
it would put twenty holes in the map and mis-seat the parents around them. Eurostat's GISCO
publishes a **Communes 2001** layer with exactly 192 Slovenian polygons, which is the
enumeration's own Gebietsstand, so no crosswalk is written and none is needed. The same file
is already on disk for Austria (sources/at_geo.py), which needed it for the same reason.

**THE JOIN IS ON THE OFFICIAL OBČINA CODE AND NOT ON A NAME.** GISCO carries it as
`NSI_CODE` (and, prefixed and padded, as `COMM_CODE`); sources/si.py recovers it for the
census table from a second table of the same census, because the religion table's own
`OBČINA` dimension is an alphabetical sequence and not the code. The two sets are asserted
equal, which is a stronger check than a count: 192 for 192 would also pass on a set that
disagreed in two places.

**GISCO'S NAMES ARE NOT USED FOR ANYTHING.** `SABE_NAME` is ASCII with the carons written as
a preceding `^` (`Ajdov^s^cina`, `Bre^zice`), so it is not a name any reader wants and not
one any join should touch. The names on the map come from SURS, through data/normalized.

Usage:
    python sources/si_geo.py --fetch    one 92 MB GISCO zip, shared with Austria
    python sources/si_geo.py            rebuild from data/geo/
"""

import os
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "6")

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO = os.path.join(ROOT, "data", "geo")
OUT_DIR = os.path.join(GEO, "si")
OUT = os.path.join(OUT_DIR, "si_units.gpkg")
CSV = os.path.join(ROOT, "data", "normalized", "si.csv")

COMM_ZIP = os.path.join(GEO, "COMM_RG_01M_2001_4326.shp.zip")
COMM_URL = ("https://gisco-services.ec.europa.eu/distribution/v2/communes/shp/"
            "COMM_RG_01M_2001_4326.shp.zip")

EXPECTED_UNITS = 192


def fetch():
    if os.path.exists(COMM_ZIP) and os.path.getsize(COMM_ZIP) > 50_000_000:
        print("already have", COMM_ZIP)
        return
    os.makedirs(GEO, exist_ok=True)
    print("GET", COMM_URL)
    req = urllib.request.Request(COMM_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=1800) as r, open(COMM_ZIP, "wb") as fh:
        fh.write(r.read())
    print(f"  {os.path.getsize(COMM_ZIP):,} bytes")


def main():
    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(COMM_ZIP):
        raise SystemExit(f"missing {COMM_ZIP} -- run with --fetch first")

    # The .dbf is cp1250 and its .cpg claims otherwise, so pyogrio's default utf-8 read
    # raises on the first caron.  Nothing here reads a name, but the file still has to open.
    gdf = gpd.read_file(COMM_ZIP, encoding="cp1250",
                        columns=["CNTR_CODE", "NSI_CODE", "COMM_CODE"])
    si = gdf[gdf["CNTR_CODE"] == "SI"].copy()
    si["obcina"] = si["NSI_CODE"].astype(str).str.zfill(3)
    print(f"GISCO Communes 2001: {len(si)} Slovenian polygons ({si.crs})")

    if len(si) != EXPECTED_UNITS:
        raise SystemExit(f"{len(si)} polygons, expected {EXPECTED_UNITS}")
    if si["obcina"].duplicated().any():
        dup = sorted(si.loc[si["obcina"].duplicated(), "obcina"])
        raise SystemExit(f"duplicate občina codes in GISCO: {dup}")

    # The check that matters: the polygon set and the count set must be the SAME 192 codes.
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} -- run sources/si.py first")
    counts = pd.read_csv(CSV, dtype={"geo_id": str})
    want = set(counts.loc[counts["geo_level"] == "municipality", "geo_id"])
    got = set(si["obcina"])
    if want != got:
        raise SystemExit(f"census has {sorted(want - got)} with no polygon; GISCO has "
                         f"{sorted(got - want)} with no counts")
    print(f"  OK  the {len(got)} codes in si.csv are exactly the {len(got)} in GISCO")

    names = dict(zip(counts["geo_id"], counts["geo_name"]))
    si["name"] = si["obcina"].map(names)
    si = si[["obcina", "name", "COMM_CODE", "geometry"]].rename(
        columns={"COMM_CODE": "comm_code"}).sort_values("obcina")

    os.makedirs(OUT_DIR, exist_ok=True)
    si.to_file(OUT, layer="si_units", driver="GPKG")
    print(f"wrote {OUT} ({len(si)} units)")
    area = si.to_crs(3794).area / 1e6
    print(f"  area: median {area.median():.0f} km2, largest "
          f"{si.loc[area.idxmax(), 'name']} {area.max():.0f} km2")


if __name__ == "__main__":
    main()
