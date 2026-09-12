"""Austria — the drawn units: GISCO *Communes 2001* plus Vienna's 23 Gemeindebezirke.

Writes data/geo/at/at_units.gpkg and data/geo/at/at_lookup.csv.

**THE BOUNDARY VINTAGE IS THE WHOLE PROBLEM, AND GISCO SOLVES IT OUTRIGHT.** Austria has
merged Gemeinden hard since the census: Styria alone went from 542 to 287 in the reform of
2015, and Carinthia, Burgenland and Upper Austria have all moved. Joining a 2001 table to a
current boundary file therefore loses a third of one Bundesland and silently mis-seats the
rest — §8.1's Connecticut trap in a much larger size. Eurostat's GISCO publishes a
**Communes 2001** layer, which is the Gebietsstand of the census to the day, so no crosswalk
is written, none is needed, and nothing has to be aggregated up.

The join is on the **Topographische Kennziffer**, which the census prints in its Vorspalte and
GISCO carries as `NSI_CODE` (and, prefixed, as `COMM_ID`). Not on names: 2,358 Austrian
Gemeinde names include dozens of near-duplicates across Bundesländer, which is exactly the
shape `[[reference_name_join_wrong_neighbour]]` describes. Names are carried through for the
unit panel and are checked *against* the join rather than used to make it.

**VIENNA IS NOT A GEMEINDE HERE, AND THAT IS AN UPGRADE.** GISCO has one polygon for Wien,
1,550,123 people — which would be the coarsest drawn unit on this map by a factor of twenty,
in the one part of Austria whose composition is least like the rest (25.6% ohne Bekenntnis
against 12.0% nationally). The Wien volume publishes Tabelle 4 by Zählbezirk, with the 23
Gemeindebezirke as the tier above, so the city is drawn at Bezirk instead.

**AND THE CITY PUBLISHES THE STATISTICAL OFFICE'S OWN KEY.** `data.wien.gv.at`'s
BEZIRKSGRENZEOGD layer carries `STATAUSTRIA_BEZ_CODE` (901…923) and `STATAUSTRIA_GEM_CODE`
alongside its own district number, so the join to the census is an integer equality and there
is no name matching and no assumption that district *n* is Kennziffer 900+*n*. That the two
agree anyway is asserted, because it is free.

**THE ZÄHLBEZIRKE ARE PARSED AND NOT DRAWN.** at.csv carries all 245 of them. Vienna's
current OGD Zählbezirk layer has 250, so the two vintages do not correspond and pairing them
would need a crosswalk nobody publishes; Benin's rule applies (parsed, kept, not drawn). If a
2001-vintage Zählbezirk layer ever turns up, Vienna gets six times finer for free.

Usage:
    python sources/at_geo.py --fetch    one 80 MB GISCO zip + one 3 MB GeoJSON
    python sources/at_geo.py            rebuild from data/geo/
"""

import json
import os
import ssl
import sys
import urllib.request
from collections import Counter

import geopandas as gpd
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO = os.path.join(ROOT, "data", "geo")
OUT_DIR = os.path.join(GEO, "at")
OUT = os.path.join(OUT_DIR, "at_units.gpkg")
LOOKUP = os.path.join(OUT_DIR, "at_lookup.csv")
CSV = os.path.join(ROOT, "data", "normalized", "at.csv")

COMM_ZIP = os.path.join(GEO, "COMM_RG_01M_2001_4326.shp.zip")
COMM_URL = ("https://gisco-services.ec.europa.eu/distribution/v2/communes/shp/"
            "COMM_RG_01M_2001_4326.shp.zip")
COMM_CSV = os.path.join(GEO, "comm2001_at.csv")
COMM_CSV_URL = ("https://gisco-services.ec.europa.eu/distribution/v2/communes/csv/"
                "COMM_AT_2001.csv")
WIEN_JSON = os.path.join(GEO, "wien_bezirke.json")
WIEN_URL = ("https://data.wien.gv.at/daten/geo?service=WFS&request=GetFeature"
            "&version=1.1.0&typeName=ogdwien:BEZIRKSGRENZEOGD&srsName=EPSG:4326"
            "&outputFormat=json")

# Stallehr (80125), 272 people, in the Klostertal between Bludenz and Bürs — the smallest
# Gemeinde in Vorarlberg and one of the smallest in Austria.  GISCO's 2001 commune layer has
# no polygon for it at all; every one of its neighbours (80120-80124, 80126-80129) is present,
# so this is an omission in that layer rather than a boundary-vintage disagreement.  Its 272
# people are dropped rather than folded into a neighbour (§3.5: marked, not filled) and are
# named in countries.py's `gap`.  Pinned by code, not by a tolerance, so that a SECOND missing
# unit fails the build instead of being absorbed into "a few".
KNOWN_MISSING = {"80125"}


def _fetch():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    os.makedirs(GEO, exist_ok=True)
    for dest, url, what in ((COMM_ZIP, COMM_URL, "GISCO Communes 2001 (80 MB)"),
                            (COMM_CSV, COMM_CSV_URL, "GISCO commune names (8 MB)"),
                            (WIEN_JSON, WIEN_URL, "Vienna Gemeindebezirke (3 MB)")):
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            print("  have %s" % os.path.basename(dest))
            continue
        print("  downloading %s" % what)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=1800) as r:
            body = r.read()
        with open(dest + ".part", "wb") as f:
            f.write(body)
        os.replace(dest + ".part", dest)
        print("  got %s (%d bytes)" % (os.path.basename(dest), len(body)))


def main():
    if "--fetch" in sys.argv:
        _fetch()
    for p in (COMM_ZIP, COMM_CSV, WIEN_JSON, CSV):
        if not os.path.exists(p):
            raise SystemExit("missing %s — run sources/at.py and at_geo.py --fetch" % p)

    # ---------------------------------------------------------------- the census units
    df = pd.read_csv(CSV, dtype=str)
    gem = df[df["geo_level"] == "gemeinde"][["geo_id", "geo_name"]].drop_duplicates()
    bez = df[df["geo_level"] == "gemeindebezirk"][["geo_id", "geo_name"]].drop_duplicates()
    print("census: %d Gemeinden outside Vienna, %d Wiener Gemeindebezirke"
          % (len(gem), len(bez)))

    # ---------------------------------------------------------------- GISCO
    # THE SHAPEFILE LIES ABOUT ITS OWN ENCODING.  Its .cpg declares UTF-8 and its .dbf holds
    # bytes that are not UTF-8, so the default read dies on a commune somewhere in Europe
    # before it ever reaches Austria.  Reading it as latin-1 gets the geometry out; the NAMES
    # are then taken from GISCO's separate attribute CSV, which really is UTF-8 and returns
    # `Bartholomäberg` rather than a mojibake of it.  Nothing here joins on a name anyway.
    print("reading GISCO Communes 2001 …")
    comm = gpd.read_file("zip://" + COMM_ZIP, encoding="latin1")
    at = comm[comm["CNTR_CODE"] == "AT"].copy()
    at["code"] = at["NSI_CODE"].astype(str).str.strip().str.strip('"')
    # One commune is several FEATURES here, split by surface cover (SURF_COVR), so the layer
    # has more rows than Austria has Gemeinden.  Dissolving on the Kennziffer is what makes
    # "one polygon per drawn unit" true; without it the same Gemeinde is drawn several times
    # and the placement layer double-counts its area.
    n_feat = len(at)
    at = at.dissolve(by="code", as_index=False, aggfunc="first")
    print("  %d features dissolved on the Kennziffer to %d communes" % (n_feat, len(at)))
    names = pd.read_csv(COMM_CSV, dtype=str)
    names = names[names["CNTR_CODE"] == "AT"]
    names["code"] = names["NSI_CODE"].astype(str).str.strip().str.strip('"')
    at = at.drop(columns=["SABE_NAME"]).merge(
        names[["code", "SABE_NAME"]], on="code", how="left", validate="one_to_one")
    if at["SABE_NAME"].isna().any():
        raise SystemExit("%d Austrian communes have no name in GISCO's attribute CSV"
                         % int(at["SABE_NAME"].isna().sum()))
    print("  %d Austrian communes at the 2001 Gebietsstand" % len(at))
    if at["code"].duplicated().any():
        raise SystemExit("GISCO has duplicate Austrian commune codes")

    wien_poly = at[at["code"] == "90001"]
    if len(wien_poly) != 1:
        raise SystemExit("expected exactly one GISCO polygon for Wien, got %d" % len(wien_poly))
    at_nonwien = at[at["code"] != "90001"].copy()

    # ---------------------------------------------------------------- the join
    census_codes = set(gem["geo_id"].str[2:])
    gisco_codes = set(at_nonwien["code"])
    missing = census_codes - gisco_codes
    extra = gisco_codes - census_codes
    if extra:
        raise SystemExit("GISCO has %d communes the census does not: %s"
                         % (len(extra), sorted(extra)[:10]))
    if missing != KNOWN_MISSING:
        raise SystemExit("the census has %d Gemeinden GISCO does not, expected %s: %s"
                         % (len(missing), sorted(KNOWN_MISSING), sorted(missing)[:10]))
    print("  join on the Kennziffer: %d of %d Gemeinden matched, %d missing (%s)"
          % (len(census_codes & gisco_codes), len(census_codes), len(missing),
             ", ".join(sorted(missing))))

    # names are a CHECK on the join, never the join itself
    nm = gem.assign(code=gem["geo_id"].str[2:]).merge(
        at_nonwien[["code", "SABE_NAME"]], on="code", how="inner")

    def fold(s):
        s = str(s).lower()
        for a, b in (("ä", "a"), ("ö", "o"), ("ü", "u"), ("ß", "ss"), ("-", " "),
                     ("sankt", "st"), (".", ""), ("'", "")):
            s = s.replace(a, b)
        return " ".join(s.split())

    agree = sum(fold(a) == fold(b) for a, b in zip(nm["geo_name"], nm["SABE_NAME"]))
    print("  names agree on %d of %d after folding (%.1f%%); the census PDFs lose every "
          "umlaut to a broken font encoding, so this is a floor, not a defect rate"
          % (agree, len(nm), agree / len(nm) * 100))
    if agree / len(nm) < 0.55:
        raise SystemExit("only %.1f%% of names agree — the Kennziffer join is suspect"
                         % (agree / len(nm) * 100))

    # ---------------------------------------------------------------- Vienna
    wien = gpd.read_file(WIEN_JSON)
    if len(wien) != 23:
        raise SystemExit("expected 23 Wiener Gemeindebezirke, got %d" % len(wien))
    wien["code"] = wien["STATAUSTRIA_BEZ_CODE"].astype(int).astype(str)
    if (wien["STATAUSTRIA_BEZ_CODE"].astype(int) != 900 + wien["BEZNR"].astype(int)).any():
        raise SystemExit("a Vienna district's STATAUSTRIA_BEZ_CODE is not 900 + its number")
    if set(wien["code"]) != set(bez["geo_id"].str[2:]):
        raise SystemExit("Vienna's OGD codes do not match the census Gemeindebezirk codes")
    print("  Vienna: 23 Gemeindebezirke joined on STATAUSTRIA_BEZ_CODE, an integer equality")

    # the 23 districts must reproduce Vienna's own commune polygon
    a_gisco = wien_poly.to_crs(3035).geometry.area.sum()
    a_wien = wien.to_crs(3035).geometry.area.sum()
    ratio = a_wien / a_gisco
    if not 0.97 <= ratio <= 1.03:
        raise SystemExit("the 23 districts cover %.3f of GISCO's Wien polygon" % ratio)
    print("  and they cover %.4f of GISCO's single Wien polygon by area" % ratio)

    # ---------------------------------------------------------------- assemble
    a = at_nonwien[["code", "SABE_NAME", "geometry"]].rename(columns={"SABE_NAME": "name"})
    a["level"] = "gemeinde"
    b = wien[["code", "NAMEK", "geometry"]].rename(columns={"NAMEK": "name"})
    b["level"] = "gemeindebezirk"
    out = pd.concat([a, b], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=4326)
    out["unit"] = "AT" + out["code"]

    if out["unit"].duplicated().any():
        raise SystemExit("duplicate unit ids after assembling Vienna")
    print("\n%d drawn units: %d Gemeinden + 23 Wiener Gemeindebezirke"
          % (len(out), len(a)))

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "level", "geometry"]].to_file(OUT, layer="units", driver="GPKG")
    print("wrote %s" % OUT)
    lut = out[["unit", "name", "level"]].copy()
    lut.to_csv(LOOKUP, index=False)
    print("wrote %s (%d rows)" % (LOOKUP, len(lut)))


if __name__ == "__main__":
    main()
