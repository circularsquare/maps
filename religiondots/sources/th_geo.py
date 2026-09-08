"""Thailand — boundaries for the 77 changwat.

Writes data/geo/th/th_provinces.gpkg and data/geo/th/th_lookup.csv.

**geoBoundaries gbOpen THA ADM1, and it needs no reconciling: 77 polygons for 77
provinces.** That is unusual enough to be worth stating, because §12's whole boundary section
is about the cases where it is not — Cambodia's Tbong Khmum, Nepal's national parks, Kenya's
two files both labelled ADM2. Thailand's provinces are stable, coded and few, and both sides
of this join agree on all 77.

**THE JOIN IS ON THE TIS 1099 CODE, NOT ON A NAME, AND OCHA'S COD SUPPLIES THE BRIDGE.**
geoBoundaries carries English `shapeName` only; the census carries Thai only. OCHA's COD
attribute table (`sources/th.py` downloads it) carries `adm1_name`, `adm1_name1` and
`adm1_pcode` in one row, so English and Thai meet on a published standard instead of on each
other. That matters more than it looks: `Phra Nakhon Si Ayutthaya`, `Nakhon Si Thammarat`,
`Nakhon Ratchasima`, `Nakhon Sawan`, `Nakhon Nayok`, `Nakhon Pathom` and `Nakhon Phanom` are
seven provinces whose names differ only in the second word, and Thai romanisation is not
stable across sources — `Chainat`/`Chai Nat`, `Buriram`/`Buri Ram`, `Sisaket`/`Si Sa Ket`.

**THE UNIT ID IS `<region digit><two-digit TIS code>`** and it is not arbitrary: spec §3.10's
allocation runs `--within 1`, which finds a province's region by taking the first character
of its geo_id. So the id has to carry the region, and `sources/th.py` builds it. This module
reads `data/normalized/th.csv` for the mapping rather than repeating it, which means **th.py
must be run first** — the same ordering cn_geo.py/cn.py have, in the other direction.

Usage:
    python sources/th_geo.py --fetch    one ~9 MB geojson from geoBoundaries
    python sources/th_geo.py            rebuild from data/raw/th/
"""

import csv
import io
import json
import os
import re
import sys
import unicodedata
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "th")
GEO = os.path.join(ROOT, "data", "geo", "th")
NORM = os.path.join(ROOT, "data", "normalized", "th.csv")
OUT = os.path.join(GEO, "th_provinces.gpkg")
LOOKUP = os.path.join(GEO, "th_lookup.csv")

UA = "religiondots/1.0 (map research; anitaxinchen@gmail.com)"
GB_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
          "THA/ADM1/geoBoundaries-THA-ADM1.geojson")
GB_NAME = "geoBoundaries-THA-ADM1.geojson"
EXPECTED = 77
# TIS 1099 codes. Bueng Kan post-dates the census by seven months — see main().
BUENG_KAN = "38"
NONG_KHAI = "43"


def fold(s):
    """Lowercase, drop everything but letters and digits, and strip the generic word.

    **geoBoundaries suffixes 75 of its 77 names with ` Province` and the other two with
    nothing** — `Bangkok` and, for no reason anybody records, `Kalasin`. So the suffix
    cannot be relied on either way and is simply removed from both sides of the join. This
    is §12's rule about a name join: the difference has to be handled as a CLASS (a generic
    administrative word) rather than as 75 aliases.
    """
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[^0-9a-z]+", "", s.lower())
    return re.sub(r"province$", "", s)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, GB_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        print("already have", dest)
        return
    print("GET", GB_URL)
    req = urllib.request.Request(GB_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    # §5a: a 200 is not a download.
    if not data.lstrip()[:1] == b"{":
        raise SystemExit("geoBoundaries did not return JSON (%d bytes)" % len(data))
    open(dest, "wb").write(data)
    print("  %s bytes" % "{:,}".format(len(data)))


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    os.makedirs(GEO, exist_ok=True)

    # ---- the census's own unit ids, which carry the region prefix
    if not os.path.exists(NORM):
        raise SystemExit("run `python sources/th.py --fetch` first -- this module reads "
                         "%s for the region-prefixed unit ids" % NORM)
    th = pd.read_csv(NORM, dtype={"geo_id": str})
    prov = th[th["geo_level"] == "province"][["geo_id", "geo_name"]].drop_duplicates()
    by_code = {g[1:]: g for g in prov["geo_id"]}          # two-digit TIS -> unit id
    thai_name = dict(zip(prov["geo_id"], prov["geo_name"]))
    print("th.csv: %d province unit ids" % len(by_code))

    # ---- OCHA COD: English name -> TIS code
    cod = pd.read_excel(os.path.join(RAW, "tha_admin_boundaries.xlsx"),
                        sheet_name="tha_admin1")
    en_to_code = {}
    for _, r in cod.iterrows():
        code = str(r["adm1_pcode"]).strip()
        if re.fullmatch(r"TH\d{2}", code):
            en_to_code[fold(r["adm1_name"])] = code[2:]
    print("COD: %d English names on the TIS code" % len(en_to_code))

    # ---- geoBoundaries polygons
    gdf = gpd.read_file(os.path.join(RAW, GB_NAME))
    print("geoBoundaries: %d polygons, crs=%s" % (len(gdf), gdf.crs))
    if len(gdf) != EXPECTED:
        print("  !! expected %d" % EXPECTED)

    gdf["code"] = gdf["shapeName"].map(lambda s: en_to_code.get(fold(s)))
    missing = gdf[gdf["code"].isna()]
    if len(missing):
        print("  !! %d shapeNames not in COD: %s"
              % (len(missing), sorted(missing["shapeName"])[:12]))
    # ---- spec §8.1: the boundaries have to be the vintage the DATA was published on.
    # Bueng Kan was carved out of Nong Khai on 23 March 2011, seven months after the
    # census, so the 2010 tables have 76 changwat and this file has 77. Dropping the extra
    # polygon would leave a province-sized hole in the northeast and quietly shrink Nong
    # Khai; dissolving it back is what puts the map on the census's own map.
    if BUENG_KAN in set(gdf["code"]) and NONG_KHAI in set(gdf["code"]):
        n_before = len(gdf)
        gdf.loc[gdf["code"] == BUENG_KAN, "code"] = NONG_KHAI
        gdf = gdf.dissolve(by="code", as_index=False, aggfunc="first")
        print("  §8.1: dissolved Bueng Kan (TH38, created 2011) back into Nong Khai "
              "(TH43) -- %d polygons -> %d" % (n_before, len(gdf)))

    gdf["unit"] = gdf["code"].map(lambda c: by_code.get(c) if isinstance(c, str) else None)
    lost = gdf[gdf["unit"].isna()]
    if len(lost):
        print("  !! %d polygons have no census unit: %s"
              % (len(lost), sorted(lost["shapeName"])[:12]))
    gdf = gdf[gdf["unit"].notna()].copy()

    if gdf["unit"].duplicated().any():
        dup = sorted(gdf[gdf["unit"].duplicated(keep=False)]["shapeName"])
        raise SystemExit("two polygons claim one unit: %s" % dup)

    gdf["name_th"] = gdf["unit"].map(thai_name)
    gdf = gdf[["unit", "shapeName", "name_th", "geometry"]].rename(
        columns={"shapeName": "name_en"})
    gdf.to_file(OUT, driver="GPKG", layer="provinces")
    print("\nwrote %s -- %d provinces" % (OUT, len(gdf)))

    with io.open(LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "name_en", "name_th"])
        for _, r in gdf.sort_values("unit").iterrows():
            w.writerow([r["unit"], r["name_en"], r["name_th"]])
    print("wrote %s" % LOOKUP)

    unmatched = set(by_code.values()) - set(gdf["unit"])
    if unmatched:
        print("  !! %d census provinces have no polygon: %s"
              % (len(unmatched), sorted(unmatched)))
    else:
        print("  every census province has a polygon")


if __name__ == "__main__":
    main()
