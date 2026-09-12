"""Where England's churches are now — a current register, by denomination and Output Area.

`sources/uk_churches.py` -> `data/normalized/uk_churches.csv`

WHY THIS EXISTS, AND IT IS A CORRECTION. `sources/uk_ecc.py` was going to place England's
denominations on its own, using the English Church Census's attendance by county. **That was
wrong and the map it drew said Merseyside was 72% Anglican and 16% Catholic**, which is not
Liverpool. The cause is in the church census's own user guide: alongside its postal returns
it took bulk data from "ten Church of England and eight Roman Catholic Dioceses", so a
county's count depends on whether its diocese sent a spreadsheet.

Measured against ground truth, the church census's Anglican response rate is:

    nationally     56.2%   (its user guide says 55% -- so the measurement is sound)
    Norfolk        25.9%   Tyne & Wear 37.3%   Durham 35.5%
    Merseyside     91.4%   Gloucestershire 92.3%   Greater Manchester 92.5%

A factor of three and a half between counties, and the high ones are the ten dioceses. A
single national correction cannot see that, which is exactly what uk_ecc.py used to apply.

SO PLACEMENT IS SPLIT IN TWO, and each source does only what it is good at:

    this file        HOW MANY churches of each denomination are in each place, now.
    sources/uk_ecc.py   HOW BIG a congregation of that denomination is there, from 2005.

    placement weight = churches x mean congregation

The church census's mean congregation size is a WITHIN-denomination quantity, so it survives
a response rate that varies by county and denomination — losing half a county's Methodist
chapels changes how many you saw, not how big they were. Its totals do not survive that, and
are no longer used for anything.

TWO REGISTERS, AND THE FIRST VALIDATES THE SECOND.

  **The Church of England's own church locations** (`Churches_July2026` on the CofE's ArcGIS
  org) — 15,784 churches, 15,492 with usable coordinates in England. Authoritative: it is
  the CofE's own operational list, not a survey of it. Anglican placement comes from here
  and OSM's Anglican churches are discarded, because two registers of the same thing added
  together is 32,000 Anglican churches in a country with about 16,000.

  **OpenStreetMap**, for everything else, because no other free register covers all
  denominations at once. Its trustworthiness is not assumed, it is measured on the one
  denomination where a complete list exists: **OSM's Anglican churches against the CofE's
  own, across the 47 counties, correlate at r = 0.97**, with median coverage 103% and an
  interquartile range of 100-105%. `check()` recomputes that on every build.

WHICH LEGS THIS FILE CAN PLACE, AND WHICH IT DELIBERATELY DOES NOT. OSM maps buildings, so
it finds denominations that own buildings and misses denominations that rent halls:

    anglican   catholic   methodist   baptist   reformed        placed here
    pentecostal   newchurch   orthodox   other                  NOT placed

OSM has **355 Pentecostal and 129 Orthodox churches in the whole of England**, both large
undercounts — Pentecostal congregations meet in industrial units and hired schools, Orthodox
parishes are mostly post-2004 and often share Anglican buildings, and `newchurch` (Vineyard,
Newfrontiers, FIEC) has no OSM denomination tag at all. Using those counts anyway put 6.0%
of Norfolk's Christians on Orthodoxy and 7.6% of Durham's on Pentecostalism, neither of
which is true. **Those four legs are deferred to a census-proxy placement** (Anita,
2026-09-07: "lets do pentecostal and orthodox later, get the real placeables first") and
until then their 8.0% of England's Christians stays on the residual rather than being
invented into a county.

A CHURCH SHARED BETWEEN DENOMINATIONS COUNTS AS A FRACTION OF EACH. OSM writes Local
Ecumenical Partnerships as `anglican;methodist`, `methodist;united_reformed` and so on, 100+
of them. Each named denomination gets 1/n of the building rather than the whole of it, which
is the only reading that keeps the national totals right.

Run: python sources/uk_churches.py --fetch   # re-download both registers first
     python sources/uk_churches.py           # -> data/normalized/uk_churches.csv
     python sources/uk_churches.py --report  # print the validation, write nothing
"""
import argparse
import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "uk")
OUT = os.path.join(ROOT, "data", "normalized", "uk_churches.csv")

COE = os.path.join(RAW, "coe_churches.csv")
OSM = os.path.join(RAW, "osm_churches.json")
CENTROIDS = os.path.join(RAW, "oa_centroids.csv")
OA_LAD = os.path.join(RAW, "oa_lad.csv")
OA_RUC = os.path.join(RAW, "oa_ruc.csv")

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

SOURCE_ID = "uk_en_churches_2026"
BASIS = "roll"          # an institution counted, spec §3.1 -- buildings, not members
YEAR = 2026

# OSM `denomination` -> leg. Anglican is absent on purpose: it comes from the CofE's own
# register. Welsh-only bodies (church_in_wales, welsh_independent, welsh_baptist) appear
# because the Overpass query covered England and Wales; only England is emitted.
OSM_LEG = {
    "catholic": "catholic", "roman_catholic": "catholic",
    "ukrainian_greek_catholic": "catholic",

    "methodist": "methodist", "calvinistic_methodist": "methodist",
    "wesleyan": "methodist", "primitive_methodist": "methodist",
    "united_methodist": "methodist", "free_methodist": "methodist",
    "independent_methodist": "methodist",

    "baptist": "baptist", "strict_baptist": "baptist", "welsh_baptist": "baptist",
    "reformed_baptist": "baptist",

    "united_reformed": "reformed", "urc": "reformed", "united_reform": "reformed",
    "presbyterian": "reformed", "congregational": "reformed", "reformed": "reformed",
    "church_of_scotland": "reformed", "efcc": "reformed",
}
PLACED = ("anglican", "catholic", "methodist", "baptist", "reformed")

# See the docstring. Named so that a future proxy placement has something to look for.
DEFERRED = ("pentecostal", "newchurch", "orthodox", "other")


def _oa_index():
    """(KDTree over OA centroids, array of county, array of settlement)."""
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
    sys.path.insert(0, ROOT)
    from uk_split import lad_to_county

    cent = pd.read_csv(CENTROIDS)
    lad = pd.read_csv(OA_LAD)
    ruc = pd.read_csv(OA_RUC)
    oa = cent.merge(lad, on="OA21CD").merge(ruc, on="OA21CD")
    oa["county"] = oa["LAD23NM"].map(lad_to_county())
    oa = oa[oa["county"].notna()].reset_index(drop=True)
    oa["settlement"] = oa["RUC21CD"].str.startswith("R").map({True: "rural",
                                                             False: "urban"})
    tree = cKDTree(np.c_[oa["LAT"].values, oa["LONG"].values])
    return tree, oa["county"].values, oa["settlement"].values


def _assign(tree, counties, settlements, lats, lons):
    import numpy as np
    _, idx = tree.query(np.c_[lats, lons], k=1)
    return counties[idx], settlements[idx]


def register():
    """(counts keyed (county, settlement, leg), diagnostics)."""
    import numpy as np
    import pandas as pd

    for path in (COE, OSM, CENTROIDS, OA_LAD, OA_RUC):
        if not os.path.exists(path):
            sys.exit(f"missing {path}\n  run: python sources/uk_churches.py --fetch")

    tree, counties, settlements = _oa_index()
    counts = collections.Counter()
    diag = {}

    # --- Anglican, from the Church of England's own list
    coe = pd.read_csv(COE).dropna(subset=["Latitude", "Longitude"])
    n_raw = len(coe)
    coe = coe[coe["Latitude"].between(49.0, 56.0) & coe["Longitude"].between(-7.0, 2.5)]
    cty, st = _assign(tree, counties, settlements, coe["Latitude"].values,
                      coe["Longitude"].values)
    for c, s in zip(cty, st):
        counts[(c, s, "anglican")] += 1.0
    diag["coe_rows"] = n_raw
    diag["coe_placed"] = len(coe)

    # --- everything else, from OpenStreetMap
    els = json.load(open(OSM, encoding="utf-8"))["elements"]
    lats, lons, legs = [], [], []
    unmapped = collections.Counter()
    for e in els:
        tags = e.get("tags", {})
        raw = tags.get("denomination")
        if not raw:
            continue
        lat = e.get("lat") or (e.get("center") or {}).get("lat")
        lon = e.get("lon") or (e.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        parts = [p.strip().lower() for p in str(raw).split(";") if p.strip()]
        hits = [OSM_LEG[p] for p in parts if p in OSM_LEG]
        osm_anglican = [p for p in parts if p in ("anglican", "church_of_england")]
        if not hits:
            if not osm_anglican:
                unmapped[raw] += 1
            continue
        # a shared building is a fraction of each denomination named on it, including the
        # Anglican share, which is dropped -- the CofE register already counts that building
        share = 1.0 / len(parts)
        for leg in hits:
            lats.append(lat)
            lons.append(lon)
            legs.append((leg, share))
    if lats:
        cty, st = _assign(tree, counties, settlements, np.array(lats), np.array(lons))
        for c, s, (leg, share) in zip(cty, st, legs):
            counts[(c, s, leg)] += share
    diag["osm_elements"] = len(els)
    diag["osm_placed"] = len(lats)
    diag["osm_unmapped_tags"] = unmapped

    # --- OSM's own Anglicans, kept only to validate OSM against the CofE register
    a_lat, a_lon = [], []
    for e in els:
        if (e.get("tags", {}).get("denomination") or "").strip().lower() == "anglican":
            lat = e.get("lat") or (e.get("center") or {}).get("lat")
            lon = e.get("lon") or (e.get("center") or {}).get("lon")
            if lat is not None:
                a_lat.append(lat)
                a_lon.append(lon)
    ac, _ = _assign(tree, counties, settlements, np.array(a_lat), np.array(a_lon))
    diag["osm_anglican_by_county"] = collections.Counter(ac)
    return counts, diag


def check(counts, diag):
    """Print the register and, crucially, OSM measured against the CofE's own list."""
    import pandas as pd
    fails = 0
    print("England's churches, by denomination and county x settlement")
    print(f"  Church of England register       {diag['coe_placed']:,} placed "
          f"of {diag['coe_rows']:,}")
    print(f"  OpenStreetMap                    {diag['osm_placed']:,} placed "
          f"of {diag['osm_elements']:,} Christian places of worship")

    by_leg = collections.Counter()
    for (_, _, leg), n in counts.items():
        by_leg[leg] += n
    print("\n  leg           churches   cells")
    cells = collections.Counter()
    for (c, s, leg) in counts:
        cells[leg] += 1
    for leg in PLACED:
        print(f"  {leg:12s} {by_leg[leg]:9,.0f} {cells[leg]:7d}")
    print(f"  deferred, not placed here: {', '.join(DEFERRED)}")

    truth = collections.Counter()
    for (c, s, leg), n in counts.items():
        if leg == "anglican":
            truth[c] += n
    osm_ang = diag["osm_anglican_by_county"]
    t = pd.DataFrame({"cofe": pd.Series(truth), "osm": pd.Series(osm_ang)}).dropna()
    t["coverage %"] = 100 * t["osm"] / t["cofe"]
    r = t["osm"].corr(t["cofe"])
    print("\n  VALIDATION -- OSM's Anglicans against the CofE's own register, 47 counties")
    print(f"    correlation r                  {r:.4f}")
    print(f"    national coverage              {100*t['osm'].sum()/t['cofe'].sum():.1f}%")
    print(f"    per-county median              {t['coverage %'].median():.0f}%"
          f"   IQR {t['coverage %'].quantile(.25):.0f}-{t['coverage %'].quantile(.75):.0f}%"
          f"   range {t['coverage %'].min():.0f}-{t['coverage %'].max():.0f}%")
    if r < 0.90:
        print("    ! OSM no longer tracks the real Anglican geography; the other legs "
              "rest on this and should not be trusted until it is understood")
        fails += 1

    if diag["osm_unmapped_tags"]:
        top = diag["osm_unmapped_tags"].most_common(8)
        big = [f"{k} ({v})" for k, v in top if v >= 20]
        if big:
            print("\n  OSM denominations with 20+ churches and no leg (expected: the "
                  "deferred four, plus bodies outside this split):")
            print("    " + ", ".join(big))
    return fails


def _rows(counts):
    for (county, settlement, leg), n in sorted(counts.items()):
        yield {
            "geo_id": f"{county}|{settlement}",
            "geo_level": "county_settlement",
            "geo_name": f"{county}, {settlement}",
            "source_category": leg,
            "count": round(n, 3),
            "basis": BASIS,
            "year": YEAR,
            "source_id": SOURCE_ID,
            "note": "churches, not people; shared buildings split between denominations",
        }


def fetch():
    sys.exit("re-fetch is manual for now; see sources/uk_churches.md for the two URLs "
             "and the Overpass query")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()

    counts, diag = register()
    fails = check(counts, diag)
    if args.report:
        return 1 if fails else 0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = list(_rows(counts))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows):,} rows -> {OUT}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
