"""Caribbean Netherlands: CBS survey shares per island, on CBS's island populations. 2026-09-14.

    python sources/bq.py --fetch      CBS 82868NED and 83774NED as JSON
    python sources/bq.py              normalise + build the island polygons

Writes:
    data/normalized/bq.csv          one row per (island, answer), counts = share x population
    data/geo/bq/bq_islands.gpkg     three island polygons from Natural Earth's map units

**A SURVEY, NOT A CENSUS.** The Netherlands runs no census question on religion. CBS's
Omnibus survey asks *kerkelijke gezindte* of persons aged 15 and over on each island and
publishes the share per island with a 95% margin (table 82868NED; rounds 2013, 2017/2018 and
2021; the 2021 figures are still marked provisional). The 2021 shares are applied to each
island's population on 1 January 2022 (table 83774NED), the date nearest the October to
December 2021 fieldwork, because the population has grown by about a fifth since through
immigration and a later base would describe people the survey did not. Every row is
`modelled` in countries.py, as for every other survey here.

**CELLS ARE WITHHELD.** CBS prints `.` where an estimate is too unreliable, so the printed
shares sum to 99.3 (Bonaire), 99.3 (Sint Eustatius) and 98.2 (Saba). The remainder is written
as `Niet gepubliceerd` and excluded (taxonomy/bq2021.py), so it becomes the gap rather than
being spread over religions the table did not print.

**NO POPULATION GRID.** Kontur's BQ extract is a valid GeoPackage with no hexes in it, so the
islands are placed uniformly inside Natural Earth's polygons (spec §8.2). Bonaire's 23 or so
dots will include some in Washington Slagbaai park and on the salt pans; at one dot per
thousand people that is a handful of dots, and it is stated here rather than discovered.
"""

import csv
import json
import os
import ssl
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bq")

SHARES_URL = "https://opendata.cbs.nl/ODataApi/odata/82868NED/TypedDataSet?$format=json"
POP_URL = "https://opendata.cbs.nl/ODataApi/odata/83774NED/TypedDataSet?$format=json"
SHARES = os.path.join(RAW, "82868NED_TypedDataSet.json")
POP = os.path.join(RAW, "83774NED_TypedDataSet.json")
NE_UNITS = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_map_units.geojson")

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

PERIOD = "2021JJ00"
POP_PERIOD = "2022JJ00"                  # BevolkingOp1Januari_1 of 2022 = 1 January 2022
ALL_PERSONS = "T009002"
SHARE, MARGIN = "MW00000", "B000150"
ISLANDS = {"GM9001": "Bonaire", "GM9002": "Sint Eustatius", "GM9003": "Saba"}
EXPECTED_POP = {"GM9001": 22_573, "GM9002": 3_242, "GM9003": 1_911}
# A few published shares pinned, so a revision of the provisional 2021 figures fails loudly.
PINNED = {("GM9001", "Rooms Katholiek"): 60.3, ("GM9002", "Methodist"): 24.8,
          ("GM9002", "Adventist"): 18.9, ("GM9003", "Anglicaans"): 8.9}
WITHHELD = "Niet gepubliceerd"

TOPICS = [("GeenGodsdienst_1", "Geen godsdienst"), ("RoomsKatholiek_2", "Rooms Katholiek"),
          ("Pinkstergemeente_3", "Pinkstergemeente"), ("Protestant_4", "Protestant"),
          ("Adventist_5", "Adventist"), ("Methodist_6", "Methodist"),
          ("Evangelisch_7", "Evangelisch"), ("Anglicaans_8", "Anglicaans"),
          ("Islam_9", "Islam"), ("Jehova_10", "Jehova"), ("Hindoeisme_11", "Hindoeïsme"),
          ("Anders_12", "Anders")]


def _get(url, dest):
    ctx = ssl.create_default_context()
    body = urllib.request.urlopen(url, timeout=120, context=ctx).read()
    with open(dest + ".part", "wb") as fh:
        fh.write(body)
    os.replace(dest + ".part", dest)
    print(f"  got {os.path.basename(dest)} ({len(body):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(SHARES_URL, SHARES)
    _get(POP_URL, POP)


def _rows(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run with --fetch")
    return json.load(open(path, encoding="utf-8"))["value"]


def normalise():
    shares = _rows(SHARES)
    pops = {r["CaribischNederland"].strip(): r["BevolkingOp1Januari_1"]
            for r in _rows(POP) if r["Perioden"] == POP_PERIOD}
    out = []
    drawn_total = withheld_total = 0
    for code, island in ISLANDS.items():
        pop = pops.get(code)
        if pop != EXPECTED_POP[code]:
            raise SystemExit(f"{island}: population {pop} on {POP_PERIOD}, expected "
                             f"{EXPECTED_POP[code]:,}")
        pick = {r["Marges"]: r for r in shares
                if r["CaribischNederland"].strip() == code and r["Perioden"] == PERIOD
                and r["Persoonskenmerken"].strip() == ALL_PERSONS}
        if set(pick) != {SHARE, MARGIN}:
            raise SystemExit(f"{island}: no {PERIOD} share and margin rows")
        printed = {label: pick[SHARE][key] for key, label in TOPICS
                   if pick[SHARE][key] is not None}
        for (c, label), want in PINNED.items():
            if c == code and printed.get(label) != want:
                raise SystemExit(f"{island}: {label} is {printed.get(label)}, pinned {want}")
        total_share = sum(printed.values())
        if not 97.5 <= total_share <= 100.05:
            raise SystemExit(f"{island}: printed shares sum to {total_share:.1f}")
        used = 0
        for key, label in TOPICS:
            s = pick[SHARE][key]
            if s is None:
                continue
            n = round(s / 100.0 * pop)
            used += n
            m = pick[MARGIN][key]
            out.append(dict(geo_id=code, geo_level="island", geo_name=island,
                            source_category=label, count=n, basis="self_id", year=2021,
                            source_id="cbs_82868NED_2021",
                            note=f"share={s}%; margin={m}; persons 15+; population "
                                 f"1 Jan 2022={pop}"))
        rest = pop - used
        out.append(dict(geo_id=code, geo_level="island", geo_name=island,
                        source_category=WITHHELD, count=rest, basis="self_id", year=2021,
                        source_id="cbs_82868NED_2021",
                        note=f"100 minus the printed shares ({total_share:.1f}); withheld "
                             f"cells and rounding"))
        drawn_total += used
        withheld_total += rest
        withheld = [label for key, label in TOPICS if pick[SHARE][key] is None]
        print(f"  {island:<16} pop {pop:>6,}  printed {total_share:5.1f}%  "
              f"withheld {rest:>4} ({100 * rest / pop:.1f}%): {', '.join(withheld)}")

    # The earlier round, for the record: does the island's shape hold between rounds?
    for code, island in ISLANDS.items():
        a = {r["Perioden"]: r for r in shares
             if r["CaribischNederland"].strip() == code and r["Marges"] == SHARE
             and r["Persoonskenmerken"].strip() == ALL_PERSONS}
        top = sorted(((a[PERIOD][k] or 0, lbl, k) for k, lbl in TOPICS), reverse=True)[:4]
        print(f"    {island:<16}" + "; ".join(
            f"{lbl} {a['2017JJ00'][k]} -> {v}" for v, lbl, k in top))

    path = os.path.join(ROOT, "data", "normalized", "bq.csv")
    with open(path + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    os.replace(path + ".part", path)
    print(f"  bq: {drawn_total:,} drawn, {withheld_total:,} withheld, of "
          f"{drawn_total + withheld_total:,}")


def geometry():
    import geopandas as gpd

    if not os.path.exists(NE_UNITS):
        raise SystemExit(f"missing {NE_UNITS} (country_shapes.py names the URL)")
    mu = gpd.read_file(NE_UNITS)
    cn = mu[mu["GU_A3"] == "NLY"]
    if len(cn) != 1:
        raise SystemExit(f"Natural Earth map units: {len(cn)} features with GU_A3 NLY")
    parts = cn.explode(index_parts=False).reset_index(drop=True)
    cx, cy = parts.geometry.centroid.x, parts.geometry.centroid.y
    parts["unit"] = "GM9002"
    parts.loc[cx < -65.0, "unit"] = "GM9001"
    parts.loc[(cx > -65.0) & (cy > 17.56), "unit"] = "GM9003"
    got = parts.groupby("unit").size().to_dict()
    if got != {"GM9001": 1, "GM9002": 1, "GM9003": 1}:
        raise SystemExit(f"Caribbean Netherlands parts by island: {got}")
    parts["name"] = parts["unit"].map(ISLANDS)
    out_dir = os.path.join(ROOT, "data", "geo", "bq")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "bq_islands.gpkg")
    parts[["unit", "name", "geometry"]].set_crs("EPSG:4326", allow_override=True).to_file(
        out, layer="islands", driver="GPKG")
    for _, r in parts.iterrows():
        print(f"    {r['name']:<16} {r['unit']}  centroid {r.geometry.centroid.x:.3f}, "
              f"{r.geometry.centroid.y:.3f}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    normalise()
    geometry()


if __name__ == "__main__":
    main()
