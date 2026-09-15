"""Five small territories from their own census tables: vg fk kn aw cw. 2026-09-14.

    python sources/terr.py --fetch      the office documents (for the record) + Kontur extracts
    python sources/terr.py              normalise all five + build their placement layers
    python sources/terr.py vg fk        just those

Writes, per territory:
    data/normalized/<cc>.csv        one row per (unit, religion), exactly as the office printed it
    data/geo/<cc>/<cc>_hexes.gpkg   Kontur population hexagons, `unit` set on every hex

**THE MICROSTATE TIER'S SHAPE, FROM OFFICES INSTEAD OF THE YEARBOOK.** sources/micro.py reads
UNSD table 28; these five are either not in it (St Kitts and Nevis), newer than it (Falklands
2016 against 2006), finer than it (British Virgin Islands by island), or labelled better than
it (Aruba, whose UNSD `Pagan` is the census's `No religion`). Curaçao is in no UNSD row at all.
Sint Maarten and Anguilla are fine in the Yearbook and live in sources/micro.py.

**THE TABLES ARE TRANSCRIBED, AND EVERY ONE IS CHECKED AGAINST ITSELF.** Four of the five are
PDFs or an HTML table, so the counts are typed in below from the page, and `check()` asserts
that every unit's rows sum to that unit's printed total, that the units sum to the printed
grand total, and the category count. Where a machine-readable copy is on disk (Curaçao's
workbook, St Kitts's HTML table) the transcription is compared to it cell by cell. Where UNSD
has a row for the same census (British Virgin Islands and Aruba, both 2010) it is compared too,
and the one disagreement is pinned: UNSD's BVI Muslim figure is 255 where Table 77 prints 266.

**TWO ARE DRAWN BELOW THE TERRITORY.** British Virgin Islands at island, Falklands at Stanley,
Camp and the Mount Pleasant Complex. The hexes are put on units by coordinates (`_unit_vg`,
`_unit_fk`), not by a boundary file, because no boundary file for either tier was needed:
the islands are separated by sea, and the two Falklands settlements are points. Three BVI
census islands are too small to carry a Kontur hex (Cooper Island 26 people, Great Camanoe 6,
people living on yachts 18) and countries.py draws them with Tortola.

**CURAÇAO IS ABOVE §11aa'S ~100k LINE AND IS STILL ONE POLYGON**, sources/terr.md §4.
"""

import csv
import gzip
import json
import os
import re
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import micro                                            # noqa: E402  Kontur paths and URL

RAW = os.path.join(ROOT, "data", "raw", "terr")
COLUMNS = micro.COLUMNS
UA = micro.UA

# ---------------------------------------------------------------------------------------
# The tables. Every figure below was read off the document named in `doc`, 2026-09-14.
# `units` is the column order; each row is one count per unit in that order.
# ---------------------------------------------------------------------------------------
TABLES = {
    "vg": dict(
        name="British Virgin Islands", year=2010, level="island",
        source_id="vg_census_2010_table77",
        doc=("Virgin Islands 2010 Population and Housing Census Report, Table 77, *What is "
             "your affiliation with a religion or faith by Island* (report p.60)"),
        url="https://unstats.un.org/unsd/demographic/sources/census/wphc/BVI/VGB-2016-09-08.pdf",
        file="VGB-2016-09-08.pdf",
        units=["Anegada", "Cooper Island", "Great Camanoe Island", "Jost Van Dyke",
               "Tortola", "Virgin Gorda", "Yachts"],
        rows={
            "Anglican":                    (10, 0, 0, 4, 2158, 502, 0),
            "Church of God":               (18, 0, 0, 69, 2183, 643, 0),
            "Evangelical":                 (7, 0, 0, 3, 156, 18, 0),
            "Methodist":                   (77, 2, 2, 116, 4216, 528, 0),
            "Moravian":                    (1, 0, 0, 3, 77, 11, 0),
            "New Testament Church of God": (11, 0, 0, 8, 1588, 317, 0),
            "Pentecostal":                 (23, 0, 0, 18, 1734, 517, 0),
            "Presbyterian":                (2, 0, 0, 0, 56, 10, 0),
            "Roman Catholic":              (50, 16, 4, 10, 2003, 409, 0),
            "Seventh Day Adventist":       (2, 0, 0, 0, 2182, 337, 2),
            "Jehovah Witness":             (8, 0, 0, 4, 607, 74, 0),
            "Baptist":                     (4, 2, 0, 2, 1967, 105, 0),
            "Bahai":                       (0, 0, 0, 0, 10, 0, 0),
            "Hindu":                       (1, 0, 0, 4, 454, 69, 0),
            "Judaism":                     (0, 0, 0, 0, 8, 3, 0),
            "Mormon":                      (0, 0, 0, 0, 65, 10, 0),
            "Muslim/Islam":                (6, 0, 0, 0, 256, 4, 0),
            "Rastafarian":                 (14, 0, 0, 0, 140, 25, 0),
            "Budhaism":                    (0, 0, 0, 0, 41, 2, 0),
            "Other affiliation":           (5, 2, 0, 7, 1072, 67, 0),
            "None/No Religion":            (19, 2, 0, 27, 1950, 216, 16),
            "Not Stated":                  (27, 2, 0, 23, 568, 63, 0),
        },
        totals=(285, 26, 6, 298, 23491, 3930, 18), grand=28_054, slack=0,
    ),
    "fk": dict(
        name="Falkland Islands", year=2016, level="location",
        source_id="fk_census_2016_table6",
        doc=("Falkland Islands Census 2016 Full Report, Table 6, *Population by religion, sex "
             "and location* (PDF p.73)"),
        url=("https://www.falklands.gov.fk/policy/jdownloads/Reports%20&%20Publications/"
             "Census%20and%20Statistical%20Reports/Falkland_Islands_Census_2016_-_Full_Report.pdf"),
        file="Falkland_Islands_Census_2016_-_Full_Report.pdf",
        units=["Stanley", "Camp", "Mount Pleasant Complex"],
        rows={
            "Baha'i":            (3, 0, 0),
            "Buddhist":          (8, 0, 0),
            "Christian":         (1367, 191, 267),
            "Jehovah's Witness": (12, 0, 1),
            "Muslim":            (6, 0, 1),
            "No Religion":       (882, 160, 89),
            "Not Specified":     (162, 29, 1),
            "Other":             (18, 1, 0),
        },
        totals=(2458, 381, 359), grand=3_198, slack=0,
    ),
    "kn": dict(
        name="Saint Kitts and Nevis", year=2011, level="country",
        source_id="kn_stats_religion_2011",
        doc="Department of Statistics, *Population by Religious Belief, 2011* (web table)",
        url=("https://www.stats.gov.kn/topics/demographic-social-statistics/population/"
             "population-by-religious-belief-2011/"),
        file="kn_population-by-religious-belief-2011.html",
        units=["Saint Kitts and Nevis"],
        rows={
            "Anglican": (7842,), "Baptist": (2564,), "Bahai": (33,), "Brethren": (801,),
            "Church of God": (3495,), "Evangelical": (975,), "Hindu": (860,),
            "Jehovah Witness": (661,), "Methodist": (7447,), "Moravian": (2265,),
            "Muslim": (244,), "Pentecostal": (5081,), "Presbyterian": (159,),
            "Rastafarian": (608,), "Roman Catholic": (2801,), "Salvation Army": (55,),
            "Seventh Day Adventist": (2554,), "Wesleyan Holiness": (2506,), "None": (4141,),
            "Other": (2047,), "Not stated": (56,),
        },
        totals=(47195,), grand=47_195, slack=0,
    ),
    "aw": dict(
        name="Aruba", year=2010, level="country",
        source_id="aw_census_2010_tablePA5",
        doc=("Fifth Population and Housing Census Aruba 2010, Table P-A.5, *Population by "
             "religion, age and sex* (report p.82)"),
        # `id_` asks the Wayback Machine for the archived bytes; without it urllib gets a
        # 9 KB redirect page and saves that as the PDF (happened 2026-09-14).
        url=("https://web.archive.org/web/2016id_/http://cbs.aw/wp/wp-content/uploads/2012/07/"
             "Fifth-Population-and-Housing-Census-Aruba.pdf"),
        file="Fifth-Population-and-Housing-Census-Aruba.pdf",
        units=["Aruba"],
        rows={
            "Roman Catholic": (76464,), "Protestant": (2698,), "Jehovah's witness": (1703,),
            "Methodist": (932,), "Adventist": (880,), "Anglican": (450,), "Jewish": (354,),
            "No religion": (5625,), "Other": (11862,), "Not reported": (515,),
        },
        # The ten printed rows sum to 101,483 and the printed total is 101,484. In the census
        # report, and UNSD's row carries the same ten numbers; allowed here and nowhere else.
        totals=(101484,), grand=101_484, slack=1,
    ),
    "cw": dict(
        name="Curaçao", year=2023, level="country",
        source_id="cw_census_2023_tableD5",
        doc="CBS Curaçao, Census 2023, Table D-5, *Population by religion, age group and sex*",
        url=("https://cuatro.sim-cdn.nl/sensocbs/uploads/"
             "table-d-5.-population-by-religion-age-group-and-sex.xlsx"),
        file="cw_table-d-5_2023.xlsx",
        units=["Curaçao"],
        rows={
            "Adventist": (4694,), "Anglican": (198,), "Evangelical": (2983,),
            "Hinduism": (1211,), "Islam": (635,), "Jehovah's Witness": (3184,),
            "Judaism": (265,), "Methodist": (597,), "Pentecostal Church": (8524,),
            "Protestant": (4096,), "Roman Catholic": (106309,),
            "I don't have a religion": (13634,), "Other": (5408,), "Not reported": (4088,),
        },
        totals=(155826,), grand=155_826, slack=0,
    ),
}

# UNSD table 28 rows for the same census, and the disagreements that are known and pinned.
# Keyed census label -> UNSD label where they differ.
ORACLE = {
    "vg": ("British Virgin Islands", 2010,
           {"Muslim/Islam": "Muslim", "Budhaism": "Buddhist", "Bahai": "Baha'i",
            "None/No Religion": "No Religion", "Other affiliation": "Other"},
           {"Muslim/Islam": (266, 255)}),
    "aw": ("Aruba", 2010,
           {"Jehovah's witness": "Jehovah Witness", "No religion": "Pagan",
            "Other": "Other Religions and Persuasions", "Not reported": "Not Stated"},
           {}),
}


def _slug(s):
    return re.sub(r"[^A-Z0-9]+", "-", s.upper()).strip("-")


def geo_id(cc, unit):
    t = TABLES[cc]
    return cc.upper() if len(t["units"]) == 1 else f"{cc.upper()}-{_slug(unit)}"


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  have {os.path.basename(dest)}")
        return
    body = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                  timeout=300, context=_ctx()).read()
    magic = {".pdf": b"%PDF", ".xlsx": b"PK", ".gz": b"\x1f\x8b"}
    want = next((m for ext, m in magic.items() if dest.lower().endswith(ext)), None)
    if want and not body.startswith(want):
        raise RuntimeError(f"{os.path.basename(dest)}: got {len(body):,} bytes that are not a "
                           f"{os.path.splitext(dest)[1]} file (starts {body[:16]!r})")
    with open(dest + ".part", "wb") as fh:
        fh.write(body)
    os.replace(dest + ".part", dest)
    print(f"  got {os.path.basename(dest)} ({len(body):,} bytes)")


def fetch(ccs):
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(micro.RAW, exist_ok=True)
    for cc in ccs:
        t = TABLES[cc]
        try:
            _get(t["url"], os.path.join(RAW, t["file"]))
        except Exception as e:                # the record copy; the transcription stands alone
            print(f"  !! {cc}: could not fetch {t['url']}: {e}")
        gz, gpkg = micro._kontur_paths(cc)
        _get(micro.KONTUR_URL.format(CC=cc.upper()), gz)


def check(cc):
    """Assert the transcription against itself, then against any second copy on disk."""
    t = TABLES[cc]
    n = len(t["units"])
    bad = [k for k, v in t["rows"].items() if len(v) != n]
    if bad:
        raise SystemExit(f"{cc}: rows with the wrong number of units: {bad}")
    for j, u in enumerate(t["units"]):
        s = sum(v[j] for v in t["rows"].values())
        if abs(s - t["totals"][j]) > t["slack"]:
            raise SystemExit(f"{cc}: {u} rows sum to {s:,}, printed total {t['totals'][j]:,}")
    if abs(sum(t["totals"]) - t["grand"]) > t["slack"]:
        raise SystemExit(f"{cc}: units sum to {sum(t['totals']):,}, grand total {t['grand']:,}")

    national = {k: sum(v) for k, v in t["rows"].items()}
    path = os.path.join(RAW, t["file"])
    if cc == "kn" and os.path.exists(path):
        html = open(path, encoding="utf-8", errors="replace").read()
        cells = {}
        for col, row, val in re.findall(
                r'data-cell-id="([A-D])(\d+)"[^>]*?data-original-value="([^"]*)"', html):
            cells.setdefault(int(row), {})[col] = val.strip()
        web = {r["A"]: int(r["D"].replace(",", "")) for r in cells.values()
               if "A" in r and "D" in r and r["D"].replace(",", "").isdigit()}
        web.pop("Total", None)
        if web != national:
            raise SystemExit(f"kn: transcription differs from the saved HTML table: "
                             f"{sorted(set(web.items()) ^ set(national.items()))}")
        print("  kn: transcription equals the office's HTML table, all 21 rows")
    if cc == "cw" and os.path.exists(path):
        import openpyxl
        ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
        book = {}
        for row in ws.iter_rows(values_only=True):
            label, total = row[1], row[14] if len(row) > 14 else None
            if isinstance(label, str) and isinstance(total, (int, float)) and label != "Total":
                book[label.strip()] = int(total)
        if book != national:
            raise SystemExit(f"cw: transcription differs from the workbook: "
                             f"{sorted(set(book.items()) ^ set(national.items()))}")
        print("  cw: transcription equals CBS Curaçao's workbook, all 14 rows")

    if cc in ORACLE:
        import oracle
        name, year, relabel, known = ORACLE[cc]
        got = oracle.oracle(name, year)
        if not got or oracle.TOTAL not in got:
            print(f"  !! {cc}: UNSD has no {name} {year} row to compare")
        else:
            cats, _stated, _exact = oracle.partition(got[oracle.TOTAL])
            diff = {}
            for k, v in national.items():
                u = cats.get(relabel.get(k, k))
                if u != v:
                    diff[k] = (v, u)
            if diff != known:
                raise SystemExit(f"{cc}: UNSD {year} disagrees differently than pinned: {diff}")
            print(f"  {cc}: UNSD {year} agrees on every row"
                  + (f" except the pinned {known}" if known else ""))
    return national


def normalise(cc):
    t = TABLES[cc]
    check(cc)
    rows = []
    for cat, vals in t["rows"].items():
        for u, n in zip(t["units"], vals):
            rows.append(dict(geo_id=geo_id(cc, u), geo_level=t["level"], geo_name=u,
                             source_category=cat, count=n, basis="self_id", year=t["year"],
                             source_id=t["source_id"], note=t["doc"]))
    out = os.path.join(ROOT, "data", "normalized", f"{cc}.csv")
    with open(out + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(out + ".part", out)
    print(f"  {cc}  {t['name']:<24}{t['year']}  {len(t['rows']):>3} cats  "
          f"{len(t['units'])} unit(s)  {t['grand']:>8,}")


# ---- placement ----------------------------------------------------------------------------
def _km(lon, lat, lon0, lat0):
    import numpy as np
    dx = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    dy = (lat - lat0) * 110.574
    return np.hypot(dx, dy)


def _unit_vg(lon, lat):
    """Four drawn islands by coordinates. Tortola also takes Beef, Great Camanoe, Scrub,
    Guana, Peter, Norman and Cooper islands, which is where their few residents are drawn;
    Virgin Gorda takes Mosquito, Necker and the Dogs."""
    import numpy as np
    out = np.full(len(lon), "VG-TORTOLA", dtype=object)
    out[lon >= -64.47] = "VG-VIRGIN-GORDA"
    out[(lon <= -64.705) & (lat >= 18.43)] = "VG-JOST-VAN-DYKE"
    out[lat >= 18.65] = "VG-ANEGADA"
    return out


STANLEY = (-57.86, -51.695)
MOUNT_PLEASANT = (-58.45, -51.82)


def _unit_fk(lon, lat):
    """Stanley and the Mount Pleasant Complex as 5 km circles; everything else is Camp, which
    is the census's own word for everywhere outside Stanley."""
    import numpy as np
    out = np.full(len(lon), "FK-CAMP", dtype=object)
    out[_km(lon, lat, *MOUNT_PLEASANT) <= 5.0] = "FK-MOUNT-PLEASANT-COMPLEX"
    out[_km(lon, lat, *STANLEY) <= 5.0] = "FK-STANLEY"
    return out


# The units each layer must carry, after countries.py's fold of the three hex-less BVI islands.
DRAWN = {
    "vg": {"VG-ANEGADA", "VG-JOST-VAN-DYKE", "VG-TORTOLA", "VG-VIRGIN-GORDA"},
    "fk": {"FK-STANLEY", "FK-CAMP", "FK-MOUNT-PLEASANT-COMPLEX"},
}
FOLD = {"vg": {"VG-COOPER-ISLAND": "VG-TORTOLA", "VG-GREAT-CAMANOE-ISLAND": "VG-TORTOLA",
               "VG-YACHTS": "VG-TORTOLA"}}


def geometry(cc):
    import geopandas as gpd
    import shapely

    gz, gpkg = micro._kontur_paths(cc)
    if not os.path.exists(gpkg) or os.path.getsize(gpkg) == 0:
        if not os.path.exists(gz):
            raise SystemExit(f"{cc}: missing {gz}; run with --fetch")
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    hexes = gpd.read_file(gpkg)
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    hexes = hexes.rename(columns={popcol: "pop"})
    hexes = hexes[hexes["pop"] > 0].copy().to_crs("EPSG:4326")
    if hexes.empty:
        raise SystemExit(f"{cc}: Kontur has no populated hexes")
    c = shapely.centroid(hexes.geometry.values)
    lon, lat = shapely.get_x(c), shapely.get_y(c)
    if cc == "vg":
        hexes["unit"] = _unit_vg(lon, lat)
    elif cc == "fk":
        hexes["unit"] = _unit_fk(lon, lat)
    else:
        hexes["unit"] = cc.upper()
    hexes = hexes.reset_index(drop=True)
    hexes["cellcode"] = cc.upper() + ":" + hexes.index.astype(str)

    want = DRAWN.get(cc, {cc.upper()})
    have = set(hexes["unit"])
    if have != want:
        raise SystemExit(f"{cc}: hex units {sorted(have)} != expected {sorted(want)}")

    t = TABLES[cc]
    census = {}
    for u, n in zip(t["units"], t["totals"]):
        g = FOLD.get(cc, {}).get(geo_id(cc, u), geo_id(cc, u))
        census[g] = census.get(g, 0) + n
    for u in sorted(want):
        sub = hexes[hexes["unit"] == u]
        k = float(sub["pop"].sum())
        print(f"      {u:<28}{len(sub):>6,} hexes, Kontur {k:>9,.0f} vs census "
              f"{census[u]:>8,} (ratio {k / census[u]:.2f})")

    geo = os.path.join(ROOT, "data", "geo", cc)
    os.makedirs(geo, exist_ok=True)
    out = os.path.join(geo, f"{cc}_hexes.gpkg")
    hexes[["cellcode", "unit", "pop", "geometry"]].to_file(out, layer="hexes", driver="GPKG")


def main():
    args = [a.lower() for a in sys.argv[1:] if not a.startswith("--")]
    ccs = args or list(TABLES)
    bad = [c for c in ccs if c not in TABLES]
    if bad:
        raise SystemExit(f"not in this module: {bad}; have {sorted(TABLES)}")
    if "--fetch" in sys.argv:
        fetch(ccs)
    for cc in ccs:
        normalise(cc)
        geometry(cc)


if __name__ == "__main__":
    main()
