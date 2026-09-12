"""Samoa — Samoa Bureau of Statistics, 2021 Census, Table 2.

Reads (or fetches) data/raw/ws/ and writes data/normalized/ws.csv.

**THE CLEANEST TABLE ON THIS MAP.** `Table 2. Total population by sex, religion and place of
residence, 2021` is 401 x 82: **26 named religions and a total, on four nested tiers**, and it
closes everywhere. The 26 categories sum to the printed total at **every one of its 395 place
rows**, with no tolerance; the 339 villages sum to their 51 districts, the districts to their 4
regions, and the regions to the country. Nothing else here reconciles on all of that at once.
There is no `Not stated` row at all, and the only residual is a named `OTHER CHURCHES` at 1.9%.

**THE TIERS ARE INDENTATION IN COLUMN A AND NOTHING ELSE** — §9p/§9af/§9az's pattern in a fourth
country, here with four levels rather than two: 0 = Samoa, 4 = **4 statistical regions**,
8 = **51 districts**, 12 = **339 villages**, about 606 people each.

**WHAT IS DRAWN IS COARSER THAN WHAT IS READ, AND THE REASON IS GEOMETRY, NOT DATA.** This
module writes all 339 villages. `sources/ws_geo.py` then aggregates them to **25 traditional
districts**, because no polygon layer exists for the census's own 51 districts or its 339
villages, and the only real layer for Samoa (43 electoral districts, one Pacific Data Hub
lineage behind both geoBoundaries and GADM) is a *different cut* of the same 25. Keeping the
villages here means that if a 51 or 339 layer ever appears, only `ws_geo.py` changes.

**LATTER DAY SAINTS ARE 17.6%**, second only to Tonga's 19.7% among the censuses in UNSD table
28 that count them separately. **`ASO FITU (SISDAC)`, 1,962 people**, is the Samoa Independent
Seventh Day Adventist Church, a local schism that no other census on earth counts.

Usage:
    python sources/ws.py --fetch    one ~2.7 MB workbook from sbs.gov.ws
    python sources/ws.py            normalise from data/raw/ws/
"""

import csv
import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

RAW = os.path.join(ROOT, "data", "raw", "ws")
OUT = os.path.join(ROOT, "data", "normalized", "ws.csv")

SOURCE_ID = "ws_phc_2021"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# One GET, no wall, no plugin. SBS puts the whole census table set in a single workbook.
XLSX_URL = "https://www.sbs.gov.ws/wp-content/uploads/2022/12/CensusTablesEXCELFiles.xlsx"
XLSX_NAME = "CensusTablesEXCELFiles.xlsx"
SHEET = "Table 2"

NATIONAL = 205_557
EXPECTED_VILLAGES = 339
EXPECTED_DISTRICTS = 51
EXPECTED_REGIONS = 4
EXPECTED_CATEGORIES = 26

# Column A's indentation IS the tier. Four levels, four spaces apart.
TIER = {0: "country", 4: "region", 8: "district", 12: "village"}

RESIDUALS = []          # `OTHER CHURCHES` is a named category, not a residual: see ws2021.py


def fold(s):
    """Casefold, strip accents and punctuation — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("‘", "'").replace("’", "'").replace("`", "'")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, XLSX_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 2_000_000:
        print("already have", dest)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
    print("GET", XLSX_URL)
    r = requests.get(XLSX_URL, headers=ua, timeout=900)
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        raise SystemExit(f"sbs.gov.ws returned something that is not a workbook "
                         f"({len(r.content):,} bytes)")
    tmp = dest + ".part"                                  # [[reference_wb_truncates]]
    with open(tmp, "wb") as fh:
        fh.write(r.content)
    os.replace(tmp, dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def _columns(rows):
    """The (index, name) of every `Total` column, from the two stacked header rows.

    Row 1 carries a merged religion name over three columns and row 2 carries
    `Total | MALE | FEMALE` under it, so the religion name only appears above the first of
    the three. Anchoring on row 2's `Total` and carrying row 1's last non-blank name forward
    is what picks the 26 without hard-coding a stride: a table that gained a `Not stated`
    column would still be read correctly rather than silently shifted.
    """
    h1, h2 = rows[1], rows[2]
    out, cur = [], None
    for i in range(len(h1)):
        if h1[i] not in (None, ""):
            cur = " ".join(str(h1[i]).split())
        sub = "" if h2[i] is None else str(h2[i]).strip()
        if sub.lower() == "total":
            if cur is None:
                raise SystemExit(f"column {i} has a `Total` with no religion above it")
            out.append((i, cur))
    return out


def parse():
    """Table 2 -> (categories, rows), with the four-tier nesting asserted.

    `rows` is [(tier, name, [total, *counts], region, district)] in print order.
    """
    import openpyxl

    path = os.path.join(RAW, XLSX_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing — run `python sources/ws.py --fetch`")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    if SHEET not in wb.sheetnames:
        raise SystemExit(f"{XLSX_NAME} has no sheet {SHEET!r}; sheets are {wb.sheetnames}")
    rows = list(wb[SHEET].iter_rows(values_only=True))

    cols = _columns(rows)
    if not cols or cols[0][1].upper() != "TOTAL":
        raise SystemExit(f"first Total column is {cols[0][1]!r}, expected the universe")
    cats = [name for _, name in cols[1:]]
    if len(cats) != EXPECTED_CATEGORIES:
        raise SystemExit(f"{len(cats)} religion columns, expected {EXPECTED_CATEGORIES}: {cats}")

    data = []
    for r in rows:
        a = r[0]
        if not isinstance(a, str) or not a.strip():
            continue
        if not isinstance(r[1], (int, float)):
            continue
        ind = len(a) - len(a.lstrip(" "))
        if ind not in TIER:
            raise SystemExit(f"row {a.strip()!r} is indented {ind}, which is not a tier")
        data.append((ind, a.strip(), [int(r[i]) for i, _ in cols]))

    # --- the categories partition, at EVERY row and not only nationally.
    for ind, nm, vals in data:
        if sum(vals[1:]) != vals[0]:
            raise SystemExit(f"{TIER[ind]} {nm!r}: categories sum to {sum(vals[1:])}, "
                             f"total says {vals[0]}")

    # --- the four tiers nest, on all 27 columns.
    out, i = [], 0
    if data[0][0] != 0:
        raise SystemExit("Table 2 does not open with the national row")
    national = data[0][2]
    out.append((0, data[0][1], national, None, None))
    i = 1
    nreg = ndist = nvill = 0
    rsum = [0] * len(cols)
    while i < len(data):
        if data[i][0] != 4:
            raise SystemExit(f"expected a region at row {i}, got {data[i][1]!r}")
        region, rvals = data[i][1], data[i][2]
        out.append((4, region, rvals, region, None))
        nreg += 1
        i += 1
        dsum = [0] * len(cols)
        while i < len(data) and data[i][0] == 8:
            district, dvals = data[i][1], data[i][2]
            out.append((8, district, dvals, region, district))
            ndist += 1
            i += 1
            vsum = [0] * len(cols)
            while i < len(data) and data[i][0] == 12:
                out.append((12, data[i][1], data[i][2], region, district))
                for k in range(len(cols)):
                    vsum[k] += data[i][2][k]
                nvill += 1
                i += 1
            if vsum != dvals:
                raise SystemExit(f"district {district!r}: villages sum to {vsum[0]}, "
                                 f"the district row says {dvals[0]}")
            for k in range(len(cols)):
                dsum[k] += dvals[k]
        if dsum != rvals:
            raise SystemExit(f"region {region!r}: districts sum to {dsum[0]}, "
                             f"the region row says {rvals[0]}")
        for k in range(len(cols)):
            rsum[k] += rvals[k]
    if rsum != national:
        raise SystemExit(f"regions sum to {rsum[0]}, the national row says {national[0]}")

    for got, want, what in ((nreg, EXPECTED_REGIONS, "regions"),
                            (ndist, EXPECTED_DISTRICTS, "districts"),
                            (nvill, EXPECTED_VILLAGES, "villages")):
        if got != want:
            raise SystemExit(f"{got} {what}, expected {want}")
    if national[0] != NATIONAL:
        raise SystemExit(f"national total {national[0]}, expected {NATIONAL}")

    print(f"  {nreg} regions, {ndist} districts, {nvill} villages")
    print(f"  partition: EXACT at all {len(data)} place rows; nesting: EXACT on all "
          f"{len(cols)} columns")
    return cats, out, national


# UNSD table 28's 2016 name -> the 2021 workbook's name for the same body. The Yearbook
# abbreviates (`Catholic`, `Jehovah Witness`) and the 2021 workbook renders three of them in
# Samoan instead of English, which is why this cannot be a string match: `Protestant` becomes
# `POROTESANO`, `Baptist` becomes `PABTISM` (the workbook's own spelling), and `Aoga Tusi
# Paia` — Samoan for Bible school — becomes `BIBLE STUDY`.
DYB2016 = {
    "Congregational": "CONGREGATIONAL CHRISTIAN CHURCH OF SAMOA",
    "Catholic": "ROMAN CATHOLIC",
    "Latter Day Saints": "LATTER DAY SAINTS",
    "Methodist": "METHODIST",
    "Assembly of God": "ASSEMBLY OF GOD",
    "Seventh Day Adventist": "SEVENTH DAYS ADVENTIST",
    "Worship Centre": "WORSHIP CENTRE",
    "Other": "OTHER CHURCHES",
    "Full Gospel Church": "FIRST FULL GOSPEL PENTECOSTAL CHURCH IN SAMOA",
    "Voice of Christ": "VOICE OF CHRIST",
    "Jehovah Witness": "JEHOVAHS WITNESS",
    "Nazarene": "NAZARENE",
    "Christian": "CHRISTIAN FELLOWSHIP",
    "Baha'i": "BAHAI",
    "Peace Chapel": "PEACE CHAPEL",
    "Baptist": "PABTISM",
    "CCCJS": "CONGREGATIONAL CHRISTIAN CHURCH OF JESUS IN SAMOA (EFIS)",
    "Aoga Tusi Paia": "BIBLE STUDY",
    "No Religion": "NO RELIGION",
    "Protestant": "POROTESANO",
    "Anglican": "ANGLICAN CHURCH",
    "Elim": "ELIM CHURCH",
    "Samoa Evangelism": "SAMOA EVANGELISM",
    "Muslim": "MUSLIM",
    "Not Stated": None,           # 2016 had one; 2021 has no such column at all
}


def check(cats, national):
    """The 2016 census, from UNSD table 28, as an outside look at the INSTRUMENT.

    The Yearbook has no 2021 row for Samoa, so this cannot check the 2021 numbers. What it
    checks is that the question is the same one: every 2016 category still has a 2021 column,
    so no church was quietly merged away, and the only additions are named.
    """
    try:
        import oracle
        got = oracle.oracle("Samoa", 2016)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  2016 comparison SKIPPED ({exc})")
        return
    if not got:
        print("  2016 comparison SKIPPED — no Samoa 2016 row")
        return
    old = [k for k in got.get(oracle.TOTAL, {}) if fold(k) != "total"]
    new = {fold(c): c for c in cats}
    unknown = [k for k in old if k not in DYB2016]
    if unknown:
        raise SystemExit(f"UNSD's 2016 list has categories this file does not know: "
                         f"{unknown} — the Yearbook changed, update DYB2016")
    lost = [k for k in old if DYB2016[k] and fold(DYB2016[k]) not in new]
    if lost:
        raise SystemExit(f"2016 categories with no 2021 column: {lost}")
    mapped = {fold(DYB2016[k]) for k in old if DYB2016[k]}
    added = sorted(new[f] for f in set(new) - mapped)
    dropped = [k for k in old if DYB2016[k] is None]
    print(f"  instrument vs 2016 (UNSD table 28): all {len(mapped)} of 2016's churches still "
          "have a column")
    print(f"    2021 adds: {added}")
    print(f"    2021 drops: {dropped} — Samoa stopped having a not-stated cell")


def normalise():
    cats, rows, national = parse()
    check(cats, national)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = []
    seen = set()
    for tier, name, vals, region, district in rows:
        if tier != 12:
            continue
        # Village names repeat across districts in Samoa (Vailoa, Siufaga, Fusi, Samata),
        # so the key is the pair. [[reference_name_join_wrong_neighbour]]
        geo_id = f"{fold(district).replace(' ', '')}/{fold(name).replace(' ', '')}"
        if geo_id in seen:
            raise SystemExit(f"two villages named {name!r} inside district {district!r}")
        seen.add(geo_id)
        for k, cat in enumerate(cats, start=1):
            if vals[k] == 0:
                continue
            out.append({
                "geo_id": geo_id,
                "geo_level": "village",
                "geo_name": name,
                "source_category": cat,
                "count": vals[k],
                "basis": BASIS,
                "year": YEAR,
                "source_id": SOURCE_ID,
                "note": f"{region}|{district}",
            })

    if len(seen) != EXPECTED_VILLAGES:
        raise SystemExit(f"{len(seen)} distinct geo_ids for {EXPECTED_VILLAGES} villages")

    tmp = OUT + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, OUT)

    total = sum(r["count"] for r in out)
    print(f"wrote {OUT}")
    print(f"  {len(out):,} rows, {len(seen)} villages, {total:,} people")
    if total != NATIONAL:
        raise SystemExit(f"the CSV holds {total:,} people, the census says {NATIONAL:,}")
    print(f"  drawn {total:,} (100.00%): Samoa has no `Not stated` row and no residual "
          "beyond a named OTHER CHURCHES")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        normalise()
