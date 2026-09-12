"""Jamaica — 2011 Population and Housing Census, religion by parish.

Reads (or fetches) data/raw/jm/ and writes data/normalized/jm.csv.

**The best denominational detail in the Americas outside the United States, on the coarsest
geography this project has drawn.** 2,678,981 people on **14 parishes**, **19 religion
categories**, from the U.S. Census Bureau's transcription of STATIN's own tabulation —
the same series and the same shape as Ethiopia (§9u), Pakistan (§9t) and Bangladesh (§9v),
with the boundaries in the same file. `sources.md` §11h is the route and §11j the audit.

**WHY IT IS DRAWN AT ALL, GIVEN 14 UNITS.** It was not, for a day: §11j left it as *"a
category source on a geography that fails the floor"*, weighing 14 parishes against a
12-unit rejection invented in §11d. **spec §3.9b withdrew that floor** — Anita, 2026-09-06,
*"especially for smaller countries, we dont need that many regions for it to be a cool
plot"* — and the arithmetic was always on Jamaica's side anyway: 14 parishes over 2.7M is
**191,356 people per unit**, finer than Georgia's 11 regions at 340,000, and Georgia was
already drawn.

**WHAT THE CATEGORIES BUY, AND IT IS THE WHOLE REASON FOR THE COUNTRY.** Nineteen cells,
and three of them exist nowhere else on this map:

  * **Rastafarian — 29,026 people.** `rastafari` has been a root in `branches.py` since
    Czechia arrived with **190** of them. This is the religion's home, and it is the first
    source here that counts it where it was founded.
  * **Revivalist — 36,296 people.** Revival Zion and Pukkumina/Pocomania, the Afro-Jamaican
    revival tradition out of the Great Revival of 1860-61. It needed a new node.
  * **Four Church of God bodies kept apart** — in Jamaica, Prophecy, New Testament and
    Other — which together are **689,868 people, 25.7% of the country** and the largest
    religious bloc in it.

**THE PARISH TABLES ARE MISSING FOUR RELIGIONS AND ONLY THE METADATA SHEET SAYS SO.** This
is the country §11h's read-the-metadata-first rule was written for. STATIN's national table
counts 2,683,105 people; the 19 named columns sum to **2,678,981**, and the 4,124-person
difference is *"Baha'i, Hinduism, Islam and Judaism were not included in parish tables by
the Statistical Institute"* — **269 Bahá'í, 1,836 Hindu, 1,513 Muslim, 506 Jewish**.

They are **absent, not pooled**: every parish's 19 cells sum to its own `RLG_RTOTL` exactly,
and the 14 parishes sum to the national row exactly on all 19 categories. The gap appears
only in the national `RLG_RTOTL`. So a country built from the data sheet alone, without the
metadata, would assert that **Jamaica has no Muslims, Hindus, Bahá'ís or Jews at all** — and
every reconciliation it ran would pass. `check()` asserts the gap is exactly 4,124 rather
than tolerating it, so a re-release that starts including them fails here instead of quietly
changing what the map claims.

Those 4,124 people are not drawn (spec §3.5) and `countries.py` says so on the map.

**THE ADM2 TIER EXISTS AND THE RELIGION TABLE DOES NOT REACH IT.** The geodatabase ships
`JM_GEOG1_ADM2_2011` — STATIN's "Special Areas", built from 248 original shapefiles — and
§9p's lesson is that an extra level can hide inside the finest one, so it was checked. It
does not: the *Ethnicity and Religion* sheet has 15 rows, one country and 14 parishes, and
every other USCB table for Jamaica is `GEOG1` at ADM1 too. The Special Areas are a placement
option, not a counting tier, and Kontur is the better one (`sources/jm_geo.md`).

Usage:
    python sources/jm.py --fetch    two GETs, ~1.9 MB
    python sources/jm.py            normalise from data/raw/jm/
"""

import csv
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "jm")
OUT = os.path.join(ROOT, "data", "normalized", "jm.csv")

SOURCE_ID = "jm_phc_2011_uscb"
YEAR = 2011
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

HDX = "https://data.humdata.org/dataset/cd75b1bc-ec9a-4ab3-9f7a-b36a0ab71626/resource"
URL_GDB = f"{HDX}/7c6fda17-7713-4d79-a6c5-95361ff5422c/download/jamaica.gdb.zip"
URL_XLSX = f"{HDX}/2a0c52bb-9a55-44fa-b4d2-68966bfcc85d/download/jamaica_uscb_202302.xlsx"

ZIP = os.path.join(RAW, "jamaica.gdb.zip")
XLSX = os.path.join(RAW, "jamaica_uscb_202302.xlsx")
GDB = os.path.join(RAW, "Jamaica.gdb")

SHEET = "Ethnicity and Religion"
LAYER_RELIGION = "JM_ETHNICITY_AND_RELIGION_GEOG1_2011census_uscb_202302"

# STATIN's own national religion universe, which INCLUDES the four religions the parish
# tables omit. The one figure here that is not the sum of something else in the file.
NATIONAL_UNIVERSE = 2_683_105
# What the 19 published columns actually hold, and what this map draws.
NATIONAL_DRAWN = 2_678_981
# The difference, named. Metadata sheet: Baha'i 269, Hinduism 1,836, Islam 1,513,
# Judaism 506. Asserted exactly — see the module docstring.
OMITTED = 4_124
OMITTED_DETAIL = "269 Bahá'í, 1,836 Hindu, 1,513 Muslim, 506 Jewish"

CATEGORIES = [
    ("RLG_ANG", "Anglican"),
    ("RLG_BAP", "Baptist"),
    ("RLG_BRE", "Brethren"),
    ("RLG_CGJA", "Church of God in Jamaica"),
    ("RLG_CGPR", "Church of God of Prophecy"),
    ("RLG_CGNT", "New Testament Church of God"),
    ("RLG_CGOT", "Other Church of God"),
    ("RLG_JEH", "Jehovah's Witness"),
    ("RLG_MET", "Methodist"),
    ("RLG_MOR", "Moravian"),
    ("RLG_PEN", "Pentecostal"),
    ("RLG_RAS", "Rastafarian"),
    ("RLG_REV", "Revivalist"),
    ("RLG_ROM", "Roman Catholic"),
    ("RLG_SDA", "Seventh Day Adventist"),
    ("RLG_UC", "United Church"),
    ("RLG_OTHR", "Other religion"),
    ("RLG_NR", "Non-religious"),
    ("RLG_ND", "No Data"),
]
TOTAL_COL = "RLG_RTOTL"

LEVELS = {0: "country", 1: "parish"}
ROWS_EXPECTED = 15            # 1 country + 14 parishes
PARISHES = 14


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
    for url, dest, least in ((URL_GDB, ZIP, 1_000_000), (URL_XLSX, XLSX, 300_000)):
        if os.path.exists(dest) and os.path.getsize(dest) > least:
            print("already have", dest)
            continue
        r = requests.get(url, timeout=900, headers=ua)
        r.raise_for_status()
        # §5a: a 200 is not a download, and §11d: check the magic bytes, not the extension.
        if r.content[:4] != b"PK\x03\x04":
            raise SystemExit(f"{dest}: starts {r.content[:16]!r}, expected a zip container")
        if len(r.content) < least:
            raise SystemExit(f"{dest}: only {len(r.content):,} bytes")
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"wrote {dest} ({len(r.content):,} bytes)")

    if not os.path.isdir(GDB):
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(RAW)
        print("unzipped", GDB)


def read():
    """The xlsx is the read; the gdb is the cross-check. Returns (rows, frame, defects)."""
    import pandas as pd

    if not os.path.exists(XLSX):
        raise SystemExit(f"missing {XLSX} -- run: python sources/jm.py --fetch")
    # Row 0 of the body is a description row ("Total population (religion)", …), not data.
    x = pd.read_excel(XLSX, sheet_name=SHEET, header=0, skiprows=[1])
    if len(x) != ROWS_EXPECTED:
        raise SystemExit(f"expected {ROWS_EXPECTED} rows in '{SHEET}', got {len(x)}")

    cols = [k for k, _ in CATEGORIES] + [TOTAL_COL]
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    negatives = int(sum((x[c] < 0).sum() for c in cols))
    nulls = int(sum(x[c].isna().sum() for c in cols))

    x["level"] = x["ADM_LEVEL"].astype(int)
    x["SUM"] = x[[k for k, _ in CATEGORIES]].sum(axis=1, min_count=1)

    rows = []
    for r in x.itertuples(index=False):
        d = r._asdict()
        lv = int(d["ADM_LEVEL"])
        name = str(d["AREA_NAME"]).strip()
        note = [f"level={LEVELS[lv]}"]
        for key in ("NSO_CODE", "GENC_CODE"):
            v = d.get(key)
            if pd.notna(v) and str(v).strip():
                note.append(f"{key.split('_')[0].lower()}={str(v).strip()}")
        cmnt = d.get("USCBCMNT")
        if isinstance(cmnt, str) and cmnt.strip():
            note.append(f"uscb={' '.join(cmnt.split())}")
        for key, label in CATEGORIES:
            v = d[key]
            if pd.isna(v):
                continue
            rows.append({
                "geo_id": d["GEO_MATCH"], "geo_level": LEVELS[lv], "geo_name": name,
                "source_category": label, "count": int(v), "basis": BASIS,
                "year": YEAR, "source_id": SOURCE_ID, "note": "; ".join(note),
            })
    return rows, x, (negatives, nulls)


def check(x, defects):
    import pandas as pd

    negatives, nulls = defects
    ok = True
    print("Jamaica — 2011 census religion, USCB tabulation on 2011 parish boundaries\n")

    good = negatives == 0 and nulls == 0
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {negatives} negative and {nulls} null cells in the "
          f"religion columns (expected 0 and 0 — this file uses neither Ethiopia's -999 "
          f"sentinel nor Pakistan's real nulls)")

    nat = x.loc[x["level"] == 0]
    if len(nat) != 1:
        raise SystemExit("no country row")
    nat = nat.iloc[0]

    # 1. THE CAVEAT, ASSERTED RATHER THAN TOLERATED. The national universe exceeds the sum
    #    of its own published parts by exactly the four religions STATIN left out of the
    #    parish tables. A total that does not match its parts is the tell (§11j); this
    #    turns it into a test.
    gap = int(nat[TOTAL_COL]) - int(nat["SUM"])
    good = (int(nat[TOTAL_COL]) == NATIONAL_UNIVERSE
            and int(nat["SUM"]) == NATIONAL_DRAWN and gap == OMITTED)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national universe {int(nat[TOTAL_COL]):,} minus the "
          f"19 published columns {int(nat['SUM']):,} = {gap:,}, expected exactly "
          f"{OMITTED:,}\n        ({OMITTED_DETAIL} — absent from the parish tables, "
          f"Metadata sheet)")

    # 2. every PARISH's 19 cells sum to its own published total, exactly. The omission is
    #    national-only: the parish rows are internally complete.
    par = x[x["level"] == 1].copy()
    d = (par["SUM"] - par[TOTAL_COL]).abs()
    bad = int((d > 0).sum())
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} all {len(par)} parishes' 19 categories sum to "
          f"their own {TOTAL_COL} ({bad} failures) — so the four missing religions are "
          f"ABSENT from the parish tables, not pooled into `Other`")

    # 3. the parishes sum to the national row, category by category
    bad = []
    for key, label in CATEGORIES:
        s, n = int(par[key].sum()), int(nat[key])
        if s != n:
            bad.append((label, s, n))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the {len(par)} parishes sum to the national "
          f"figure on all {len(CATEGORIES)} categories ({len(bad)} failures)")
    for label, s, n in bad:
        print(f"        {label}: {s:,} vs {n:,}")

    # 4. the drawn tier is a clean key with nobody empty
    good = par["GEO_MATCH"].nunique() == len(par) == PARISHES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(par)} parish rows, "
          f"{par['GEO_MATCH'].nunique()} distinct GEO_MATCH (expected {PARISHES} of each)")
    empty = int((par[TOTAL_COL] <= 0).sum())
    ok &= not empty
    print(f"  {'OK ' if not empty else 'BAD'} {empty} parishes with no people (expected 0)")

    # 5. workbook against geodatabase. NOT independent — same publisher, same release — but
    #    it catches a misread of either file.
    try:
        import pyogrio
        g = pyogrio.read_dataframe(GDB, layer=LAYER_RELIGION, read_geometry=False)
        m = x[["GEO_MATCH"]].merge(g, on="GEO_MATCH", how="left", suffixes=("", "_g"))
        diff = 0
        for key, _ in CATEGORIES:
            a = pd.to_numeric(x[key], errors="coerce").fillna(-1).to_numpy(dtype=float)
            b = pd.to_numeric(m[key], errors="coerce").fillna(-1).to_numpy(dtype=float)
            diff += int((a != b).sum())
        ok &= not diff
        print(f"  {'OK ' if not diff else 'BAD'} the geodatabase layer agrees with the "
              f"workbook on every cell ({diff} differences)")
    except Exception as e:
        print(f"  --  gdb cross-check skipped: {type(e).__name__}: {e}")

    if not ok:
        raise SystemExit("checks failed")
    return ok


def normalise(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}  {len(rows):,} rows")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, x, defects = read()
    check(x, defects)
    normalise(rows)

    par = x[x["level"] == 1]
    tot = int(par[TOTAL_COL].sum())
    print(f"\n  {tot:,} people drawn on {len(par)} parishes, "
          f"{OMITTED:,} not drawn ({OMITTED_DETAIL})")
    nat = x[x["level"] == 0].iloc[0]
    for key, label in sorted(CATEGORIES, key=lambda kv: -int(nat[kv[0]]))[:6]:
        print(f"    {label:<30} {int(nat[key]):>9,}  {int(nat[key]) / tot:>6.2%}")


if __name__ == "__main__":
    main()
