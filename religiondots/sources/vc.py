"""Saint Vincent and the Grenadines — 2012 census, religion by enumeration district.

Reads (or fetches) data/raw/vc/ and writes data/normalized/vc.csv.

**THE FINEST GEOGRAPHY IN THE PROJECT PER HEAD, AND THE SMALLEST COUNTRY ON THE MAP.**
109,188 people on **221 enumeration districts** — a median of **415 people and 0.66 km²
per unit** — with **18 religion categories**. The USCB series again (§11h), transcribing the
SVG Statistical Office's own REDATAM tabulation, with the boundaries in the same file.

For comparison: Sri Lanka's 14,003 GN divisions average 1,500 people, Ireland's Small Areas
about 350 but on 5 categories. **Nothing else here combines this grain with this many
categories**, and at 221 units over 389 km² the whole country fits inside one Bosnian
municipality.

§11j's verdict was *"build it as a companion to Jamaica or not at all"* — alone it is a
rounding error at 109 dots. Jamaica was built on 2026-09-06 (§9ab), so this is the companion.

**THE RECONCILIATION IS A CROSS-TABLE ONE AND IT IS STRONGER THAN JAMAICA'S.** This file has
**no religion total column** — there is no `RLG_RTOTL` the way Jamaica has one. What it has
instead is `ETH_TPOP`, the *ethnicity* universe, sitting in the same sheet. And:

    the 18 religion cells sum to ETH_TPOP EXACTLY, on all 235 rows, at every level.

That is two independently tabulated questions agreeing to the person on 221 enumeration
districts, which is a much better check than a column summing to its own neighbour. It also
establishes the universe: religion here is asked of **everybody**, not of a 15+ subset the way
Portugal's and Chile's are, so no share needs a denominator caveat.

The levels nest exactly too: 13 census divisions and 221 enumeration districts each sum to
the national row, category by category, with zero discrepancy.

**THE ADM1 TIER IS CENSUS DIVISIONS, NOT PARISHES, AND THE FILE WARNS ABOUT IT.** The
Metadata sheet: *"2012 census materials utilized a 13 census division ADM1 structure that
differs from the 6 parish ADM1 structure generally used in maps of Saint Vincent"*. A
`PARISH` column exists on the ADM2 layer for anyone who wants the six. Only ADM2 is drawn, so
this is a note rather than a decision.

**TWO ENUMERATION DISTRICTS HAVE NO PEOPLE.** 219 of 221 are populated; the other two return
zero across every column. They are kept in the normalised file as zero rows and simply draw
nothing — no unit is dropped, so a future vintage that populates them needs no change here.

**`Traditional` is 74 people and it is NOT the indigenous population** — see
`taxonomy/vc2012.py`, which tested that rather than assuming it.

Usage:
    python sources/vc.py --fetch    two GETs, ~1.8 MB
    python sources/vc.py            normalise from data/raw/vc/
"""

import csv
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "vc")
OUT = os.path.join(ROOT, "data", "normalized", "vc.csv")

SOURCE_ID = "vc_phc_2012_uscb"
YEAR = 2012
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

HDX = "https://data.humdata.org/dataset/571447eb-d68f-48bd-8383-f5481b70ab5c/resource"
URL_GDB = (f"{HDX}/3d70b0c6-7e6e-484f-a96c-954caf0067b0/download/"
           "saint_vincent_and_the_grenadines.gdb.zip")
URL_XLSX = (f"{HDX}/cd246881-5f19-4be4-8bc5-1ef7a17b4247/download/"
            "saint_vincent_and_the_grenadines_uscb_202109.xlsx")

ZIP = os.path.join(RAW, "saint_vincent_and_the_grenadines.gdb.zip")
XLSX = os.path.join(RAW, "saint_vincent_and_the_grenadines_uscb_202109.xlsx")
GDB = os.path.join(RAW, "Saint_Vincent_and_The_Grenadines.gdb")

SHEET = "Ethnicity and Religion"
LAYER_RELIGION = "VC_ETHNICITY_AND_RELIGION_2012census_uscb_202109"

# The census population, and the universe both questions share.
NATIONAL = 109_188

CATEGORIES = [
    ("RLG_ANG", "Anglican"),
    ("RLG_EVC", "Evangelical Christian"),
    ("RLG_MTH", "Methodist"),
    ("RLG_PTC", "Pentecostal"),
    ("RLG_PBT", "Presbyterian"),
    ("RLG_RMC", "Roman Catholic"),
    ("RLG_SALV", "Salvation Army"),
    ("RLG_SDA", "Seventh Day Adventist"),
    ("RLG_JVW", "Jehovah's Witness"),
    ("RLG_BPT", "Baptist"),
    ("RLG_HIN", "Hindu"),
    ("RLG_MOR", "Mormon"),
    ("RLG_MSL", "Muslim"),
    ("RLG_RASTA", "Rastafarian"),
    ("RLG_TRAD", "Traditional"),
    ("RLG_WORL", "Without religion"),
    ("RLG_OTHR", "Other religion"),
    ("RLG_NSTA", "Not stated"),
]
# NOT a religion total — this file has none. It is the ethnicity universe, and the fact
# that the religion cells sum to it is the whole reconciliation. See the docstring.
UNIVERSE_COL = "ETH_TPOP"

LEVELS = {0: "country", 1: "division", 2: "ed"}
ROWS_EXPECTED = 235           # 1 country + 13 census divisions + 221 enumeration districts
EDS = 221
EDS_POPULATED = 219


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
        # §5a: a 200 is not a download; §11d: check the magic bytes, not the extension.
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
        raise SystemExit(f"missing {XLSX} -- run: python sources/vc.py --fetch")
    # Row 0 of the body is a description row ("Anglican", "Pentecostal", …), not data.
    x = pd.read_excel(XLSX, sheet_name=SHEET, header=0, skiprows=[1])
    if len(x) != ROWS_EXPECTED:
        raise SystemExit(f"expected {ROWS_EXPECTED} rows in '{SHEET}', got {len(x)}")

    cols = [k for k, _ in CATEGORIES] + [UNIVERSE_COL]
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
        for key in ("NSO_CODE", "PARISH", "ADM1_NAME"):
            v = d.get(key)
            if pd.notna(v) and str(v).strip():
                note.append(f"{key.lower()}={str(v).strip()}")
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
    negatives, nulls = defects
    ok = True
    print("Saint Vincent and the Grenadines — 2012 census religion, USCB tabulation\n")

    good = negatives == 0 and nulls == 0
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {negatives} negative and {nulls} null cells "
          f"(expected 0 and 0 — no -999 sentinel and no real nulls in this file)")

    nat = x.loc[x["level"] == 0]
    if len(nat) != 1:
        raise SystemExit("no country row")
    nat = nat.iloc[0]

    good = int(nat[UNIVERSE_COL]) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the census population is {int(nat[UNIVERSE_COL]):,} "
          f"(expected {NATIONAL:,})")

    # THE CROSS-TABLE CHECK, and it is the reason this source is trustworthy. There is no
    # religion total in the file; the 18 religion cells are asserted against the ETHNICITY
    # universe, on every row. Two separately tabulated questions agreeing to the person.
    d = (x["SUM"] - x[UNIVERSE_COL]).abs()
    bad = int((d > 0).sum())
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 18 religion cells sum to {UNIVERSE_COL} on "
          f"all {len(x)} rows ({bad} failures)\n"
          f"        — a CROSS-TABLE identity: this file has no religion total, so the "
          f"religion\n          question is checked against the separately tabulated "
          f"ethnicity universe")

    # both levels partition the country, category by category
    for lv in (1, 2):
        sub = x[x["level"] == lv]
        bad = []
        for key, label in CATEGORIES:
            s, n = int(sub[key].sum()), int(nat[key])
            if s != n:
                bad.append((label, s, n))
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} {LEVELS[lv]}: {len(sub):,} units sum to the "
              f"national figure on all {len(CATEGORIES)} categories ({len(bad)} failures)")
        for label, s, n in bad[:5]:
            print(f"        {label}: {s:,} vs {n:,}")

    ed = x[x["level"] == 2]
    good = ed["GEO_MATCH"].nunique() == len(ed) == EDS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(ed)} enumeration districts, "
          f"{ed['GEO_MATCH'].nunique()} distinct GEO_MATCH (expected {EDS} of each)")

    # Two EDs are genuinely empty. Asserted rather than tolerated, so a vintage that
    # populates them — or that empties more — stops the build.
    pop = int((ed[UNIVERSE_COL] > 0).sum())
    good = pop == EDS_POPULATED
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {pop} of {EDS} enumeration districts have people "
          f"(expected {EDS_POPULATED}; the other two are empty in every column)")

    try:
        import pandas as pd
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

    nat = x[x["level"] == 0].iloc[0]
    ed = x[x["level"] == 2]
    print(f"\n  {NATIONAL:,} people on {len(ed)} enumeration districts, "
          f"median {ed[UNIVERSE_COL].median():.0f} people per unit")
    for key, label in sorted(CATEGORIES, key=lambda kv: -int(nat[kv[0]]))[:6]:
        print(f"    {label:<26} {int(nat[key]):>7,}  {int(nat[key]) / NATIONAL:>6.2%}")
    small = [(label, int(nat[key])) for key, label in CATEGORIES if int(nat[key]) < 400]
    print(f"    …and {len(small)} categories under 400 people: "
          + ", ".join(f"{lbl} {n}" for lbl, n in small))


if __name__ == "__main__":
    main()
