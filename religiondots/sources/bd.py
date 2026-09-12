"""Bangladesh — 2011 Population and Housing Census, religion by upazila.

Reads (or fetches) data/raw/bd/ and writes data/normalized/bd.csv.

**The largest source left in the USCB series, and the fourth-largest country on this map.**
144.0 million people on 544 upazilas and thanas, from the same publisher and in the same
shape as Ethiopia (§9u) and Pakistan (§9t): the U.S. Census Bureau transcribes the national
office's own tabulation onto HDX as a geodatabase with the boundaries in the same file.
sources.md §11h is the general finding and §11j is the audit that picked this country.

WHAT IT IS WORTH. Bangladesh is the last nine-figure source anywhere in `sources.md`, and it
is the largest Hindu population outside India — **12,299,981 people**, more than any country
on this map except India itself. It also carries a Theravada Buddhist and tribal-Christian
geography in the Chittagong Hill Tracts that exists nowhere else here.

THE QUESTION IS FIVE CELLS AND THAT IS THE WHOLE OF IT. Muslim, Hindu, Christian, Buddhist,
Other. Shallower than Kenya's thirteen and about level with Indonesia's six; §3.9's trade
taken to the geography end, like Sri Lanka. What the shallow list buys is 264,786 people per
unit over a country of 144 million.

**THERE IS NO NON-RESPONSE CELL, AND THE FIVE CATEGORIES ARE A PERFECT PARTITION.** Every
row's five cells sum to its own `RLG_TPOP` exactly — all 617 of them, at every level — and
the 544 upazilas sum to 144,043,696, which is the national row. So 100% of the tabulation is
drawn. As with Ethiopia that does NOT mean nobody refused: it means BBS distributed or never
published a refusal cell, and nothing here can undo that. `note_public` says so.

NO -999, AND THE CONVENTION IS PER COUNTRY. Ethiopia's file uses `-999` as its null sentinel
and it parses as a number (§9u); Pakistan's uses real nulls; **this one has neither** — not a
single negative and not a single null in any religion column. So the sentinel is a per-file
convention rather than a series one, and the right move is to assert the count you expect
rather than to mask defensively and hope. This module asserts ZERO negatives and ZERO nulls,
which is a stronger check than Ethiopia's precisely because it can be.

THE BOUNDARIES ARE THE CENSUS VINTAGE, WHICH IS THE THING ETHIOPIA DID NOT HAVE. The layers
are `BD_GEOG_ADM3_2011` against `BD_RELIGION_AND_ETHNICITY_2011census` — both 2011, so there
is no re-cutting anywhere and spec §8.1 is satisfied with nothing to reconcile. The tell is
that **`USCBCMNT` is empty on every one of the 617 rows**, where Ethiopia carried 418
"Formed from part of…" notes. An empty lineage column is evidence of a matched vintage, and
it is worth checking for that reason rather than skipping because it is blank.

THE SHEET ALSO CARRIES 27 NAMED ETHNIC GROUPS, keyed identically and not read here. Chakma,
Marma, Tripura, Garo, Santal and 22 more, 1,586,183 people. Not needed to draw religion, and
worth knowing it is in the file if the Hill Tracts ever want a second look.

Usage:
    python sources/bd.py --fetch    two GETs, ~74 MB (the gdb is large; the xlsx is 4.8 MB)
    python sources/bd.py            normalise from data/raw/bd/
"""

import csv
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bd")
OUT = os.path.join(ROOT, "data", "normalized", "bd.csv")

SOURCE_ID = "bd_phc_2011_uscb"
YEAR = 2011
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

HDX = "https://data.humdata.org/dataset/862d607b-a195-4f69-b0cb-bd2fe6ac858f/resource"
URL_GDB = f"{HDX}/b01a071f-f35a-4a6a-adc4-b01b25ef8902/download/bangladesh.gdb.zip"
URL_XLSX = f"{HDX}/1fdc9232-9fb4-49d2-b66a-b7a375472f72/download/bangladesh_uscb_202107.xlsx"

ZIP = os.path.join(RAW, "bangladesh.gdb.zip")
XLSX = os.path.join(RAW, "bangladesh_uscb_202107.xlsx")
GDB = os.path.join(RAW, "Bangladesh.gdb")

SHEET = "Religion and Ethnicity"
LAYER_RELIGION = "BD_RELIGION_AND_ETHNICITY_2011census_uscb_202107"

# The census population of Bangladesh, 2011, as published by BBS. The one figure here that
# comes from outside the USCB file, and so the only genuinely independent anchor.
NATIONAL = 144_043_696

# No sex suffixes in this file — Ethiopia's are _B/_F/_M and Bangladesh publishes both sexes
# only, so there is no M + F == B identity to assert here. `RLG_TPOP` replaces it as the
# per-row internal check, and it is a better one: it is the source's own total, not a sum.
CATEGORIES = [
    ("RLG_MSL", "Muslim"),
    ("RLG_HIN", "Hindu"),
    ("RLG_CHR", "Christian"),
    ("RLG_BUD", "Buddhist"),
    ("RLG_OTH", "Other religion"),
]
TOTAL_COL = "RLG_TPOP"

LEVELS = {0: "country", 1: "division", 2: "zila", 3: "upazila"}

ROWS_EXPECTED = 617           # 1 country + 8 divisions + 64 zilas + 544 upazilas/thanas
UPAZILAS = 544                # 483 upazilas + 61 thanas, per the Metadata sheet


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
    for url, dest, magic, least in ((URL_GDB, ZIP, b"PK\x03\x04", 40_000_000),
                                    (URL_XLSX, XLSX, b"PK\x03\x04", 2_000_000)):
        if os.path.exists(dest) and os.path.getsize(dest) > least:
            print("already have", dest)
            continue
        r = requests.get(url, timeout=900, headers=ua)
        r.raise_for_status()
        # §5a: HTTP 200 is not a download, and §11d: check the magic bytes rather than the
        # extension. A `.xlsx` that is really SpreadsheetML starts `<?xm` and no engine
        # opens it; a Cloudflare page starts `<htm` and parses as neither.
        if r.content[:4] != magic:
            raise SystemExit(f"{dest}: starts {r.content[:16]!r}, expected {magic!r}")
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

    x = pd.read_excel(XLSX, sheet_name=SHEET, header=0, skiprows=[1])
    if len(x) != ROWS_EXPECTED:
        raise SystemExit(f"expected {ROWS_EXPECTED} rows in '{SHEET}', got {len(x)}")

    cols = [k for k, _ in CATEGORIES] + [TOTAL_COL]
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # Counted, not masked. Ethiopia masks `-999` because it has some; this file should have
    # none of either, and a defect here means the release changed under us.
    negatives = int(sum((x[c] < 0).sum() for c in cols))
    nulls = int(sum(x[c].isna().sum() for c in cols))

    x["level"] = x["ADM_LEVEL"].astype(int)
    x["TOT"] = x[[k for k, _ in CATEGORIES]].sum(axis=1, min_count=1)

    rows = []
    for r in x.itertuples(index=False):
        d = r._asdict()
        lv = int(d["ADM_LEVEL"])
        name = str(d["AREA_NAME"]).strip()
        note = [f"level={LEVELS[lv]}"]
        nso = d.get("NSO_CODE")
        if pd.notna(nso):
            note.append(f"nso={nso}")
        cmnt = d.get("USCBCMNT")
        if isinstance(cmnt, str) and cmnt.strip():
            # Empty throughout this file — see the module docstring. Carried anyway, so that
            # a future re-release which starts populating it cannot lose the lineage.
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
    import numpy as np
    import pandas as pd

    negatives, nulls = defects
    ok = True
    print("Bangladesh — 2011 census religion, USCB tabulation on 2011 boundaries\n")

    # 1. this file has neither of the two null conventions the series uses elsewhere
    good = negatives == 0 and nulls == 0
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {negatives} negative and {nulls} null cells in the "
          f"religion columns (expected 0 and 0 — no -999 sentinel in this file, unlike "
          f"Ethiopia's, and no real nulls, unlike Pakistan's)")

    nat = x.loc[x["GEO_MATCH"] == "BGD_00"]
    if len(nat) != 1:
        raise SystemExit("no BGD_00 row")
    nat = nat.iloc[0]

    # 2. the one external anchor
    good = int(nat["TOT"]) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the five categories sum to {int(nat['TOT']):,}, "
          f"the published 2011 census population ({NATIONAL:,})")

    # 3. the source's OWN total agrees with the sum of its own parts, every row.
    #    This is what replaces Ethiopia's M + F == B identity, and it is stronger: RLG_TPOP
    #    is published beside the categories rather than derived from them, so a transcription
    #    slip in any one cell breaks it.
    d = (x["TOT"] - x[TOTAL_COL]).abs()
    bad = int((d > 0).sum())
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the five categories sum to the published "
          f"{TOTAL_COL} on all {len(x)} rows ({bad} failures) — no non-response cell exists, "
          f"so the partition is exact by construction and this proves it")

    # 4. every level partitions the country exactly, category by category
    for lv in (1, 2, 3):
        sub = x[x["level"] == lv]
        drawn = sub[sub["TOT"].notna()]
        bad = []
        for key, label in CATEGORIES:
            s, n = drawn[key].sum(), nat[key]
            if int(s) != int(n):
                bad.append((label, int(s), int(n)))
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} {LEVELS[lv]}: {len(drawn):,} units sum to "
              f"the national figure on all 5 categories ({len(bad)} failures)")
        for label, s, n in bad:
            print(f"        {label}: {s:,} vs {n:,}")

    # 5. the drawn tier is a clean key, with nothing empty
    up = x[x["level"] == 3]
    good = up["GEO_MATCH"].nunique() == len(up) == UPAZILAS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(up)} upazila rows, "
          f"{up['GEO_MATCH'].nunique()} distinct GEO_MATCH (expected {UPAZILAS} of each)")
    empty = int((up["TOT"] <= 0).sum())
    ok &= not empty
    print(f"  {'OK ' if not empty else 'BAD'} {empty} upazilas with no people (expected 0 — "
          f"unlike Ethiopia, no unit here is undrawable)")

    # 6. the workbook and the geodatabase are the same numbers. NOT an independent check —
    #    same publisher, same release — but it catches a misread of either file.
    try:
        import pyogrio
        g = pyogrio.read_dataframe(GDB, layer=LAYER_RELIGION, read_geometry=False)
        m = x[["GEO_MATCH"]].merge(g, on="GEO_MATCH", how="left")
        diff = 0
        for key, _ in CATEGORIES:
            a = m[key].fillna(-1).to_numpy(dtype=float)
            b = x[key].fillna(-1).to_numpy(dtype=float)
            diff += int((np.abs(a - b) > 0).sum())
        ok &= not diff
        print(f"  {'OK ' if not diff else 'BAD'} the .gdb religion layer agrees with the "
              f".xlsx on all {len(x) * len(CATEGORIES):,} cells ({diff} differ) — a read "
              f"check, not an independent one")
    except ImportError:
        print("  --  pyogrio not installed, skipping the gdb/xlsx cross-check")

    # 7. the lineage column is empty, which is the evidence for the matched vintage
    cmnt = int(x["USCBCMNT"].notna().sum()) if "USCBCMNT" in x.columns else -1
    good = cmnt == 0
    print(f"  {'OK ' if good else '!! '} USCBCMNT is populated on {cmnt} rows (expected 0). "
          f"Ethiopia's carries 418 'Formed from part of…' notes because its 2007 counts were "
          f"re-cut onto 2021 units; Bangladesh's counts and boundaries are both 2011, so an "
          f"empty column is what a matched vintage looks like")

    # ---- what is being drawn
    print(f"\n  {len(up)} upazilas drawn, {int(up['TOT'].sum()):,} people, "
          f"{int(up['TOT'].sum()) / len(up):,.0f} each. Categories, national:")
    for key, label in CATEGORIES:
        n = int(nat[key])
        print(f"    {n:>12,}  {100.0 * n / NATIONAL:6.2f}%  {label}")

    # ---- the geography that is the reason to draw the country
    up = up.copy()
    for key, label in CATEGORIES:
        up[f"sh_{key}"] = up[key] / up["TOT"]
    print("\n  where the minorities actually are — top 6 upazilas by share:")
    for key, label in (("RLG_HIN", "Hindu"), ("RLG_BUD", "Buddhist"),
                       ("RLG_CHR", "Christian"), ("RLG_OTH", "Other religion")):
        print(f"    {label}:")
        for r in up.nlargest(6, f"sh_{key}").itertuples(index=False):
            d = r._asdict()
            print(f"       {100 * d[f'sh_{key}']:5.1f}%  {str(d['AREA_NAME'])[:26]:26s} "
                  f"{str(d['ADM1_NAME'])[:12]:12s} pop {int(d['TOT']):>9,}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (XLSX, GDB):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run: python sources/bd.py --fetch")
    rows, x, defects = read()
    check(x, defects)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
