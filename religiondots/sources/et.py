"""Ethiopia — 2007 Population and Housing Census, religion by woreda.

Reads (or fetches) data/raw/et/ and writes data/normalized/et.csv.

**The largest untouched African source in this file, and it did not come from the Ethiopian
statistical agency at all.** sources.md §11b ranked Ethiopia 2007 the top untried source on
the continent — woreda level, ~670 units, 74M people, six categories — and priced it as
"eleven regional PDF volumes on a website that has since moved". None of that is needed. The
**U.S. Census Bureau** publishes Ethiopia's census tabulations on HDX as a geodatabase with
the boundaries in the same file, and the religion table is one attribute layer of it.

THE ROUTE, AND IT GENERALISES (see §11h). `data.humdata.org` is a CKAN. The organization
`us-census-bureau` publishes 34 datasets titled "<Country> Subnational Population and Housing
Data Tables with Administrative Boundaries", of which **eight carry a religion layer**:
Ethiopia, Bangladesh, Pakistan, Jamaica, Saint Vincent, Central African Republic, DR Congo
and the Philippines — the last of which this project already uses for its geography
(sources/ph_geo.md) without ever having looked at what else was in the file.

WHAT IT IS WORTH. 73,750,932 people on **738 woredas**, about 99,900 each — finer than
Kenya's 1.01 million (§9o) and close to Guyana's 74,700 (§9r), on the second most populous
country in Africa. Six categories: Orthodox, Protestant, Catholic, Islam, Traditional, Other.
The Orthodox cell is the Ethiopian Orthodox Tewahedo Church, which is **Oriental Orthodox and
not Eastern Orthodox** — 32.1 million people, and it lands on `christianity.oriental.ethiopian`,
a node the US Religion Census created for a diaspora of 66,000. See taxonomy/et2007.py.

THE SENTINEL IS -999 AND IT PARSES AS A NUMBER. Four woredas carry `-999` in every category
instead of a value. Summed naively they take 23,976 people off the national total — 0.0325%,
small enough to look like rounding and to be explained away. §5a's rule in a new disguise:
the defect is in-band and arithmetic-safe, so nothing raises. `mask(< 0)` before summing, and
the 738 remaining woredas then reconcile to the national figure EXACTLY, category by
category, which is how you know the sentinel was the whole of the discrepancy.

THE COUNTS ARE 2007 AND THE BOUNDARIES ARE 2021, AND USCB DID THE MATCHING. The layer names
say so outright — `ET_RELIGION_2007census` against `ET_GEOG_ADM3_2021`. Sidama is a separate
region here and was part of SNNPR in 2007, so the re-cutting is real and reaches ADM1. At
woreda level 418 of 743 units carry a `USCBCMNT` reading "Formed from part of <census-era
unit>", and **70 census-era woredas are split across two to four modern ones**. Every count
is an integer, so nothing was apportioned by area into fractions, and the exact partition
proves nothing is duplicated or dropped — but the per-unit split is USCB's work and is NOT
independently checked here. The comment is carried into the `note` column of every row so it
cannot be lost, which is what ph_geo.py learned to do with `USCBCMNT` for the BARMM.

FIVE WOREDAS HAVE NO DATA AND THREE OF THEM SAY WHY. `ADEAR`, `BEDU` and one unnamed unit in
Āfar are marked "Population data not available" by USCB; `BELTU` in Oromīya is blank; and
`FINFINNE ZURIA SPECIAL ZONE` has no polygon either ("No polygon included in FGDB"). They are
dropped, and dropping them costs nothing measurable because the national total does not
include them: the 738 woredas with data already sum to it exactly. That is the tell that the
2007 census itself never counted these areas — parts of Āfar and Somali were not fully
enumerated — rather than that USCB lost them.

Usage:
    python sources/et.py --fetch    two GETs, ~5.4 MB
    python sources/et.py            normalise from data/raw/et/
"""

import csv
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "et")
OUT = os.path.join(ROOT, "data", "normalized", "et.csv")

SOURCE_ID = "et_phc_2007_uscb"
YEAR = 2007
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

HDX = "https://data.humdata.org/dataset/5438946a-51b7-44d6-9b76-aeafdb4dc4d3/resource"
URL_GDB = f"{HDX}/71e2b4e4-6e3d-4cc9-90c3-51551dbc7043/download/ethiopia.gdb.zip"
URL_XLSX = f"{HDX}/f112d5fa-d90e-4352-b0c1-54cffbbad62d/download/ethiopia_uscb_202308.xlsx"

ZIP = os.path.join(RAW, "ethiopia.gdb.zip")
XLSX = os.path.join(RAW, "ethiopia_uscb_202308.xlsx")
GDB = os.path.join(RAW, "Ethiopia.gdb")

LAYER_RELIGION = "ET_RELIGION_2007census_uscb_202308"

# The census population of Ethiopia, 2007. The one figure here that comes from outside the
# USCB file, and so the only genuinely independent anchor the reconciliation has.
NATIONAL = 73_750_932

# USCB's suffixes are _B both sexes, _F female, _M male. Only _B is drawn; the other two are
# read solely so that M + F == B can be asserted on every row (851 x 6 equalities).
CATEGORIES = [
    ("RLG_ORDX", "Orthodox"),
    ("RLG_PROT", "Protestant"),
    ("RLG_CATH", "Catholic"),
    ("RLG_ISL", "Islam"),
    ("RLG_TRAD", "Traditional"),
    ("RLG_OTHR", "Other"),
]

LEVELS = {0: "country", 1: "region", 2: "zone", 3: "woreda"}

SENTINEL_CELLS = 24     # 4 woredas x 6 categories, all -999. Asserted, not assumed.


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
    for url, dest, magic, least in ((URL_GDB, ZIP, b"PK\x03\x04", 3_000_000),
                                    (URL_XLSX, XLSX, b"PK\x03\x04", 1_000_000)):
        if os.path.exists(dest) and os.path.getsize(dest) > least:
            print("already have", dest)
            continue
        r = requests.get(url, timeout=300, headers=ua)
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
    """The xlsx is the read; the gdb is the cross-check. Returns (rows, stats)."""
    import pandas as pd

    x = pd.read_excel(XLSX, sheet_name="Religion", header=0, skiprows=[1])
    if len(x) != 851:
        raise SystemExit(f"expected 851 rows in the Religion sheet, got {len(x)}")

    cols = [f"{k}_{s}" for k, _ in CATEGORIES for s in "BFM"]
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # ---- the -999 sentinel, counted before it is masked
    sentinels = int(sum((x[c] < 0).sum() for c in cols if c.endswith("_B")))
    for c in cols:
        x[c] = x[c].mask(x[c] < 0)

    x["level"] = x["ADM_LEVEL"].astype(int)
    x["TOT"] = x[[f"{k}_B" for k, _ in CATEGORIES]].sum(axis=1, min_count=1)

    rows = []
    for r in x.itertuples(index=False):
        d = r._asdict()
        lv = int(d["ADM_LEVEL"])
        name = str(d["AREA_NAME"]).strip()
        note = [f"level={LEVELS[lv]}"]
        nso = d.get("NSO_CODE")
        if pd.notna(nso):
            note.append(f"nso={int(nso)}")
        cmnt = d.get("USCBCMNT")
        if isinstance(cmnt, str) and cmnt.strip():
            # Carried verbatim: this is the only record of how a 2021 woreda relates to the
            # census-era unit its figures came from, and it exists per unit and nowhere else.
            note.append(f"uscb={' '.join(cmnt.split())}")
        for key, label in CATEGORIES:
            v = d[f"{key}_B"]
            if pd.isna(v):
                continue
            rows.append({
                "geo_id": d["GEO_MATCH"], "geo_level": LEVELS[lv], "geo_name": name,
                "source_category": label, "count": int(v), "basis": BASIS,
                "year": YEAR, "source_id": SOURCE_ID, "note": "; ".join(note),
            })
    return rows, x, sentinels


def check(x, sentinels):
    import numpy as np
    import pandas as pd

    ok = True
    print("Ethiopia — 2007 census religion, USCB tabulation on 2021 boundaries\n")

    # 1. the sentinel is exactly where it is expected to be
    good = sentinels == SENTINEL_CELLS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {sentinels} `-999` cells in the both-sexes columns "
          f"(expected {SENTINEL_CELLS} = 4 woredas x 6 categories)")

    nat = x.loc[x["GEO_MATCH"] == "ETH_00"]
    if len(nat) != 1:
        raise SystemExit("no ETH_00 row")
    nat = nat.iloc[0]

    # 2. the one external anchor
    good = int(nat["TOT"]) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the six categories sum to {int(nat['TOT']):,}, "
          f"the published 2007 census population ({NATIONAL:,})")

    # 3. every level partitions the country exactly, category by category
    for lv in (1, 2, 3):
        sub = x[x["level"] == lv]
        drawn = sub[sub["TOT"].notna()]
        bad = []
        for key, label in CATEGORIES:
            s, n = drawn[f"{key}_B"].sum(), nat[f"{key}_B"]
            if int(s) != int(n):
                bad.append((label, int(s), int(n)))
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} {LEVELS[lv]}: {len(drawn):,} units with "
              f"data sum to the national figure on all 6 categories "
              f"({len(bad)} failures, {len(sub) - len(drawn)} units dropped as all-null)")
        for label, s, n in bad:
            print(f"        {label}: {s:,} vs {n:,}")

    # 4. male + female == both sexes, every row, every category
    bad = 0
    for key, _ in CATEGORIES:
        d = (x[f"{key}_M"] + x[f"{key}_F"]) - x[f"{key}_B"]
        bad += int((d.abs() > 0).sum())
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} male + female == both sexes on all "
          f"{len(x) * len(CATEGORIES):,} cells ({bad} failures)")

    # 5. the workbook and the geodatabase are the same numbers. NOT an independent check —
    #    same publisher, same release — but it catches a misread of either file.
    try:
        import pyogrio
        g = pyogrio.read_dataframe(GDB, layer=LAYER_RELIGION, read_geometry=False)
        for key, _ in CATEGORIES:
            g[f"{key}_B"] = g[f"{key}_B"].mask(g[f"{key}_B"] < 0)
        m = x[["GEO_MATCH"]].merge(g, on="GEO_MATCH", how="left")
        diff = 0
        for key, _ in CATEGORIES:
            a = m[f"{key}_B"].fillna(-1).to_numpy()
            b = x[f"{key}_B"].fillna(-1).to_numpy()
            diff += int((np.abs(a - b) > 0).sum())
        ok &= not diff
        print(f"  {'OK ' if not diff else 'BAD'} the .gdb religion layer agrees with the "
              f".xlsx on all {len(x) * len(CATEGORIES):,} cells ({diff} differ) — a read "
              f"check, not an independent one")
    except ImportError:
        print("  --  pyogrio not installed, skipping the gdb/xlsx cross-check")

    # ---- what is being drawn
    w = x[(x["level"] == 3) & x["TOT"].notna()]
    print(f"\n  {len(w)} woredas drawn, {int(w['TOT'].sum()):,} people, "
          f"{int(w['TOT'].sum()) / len(w):,.0f} each. Categories, national:")
    for key, label in CATEGORIES:
        n = int(nat[f"{key}_B"])
        print(f"    {n:>11,}  {100.0 * n / NATIONAL:6.2f}%  {label}")

    dropped = x[(x["level"] == 3) & x["TOT"].isna()]
    print(f"\n  {len(dropped)} woredas dropped for having no data at all:")
    for r in dropped.itertuples(index=False):
        d = r._asdict()
        why = " ".join(str(d["USCBCMNT"]).split()) if pd.notna(d["USCBCMNT"]) else "(blank)"
        print(f"    {d['GEO_MATCH']:14s} {str(d['AREA_NAME'])[:34]:34s} {why[:44]}")
    print("  They cost nothing: the national total does not include them either, which is "
          "\n  why the 738 with data reconcile to it exactly.")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (XLSX, GDB):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run: python sources/et.py --fetch")
    rows, x, sentinels = read()
    check(x, sentinels)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
