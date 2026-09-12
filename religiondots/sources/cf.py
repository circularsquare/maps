"""Central African Republic — RGPH03 (2003 census), religion by commune.

Reads (or fetches) data/raw/cf/ and writes data/normalized/cf.csv.

**The last African country in the USCB series (§11h), and the seam's sixth country.** It was
listed in §11h's table on 2026-09-05 — 177 communes, 3.8M people, 5 categories — and left
unbuilt while Ethiopia, Pakistan, Bangladesh, Jamaica and Saint Vincent were taken. The 2026-
09-07 Africa sweep re-ran the CKAN query and the HDX search for religion across the whole
continent; the seam is still 34 datasets, and **CAR is the only undrawn African country in it
with a religion layer.** DR Congo has one and is disqualified for the reason §11h gives: its
cells are 31,755 sampled heads of household, not people.

THE VINTAGE IS SINGLE, WHICH ETHIOPIA'S WAS NOT. `CF_RELIGION_GEOG1_2003census` sits against
`CF_GEOG1_ADM3_2003` — counts and boundaries are both the 2003 vintage, so there is no
re-cutting of one onto the other and none of §9u's 418 "Formed from part of…" notes. The file
also ships a GEOG2/2021 boundary set with 181 communes, used by its `Population` and
`Displacement` tables; it is NOT the one the religion layer keys to and is not read here.
Only four communes carry a `USCBCMNT` at all, and they are carried into `note` regardless.

THE JOIN IS AN IDENTITY, MEASURED. 177 polygons, 177 ADM3 count rows, zero keys on either
side alone, zero duplicate `GEO_MATCH`. That is §11h's promise checked on a sixth country
rather than assumed; sources/cf_geo.py asserts it.

**THE UNIVERSE IS 98.50% OF THE CENSUS AND THE SHORTFALL IS NOT A PUBLISHED CATEGORY.** The
data dictionary calls `RLG_BTOTL` *"Total population reporting a religion or belief system"*,
which is §9y's shape exactly: a universe smaller than the count, stated only in the metadata.
The anchor is the `Ethnicity` sheet — the same census, the same 2003 geography, in the same
workbook — whose national total is **3,895,139, the published RGPH03 population**. Religion
covers 3,836,736 of it, so **58,403 people (1.50%) were counted and not asked, or asked and
not tabulated.** Per commune the coverage runs 92.6% to 99.9% with a median of 98.8% and only
two communes below 95%, which is what evenly-spread non-response looks like rather than a
structural hole. It is reported, not filled (spec §3.5).

**DO NOT use the `Age-Sex` sheet as the denominator.** Its national total is 5,052,901,
because it is a **2016 estimate** and not the 2003 census — the HDX notes say so and the sheet
does not. It is 31% above the census and would silently turn a 98.5%-covered country into a
76%-covered one. The Ethnicity sheet is the only cross-table anchor of the right vintage.

THE PARTITION IS EXACT TO ROUNDING, TWO-SIDED. Categories minus total runs −2..+2 across all
267 rows, 90 of the 177 communes are exact, and the national row is −1 on 3.8 million. The
ADM3 totals sum to 3,836,741 against the national row's 3,836,736 — five people over. Nothing
is one-sided, which is what independently rounded published figures look like (§9at) rather
than a dropped category. The whole spread prints on every run instead of vanishing into a
tolerance.

**THERE IS NO TRADITIONAL RELIGION BOX, AND THE RESIDUAL HAS ITS GEOGRAPHY.** The five cells
are `Catholique`, `Protestante`, `Musulmane`, `Autre réligion`, `Sans réligion` — the form
offers no animist or traditional category at all, unlike Ghana, Kenya, Ethiopia, Malawi and
Benin, every one of which does. §11b's continental rule is that an exclusive traditional box
undercounts; CAR is the sharper case where there is no box to undercount. What the table does
show is that `Autre réligion` peaks at 23.7% in Topia and 21.0% in Moboma and Baleloko, and
`Sans réligion` peaks in the same communes — Lobaye and Mambéré-Kadéï, the southwestern
forest, which is the Aka homeland and the part of the country where traditional practice is
strongest. **Two residuals with one geography, and it is the geography of the missing
category.** taxonomy/cf2003.py argues the mapping call that follows from it.

Usage:
    python sources/cf.py --fetch    two GETs, ~3.4 MB
    python sources/cf.py            normalise from data/raw/cf/
"""

import csv
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cf")
OUT = os.path.join(ROOT, "data", "normalized", "cf.csv")

SOURCE_ID = "cf_rgph03_2003_uscb"
YEAR = 2003
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

HDX = ("https://data.humdata.org/dataset/4f41a2c7-167f-4e60-8e90-dd282383dd90/"
       "resource")
URL_GDB = (f"{HDX}/0e759f7b-2716-4d1c-9b44-91b937002d77/download/"
           "central_african_republic.gdb.zip")
URL_XLSX = (f"{HDX}/1670d3a6-a89f-40cd-80c9-bf580b27e08e/download/"
            "central_african_republic_uscb_202303.xlsx")

ZIP = os.path.join(RAW, "central_african_republic.gdb.zip")
XLSX = os.path.join(RAW, "central_african_republic_uscb_202303.xlsx")
GDB = os.path.join(RAW, "Central_African_Republic.gdb")

LAYER_RELIGION = "CF_RELIGION_GEOG1_2003census_uscb_202303"

# The published RGPH03 population. It is NOT hard-coded from outside the file: the Ethnicity
# sheet of this same workbook carries it, which is what makes it a usable cross-table anchor
# rather than a remembered number. Asserted in check().
CENSUS_POPULATION = 3_895_139

# The USCB column, and the census's own French label. The French is what goes into
# `source_category`, per spec §2.4: the mapping is made against the publisher's words.
CATEGORIES = [
    ("RLG_CAT", "Catholique"),
    ("RLG_PRO", "Protestante"),
    ("RLG_MUS", "Musulmane"),
    ("RLG_OTHR", "Autre réligion"),
    ("RLG_NR", "Sans réligion"),
]

LEVELS = {0: "country", 1: "prefecture", 2: "sous_prefecture", 3: "commune"}

ROWS_TOTAL = 267        # 1 country + 17 prefectures + 72 sous-préfectures + 177 communes
COMMUNES = 177

# Two-sided rounding, measured rather than tolerated. Anything outside this is a category
# that went missing, not a rounded figure.
RESIDUAL_BOUND = 2


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0 Safari/537.36"}
    for url, dest, least in ((URL_GDB, ZIP, 2_000_000),
                             (URL_XLSX, XLSX, 300_000)):
        if os.path.exists(dest) and os.path.getsize(dest) > least:
            print("already have", dest)
            continue
        r = requests.get(url, timeout=300, headers=ua)
        r.raise_for_status()
        # §5a: a 200 is not a download. Both of these are zip containers, so the magic is
        # the same for the .gdb.zip and the .xlsx; a Cloudflare page starts `<htm` and a
        # SpreadsheetML file starts `<?xm`, and neither opens.
        if r.content[:4] != b"PK\x03\x04":
            raise SystemExit(f"{dest}: starts {r.content[:16]!r}, expected a zip")
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
    """The xlsx is the read; the gdb is the cross-check. Returns (rows, frame)."""
    import pandas as pd

    x = pd.read_excel(XLSX, sheet_name="Religion", header=0, skiprows=[1])
    if len(x) != ROWS_TOTAL:
        raise SystemExit(f"expected {ROWS_TOTAL} rows in the Religion sheet, "
                         f"got {len(x)}")

    cols = [k for k, _ in CATEGORIES]
    for c in cols + ["RLG_BTOTL"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # Ethiopia's `-999` sentinel (§9u) is not in this file — asserted in check(), not
    # assumed, because it is the sort of thing that is present in one release and absent
    # in the next.
    x["level"] = x["ADM_LEVEL"].astype(int)
    x["SUM"] = x[cols].sum(axis=1, min_count=1)

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
            note.append(f"uscb={' '.join(cmnt.split())}")
        for key, label in CATEGORIES:
            v = d[key]
            if pd.isna(v):
                continue
            rows.append({
                "geo_id": d["GEO_MATCH"], "geo_level": LEVELS[lv],
                "geo_name": name, "source_category": label, "count": int(v),
                "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                "note": "; ".join(note),
            })
    return rows, x


def check(x):
    import pandas as pd

    ok = True
    cols = [k for k, _ in CATEGORIES]
    print("Central African Republic — RGPH03 2003 religion, USCB tabulation\n")

    # 1. no sentinel. Ethiopia's -999 is in-band and arithmetic-safe (§9u), so its ABSENCE
    #    has to be asserted rather than noticed.
    neg = int(sum((x[c] < 0).sum() for c in cols + ["RLG_BTOTL"]))
    ok &= neg == 0
    print(f"  {'OK ' if not neg else 'BAD'} no negative cells — Ethiopia's `-999` "
          f"sentinel is absent from this release ({neg} found)")

    # 2. row counts per level
    counts = x["level"].value_counts().to_dict()
    good = (counts.get(0), counts.get(1), counts.get(2), counts.get(3)) == \
        (1, 17, 72, COMMUNES)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} 1 country, {counts.get(1)} prefectures, "
          f"{counts.get(2)} sous-préfectures, {counts.get(3)} communes")

    nat = x.loc[x["level"] == 0]
    if len(nat) != 1:
        raise SystemExit("no national row")
    nat = nat.iloc[0]

    # 3. the residual is two-sided and bounded — rounding, not a lost category
    resid = (x["SUM"] - x["RLG_BTOTL"]).astype(int)
    worst = int(resid.abs().max())
    good = worst <= RESIDUAL_BOUND and resid.min() < 0 < resid.max()
    ok &= good
    dist = dict(sorted(resid.value_counts().items()))
    print(f"  {'OK ' if good else 'BAD'} categories vs total: residual in "
          f"[{resid.min()}, {resid.max()}], |max| {worst} "
          f"<= {RESIDUAL_BOUND}, two-sided")
    print(f"        distribution over all {len(x)} rows: {dist}")

    # 4. every level partitions the country, category by category, to rounding
    for lv in (1, 2, 3):
        sub = x[x["level"] == lv]
        bad = []
        for key, label in CATEGORIES:
            s, n = int(sub[key].sum()), int(nat[key])
            if abs(s - n) > COMMUNES * RESIDUAL_BOUND:
                bad.append((label, s, n))
        ok &= not bad
        spread = {label: int(sub[key].sum()) - int(nat[key])
                  for key, label in CATEGORIES}
        print(f"  {'OK ' if not bad else 'BAD'} {LEVELS[lv]}: {len(sub)} units, "
              f"category sums minus national = {spread}")
        for label, s, n in bad:
            print(f"        {label}: {s:,} vs {n:,}")

    # 5. THE CROSS-TABLE ANCHOR. The Ethnicity sheet is the same census, the same 2003
    #    geography, in the same workbook, and its total is the published RGPH03 count.
    eth = pd.read_excel(XLSX, sheet_name="Ethnicity", header=0, skiprows=[1])
    eth["level"] = eth["ADM_LEVEL"].astype(int)
    eth_nat = int(eth.loc[eth["level"] == 0, "ETH_BTOTL"].iloc[0])
    good = eth_nat == CENSUS_POPULATION
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the Ethnicity sheet's national total is "
          f"{eth_nat:,}, the published RGPH03 population ({CENSUS_POPULATION:,})")

    universe = int(nat["RLG_BTOTL"])
    cover = 100.0 * universe / eth_nat
    print(f"  -- the religion universe is {universe:,} = {cover:.2f}% of it; "
          f"{eth_nat - universe:,} people ({100 - cover:.2f}%) were counted and are "
          f"not\n     in this table. Not a published category; reported, not filled "
          f"(§3.5).")

    # 6. per-commune coverage, so the shortfall is shown to be spread rather than local
    e3 = eth[eth["level"] == 3].set_index("GEO_MATCH")["ETH_BTOTL"]
    r3 = x[x["level"] == 3].set_index("GEO_MATCH")["RLG_BTOTL"]
    shared = r3.index.intersection(e3.index)
    good = len(shared) == COMMUNES
    ok &= good
    pct = (100.0 * r3[shared] / e3[shared]).sort_values()
    print(f"  {'OK ' if good else 'BAD'} all {len(shared)} communes appear in both "
          f"sheets; coverage min {pct.iloc[0]:.1f}%, median {pct.median():.1f}%, "
          f"max {pct.iloc[-1]:.1f}%")
    low = pct[pct < 95]
    names = x.set_index("GEO_MATCH")["AREA_NAME"]
    named = ", ".join("{} ({:.1f}%)".format(names[k], v) for k, v in low.items())
    print(f"        {len(low)} communes below 95%: {named}")

    # 7. the workbook and the geodatabase agree. Same publisher and same release, so this
    #    catches a misread rather than being independent (§9u).
    try:
        import pyogrio
        g = pyogrio.read_dataframe(GDB, layer=LAYER_RELIGION, read_geometry=False)
        m = x[["GEO_MATCH"] + cols].merge(
            g[["GEO_MATCH"] + cols], on="GEO_MATCH", how="left",
            suffixes=("_x", "_g"))
        diff = int(sum((m[f"{c}_x"].fillna(-1) != m[f"{c}_g"].fillna(-1)).sum()
                       for c in cols))
        ok &= not diff
        print(f"  {'OK ' if not diff else 'BAD'} the .gdb religion layer agrees with "
              f"the .xlsx on all {len(x) * len(cols):,} cells ({diff} differ) — a read "
              f"check, not an independent one")
    except ImportError:
        print("  --  pyogrio not installed, skipping the gdb/xlsx cross-check")

    # ---- what is being drawn
    w = x[x["level"] == 3]
    print(f"\n  {len(w)} communes drawn, {int(w['RLG_BTOTL'].sum()):,} people, "
          f"{int(w['RLG_BTOTL'].sum()) / len(w):,.0f} each. Categories, national:")
    for key, label in CATEGORIES:
        n = int(nat[key])
        print(f"    {n:>11,}  {100.0 * n / universe:6.2f}%  {label}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (XLSX, GDB):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run: python sources/cf.py --fetch")
    rows, x = read()
    check(x)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
