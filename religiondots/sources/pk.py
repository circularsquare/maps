"""Pakistan — 2017 Population and Housing Census, religion by district.

Reads (or fetches) data/raw/pk/ and writes data/normalized/pk.csv.

Second country off the USCB seam (sources.md §11h); Ethiopia (§9u) was the first. The
publisher is the **U.S. Census Bureau**, transcribing PBS's own *Table 9, Population by sex,
religion and rural/urban*, with the boundaries in the same geodatabase keyed on `GEO_MATCH`.

**THE DRAWN TIER IS DISTRICT AND THAT IS A §14 DECISION, NOT A DATA ONE.** The file offers
585 fourth-order units (tehsils, talukas, thanas) and this module normalises all of them —
but `countries.py` draws ADM3, 135 districts. PBS publishes religion **by district**; its own
tehsil release carries Table 4 only (area, population, sex ratio, density, urban proportion)
and no religion at all, which was checked by downloading `sindh_tehsil.pdf` and reading it.
spec §14.4 says no resolution finer than the state's own publication, and for this variable
that ceiling is the district. See `sources/pk.md` §3 for what the tehsil tier would have
shown and why it is not drawn.

WHY PAKISTAN IS WORTH DRAWING AT ALL, given that 96.5% of it is one colour. Because the
other 3.5% is not spread thin — it is two of the sharpest minority geographies anywhere on
this map. **Umerkot district is 52.2% Hindu and Tharparkar 43.4%**, against a national 2.1%:
the Thar desert and the irrigated Sindh belt beside it are the largest Hindu population
outside India and Nepal, and they sit in a country that is otherwise 97% Muslim. And the
Christian belt of central Punjab — Lahore 5.1%, Sheikhupura 3.8%, Gujranwala 3.6%,
Faisalabad 3.4% — is a mission-and-labour geography with no equivalent anywhere nearby.

AZAD KASHMIR AND GILGIT-BALTISTAN HAVE NO RELIGION DATA AND USCB SAYS WHY. 20 of the 155
districts are all-null, and the metadata note is explicit: *"Religion data for the autonomous
and disputed regions of Azad Kashmir and Gilgit-Baltistan were not published by the Pakistan
Bureau of Statistics."* They are dropped and draw blank. This costs nothing arithmetically —
the 135 districts with data sum to the national figure exactly — but it is a visible hole in
the north and `note_public` says what it is rather than letting it read as "nobody lives
there". Note that Gilgit-Baltistan is also where Pakistan's Shia population is most
concentrated, so the one place the Sunni/Shia split would be most visible is also the one
place with no data.

NO -999 SENTINEL HERE, WHICH IS WORTH SAYING. Ethiopia's file used `-999` for "no data"
(§9u); Pakistan's uses genuine nulls. **The convention is per-country and not per-publisher**,
so it has to be checked each time rather than assumed from the last file. `check()` asserts
that there are zero negative cells, so a future release that adopts the sentinel fails loudly
instead of silently subtracting.

Usage:
    python sources/pk.py --fetch    two GETs, ~8.7 MB
    python sources/pk.py            normalise from data/raw/pk/
"""

import csv
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pk")
OUT = os.path.join(ROOT, "data", "normalized", "pk.csv")

SOURCE_ID = "pk_phc_2017_uscb"
YEAR = 2017
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

HDX = "https://data.humdata.org/dataset/13ba0782-feca-46bb-81b8-53ee1faeebde/resource"
URL_GDB = f"{HDX}/5d6aa059-2323-450b-81a2-7477a868f289/download/pakistan.gdb.zip"
URL_XLSX = f"{HDX}/ec76611c-2acb-44cd-ba31-51cfba5db94e/download/pakistan_uscb_202401.xlsx"

ZIP = os.path.join(RAW, "pakistan.gdb.zip")
XLSX = os.path.join(RAW, "pakistan_uscb_202401.xlsx")
GDB = os.path.join(RAW, "Pakistan.gdb")

LAYER_RELIGION = "PK_RELIGION_GEOG1_2017census_uscb_202401"

# The census population of Pakistan, 2017 — PBS's own published total, and the only figure
# here that comes from outside the USCB file.
NATIONAL = 207_684_626

CATEGORIES = [
    ("RLG_MUS", "Islam"),
    ("RLG_CHR", "Christianity"),
    ("RLG_HIN", "Hinduism"),
    ("RLG_QAD", "Qadiani/Ahmadi"),
    ("RLG_SCH", "Scheduled Castes"),
    ("RLG_OTH", "Other"),
]

LEVELS = {0: "country", 1: "province", 2: "division", 3: "district", 4: "tehsil"}

# 10 Azad Kashmir + 10 Gilgit-Baltistan districts, and their divisions, provinces and
# tehsils. Asserted rather than assumed: a release that starts publishing them should
# change this number and say so.
NULL_DISTRICTS = 20


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
    for url, dest, least in ((URL_GDB, ZIP, 6_000_000), (URL_XLSX, XLSX, 800_000)):
        if os.path.exists(dest) and os.path.getsize(dest) > least:
            print("already have", dest)
            continue
        r = requests.get(url, timeout=600, headers=ua)
        r.raise_for_status()
        # §5a / §11d: check the magic bytes, not the extension or the status code.
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
    """The xlsx is the read; the gdb religion layer is the cross-check."""
    import pandas as pd

    x = pd.read_excel(XLSX, sheet_name="Religion", header=0, skiprows=[1])
    if len(x) != 785:
        raise SystemExit(f"expected 785 rows in the Religion sheet, got {len(x)}")

    cols = [k for k, _ in CATEGORIES]
    for c in cols + ["RLG_BTOTL"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    negatives = int(sum((x[c] < 0).sum() for c in cols))
    x["level"] = x["ADM_LEVEL"].astype(int)
    x["TOT"] = x[cols].sum(axis=1, min_count=1)

    rows = []
    for r in x.itertuples(index=False):
        d = r._asdict()
        lv = int(d["ADM_LEVEL"])
        note = [f"level={LEVELS[lv]}"]
        cmnt = d.get("USCBCMNT")
        if isinstance(cmnt, str) and cmnt.strip():
            note.append(f"uscb={' '.join(cmnt.split())}")
        for key, label in CATEGORIES:
            v = d[key]
            if pd.isna(v):
                continue
            rows.append({
                "geo_id": d["GEO_MATCH"], "geo_level": LEVELS[lv],
                "geo_name": str(d["AREA_NAME"]).strip(),
                "source_category": label, "count": int(v), "basis": BASIS,
                "year": YEAR, "source_id": SOURCE_ID, "note": "; ".join(note),
            })
    return rows, x, negatives


def check(x, negatives):
    import numpy as np
    import pandas as pd

    ok = True
    cols = [k for k, _ in CATEGORIES]
    print("Pakistan — 2017 census religion, USCB tabulation of PBS Table 9\n")

    # 1. Ethiopia's sentinel is NOT this file's convention, and that is asserted
    ok &= negatives == 0
    print(f"  {'OK ' if not negatives else 'BAD'} zero negative cells — this file marks "
          f"missing with a real null, not Ethiopia's `-999` ({negatives} found)")

    nat = x[x["level"] == 0]
    if len(nat) != 1:
        raise SystemExit("no national row")
    nat = nat.iloc[0]

    good = int(nat["TOT"]) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the six categories sum to {int(nat['TOT']):,}, "
          f"the published 2017 census population ({NATIONAL:,})")

    good = int(nat["RLG_BTOTL"]) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} and they sum to the file's own RLG_BTOTL column")

    # 2. every level partitions the country, category by category
    for lv in (1, 2, 3, 4):
        sub = x[x["level"] == lv]
        drawn = sub[sub["TOT"].notna()]
        bad = []
        for key, label in CATEGORIES:
            s, n = drawn[key].sum(), nat[key]
            if int(s) != int(n):
                bad.append((label, int(s), int(n)))
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} {LEVELS[lv]}: {len(drawn):,} units with "
              f"data sum to the national figure on all 6 categories "
              f"({len(bad)} failures, {len(sub) - len(drawn)} all-null)")
        for label, s, n in bad:
            print(f"        {label}: {s:,} vs {n:,}")

    # 3. the missing districts are exactly the two disputed regions
    d = x[(x["level"] == 3) & x["TOT"].isna()]
    ok &= len(d) == NULL_DISTRICTS
    regions = sorted(set(d["ADM1_NAME"].dropna()))
    print(f"  {'OK ' if len(d) == NULL_DISTRICTS else 'BAD'} {len(d)} districts have no "
          f"religion data (expected {NULL_DISTRICTS}), and they are exactly: "
          f"{', '.join(regions)}")

    # 4. gdb vs xlsx — a read check, explicitly not an independent one
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

    # ---- what is being drawn
    dist = x[(x["level"] == 3) & x["TOT"].notna()]
    print(f"\n  {len(dist)} districts drawn, {int(dist['TOT'].sum()):,} people, "
          f"{int(dist['TOT'].sum()) / len(dist):,.0f} each. Categories, national:")
    for key, label in CATEGORIES:
        n = int(nat[key])
        print(f"    {n:>12,}  {100.0 * n / NATIONAL:6.3f}%  {label}")
    hin = int(nat["RLG_HIN"]) + int(nat["RLG_SCH"])
    print(f"    {hin:>12,}  {100.0 * hin / NATIONAL:6.3f}%  "
          f"Hinduism + Scheduled Castes, which taxonomy/pk2017.py merges")

    print(f"\n  the tehsil tier is normalised ({(x['level'] == 4).sum()} rows) and is NOT "
          f"drawn:\n     PBS publishes religion by district and its tehsil release has no "
          f"religion table,\n     so drawing it would be finer than the state's own "
          f"publication (§14.4).")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (XLSX, GDB):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run: python sources/pk.py --fetch")
    rows, x, negatives = read()
    check(x, negatives)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
