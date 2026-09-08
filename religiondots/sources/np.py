"""Nepal — NSO, National Population and Housing Census 2021 (NPHC 2078), religion Table 1.

Reads (or fetches) data/raw/np/ and writes data/normalized/np.csv.

**Ten named religions on 753 local levels for 29,164,578 people** — ~38,700 people per unit,
which is the finest counting geography of any large Asian country on this map. India is drawn
on 5,988 sub-districts at ~202,000 each and Bangladesh on 544 upazilas at ~306,000, so Nepal
counts religion five times finer than either of its neighbours and about as finely as
Mauritius does.

**IT IS AN EXACT PARTITION, WITH NO RESIDUAL AND NO `NOT STATED` CELL AT ALL.** The ten
categories sum to the row total on every one of the 918 rows, and the local levels, the
districts and the provinces each sum to 29,164,578. Nothing is suppressed, nothing rounded,
no `Other` box, no non-response — the fourth source here of which that is true, after
Zimbabwe, Malawi and Guyana, and by far the largest.

**THREE OF THE TEN ARE DRAWN ON NO OTHER MAP HERE, and each geography below was computed
before it was described** — the habit spec §12 now asks for, because the first draft of this
block guessed two of the three wrong from general knowledge:

    Kirat      924,204  (3.17%)  Kirat Mundhum, the religion of the Limbu, Rai, Yakkha
                                 and Sunuwar. Koshi province 16.8%, Panchthar district
                                 55.7%, Mahakulung 87.3%; 0.01% west of Bagmati.
    Prakriti   102,048  (0.35%)  "nature". NOT the Tharu Terai, where it is near zero:
                                 it is KHAM MAGAR country in the mid-western hills —
                                 Rukum East 16.6%, Rolpa 8.2%, Thawang 45.0%.
    Bon         67,223  (0.23%)  NOT the trans-Himalaya, where the Yungdrung Bon
                                 monasteries are: it is GANDAKI's Gurung middle hills —
                                 Manang 6.1%, Gorkha 5.7%, Dharche 32.7%. Probably the
                                 Tamu (Gurung) shamanic tradition more than monastic Bon;
                                 the census offers one box and cannot separate them.

Kirat is the third-largest count of a named indigenous religion here, after Sarna (4.96M) and
Gondi (1.03M), and the only one of the three a census asked about directly. **Bon is the only
Bon this map draws** — India's census names 697 in its C-01 Annexure under `Buddhist`, which
in2011.py excludes and draws nowhere. Tibet is the population that would really fill it and
China's source (§14.5) can see Tibetans but not their religion; Bhutan is not drawn.

**THE UNIVERSE IS THE WHOLE CENSUS POPULATION, NOT THE HOUSEHOLD POPULATION** — §3.7's
distinction, and Nepal lands on the good side of it. NSO tabulates 239,098 people (0.82%)
as `INSTITUTIONAL`, one row per district, and those rows are *additional* to the district's
local levels rather than spread through them. They are emitted here at `geo_level =
institutional` and countries.py does not draw them; see `sources/np.md` §5 for why, and the
`gap=` line in countries.py for what the reader is told. **This is the population §3.7 says
this map most wants to see** — Nepal's institutional population includes its monasteries and
gompas — and it is the one part of the table with no geography finer than the district.

**FOUR TIERS ARE READ AND THREE OF THEM ARE ONLY THE CHECK.** The sheet nests
NEPAL → province → district → local level by INDENTATION rather than by any code, so the
parse is a state machine over columns B, C and D. Reading the parent tiers as well is what
makes it checkable: every district must equal the sum of its own local levels plus its
institutional row, every province the sum of its districts, and Nepal the sum of its
provinces, on all eleven columns. A local level attached to the wrong district — the failure
this parse is actually exposed to — breaks the district identity while leaving every
national total intact.

**AND THE `Nepal` SHEET IS A FIFTH WITNESS THAT SHARES NO CELLS WITH THE FOURTH.** It is a
separate 14-row sheet in the same workbook carrying the national figures and their
percentages, written by a different hand: `Bahai` is printed `0` there against a real 537
people, which is a 1-dp percentage and not a count. Every national count is asserted against
it anyway, because a transposed column would survive all four nested identities above and
fail here.

TWO SPELLINGS ARE NSO's AND ARE KEPT VERBATIM in `source_category`, because the taxonomy keys
on them and a silent tidy-up is how a mapping stops matching: `Bouddha` for Buddhist and
`Sikha` for Sikh. `sources/np_geo.py` deals with the same habit on the geography side, where
the census writes `Metropolitian City`.

Usage:
    python sources/np.py --fetch    one 270 KB xlsx, seconds
    python sources/np.py           normalise from data/raw/np/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "np")
OUT = os.path.join(ROOT, "data", "normalized", "np.csv")

SOURCE_ID = "np_nphc_2021"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# censusresults.nsonepal.gov.np is a Next.js app and `censusnepal.cbs.gov.np` — the host
# every earlier note in sources.md names — now serves an 18-byte `<p>...working</p>` at its
# root and 404s every documented path. The live route was found by reading the SPA's own
# page chunks: the caste/ethnicity download list is a JSON literal inside
# `_next/static/chunks/pages/downloads/caste-ethnicity-*.js`, and the href it builds is
# `/files/caste/` + `xls_filename`. See sources/np.md §1.
XLSX_URL = ("https://censusresults.nsonepal.gov.np/files/caste/"
            "Religion_NPHC_2021.xlsx")
XLSX_NAME = "Religion_NPHC_2021.xlsx"

SHEET_UNITS = "Prov_District_local level"
SHEET_NATIONAL = "Nepal"

# NSO's own spelling and NSO's own print order. Both are load-bearing: the order is how the
# columns are taken, and the spelling is what taxonomy/np2021.py keys on.
RELIGIONS = ["Hindu", "Bouddha", "Islam", "Kirat", "Christian", "Prakriti", "Bon",
             "Jain", "Bahai", "Sikha"]
TOTAL_CAT = "Total Population"

NATIONAL = 29_164_578
EXPECTED_PROVINCES = 7
EXPECTED_DISTRICTS = 77
EXPECTED_LOCAL = 753          # 6 metropolitan + 11 sub-metropolitan + 276 municipalities
EXPECTED_INSTITUTIONAL = 77   #   + 460 rural municipalities; np_geo.py checks that split

# Column positions in SHEET_UNITS. The first five carry the label at exactly one
# indentation depth and are blank otherwise; column 5 is unused and column 6 is the total.
C_NEPAL, C_PROV, C_DIST, C_LOCAL, C_SEX = 0, 1, 2, 3, 4
C_TOTAL, C_FIRST_RELIGION = 6, 7


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, XLSX_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        print("already have", dest)
        return
    print("GET", XLSX_URL)
    r = requests.get(XLSX_URL, timeout=600, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        fh.write(r.content)
    # §12 shape 4 / sources.md §5a: assert the type, never the absence of an exception.
    # This host answers an unknown path with an HTML 404 page under a 200 in some routes.
    if r.content[:2] != b"PK":
        raise SystemExit(f"{dest} is not a zip container -- got {len(r.content):,} bytes "
                         f"starting {r.content[:16]!r}")
    print(f"  {os.path.getsize(dest):,} bytes")


def _label(v):
    return str(v).strip() if v not in (None, "") else None


def read():
    """Walk SHEET_UNITS and return (rows, tiers, national_sheet).

    `tiers` maps geo_id -> dict of the eleven figures, for every tier including the
    parents, so check() can test the nesting. `rows` is only what gets written.
    """
    import openpyxl

    src = os.path.join(RAW, XLSX_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
    if SHEET_UNITS not in wb.sheetnames or SHEET_NATIONAL not in wb.sheetnames:
        raise SystemExit(f"expected sheets {SHEET_UNITS!r} and {SHEET_NATIONAL!r}, "
                         f"found {wb.sheetnames}")

    # ---- the header, asserted before a single figure is taken. The ten religion labels
    # sit one row below `Religion` and are the only thing that says which column is which.
    grid = list(wb[SHEET_UNITS].iter_rows(values_only=True))
    header = [_label(v) for v in grid[3][C_FIRST_RELIGION:C_FIRST_RELIGION + 10]]
    if header != RELIGIONS:
        raise SystemExit(f"religion header changed: {header} != {RELIGIONS}")
    if _label(grid[2][C_TOTAL]) != TOTAL_CAT:
        raise SystemExit(f"total column header is {grid[2][C_TOTAL]!r}, "
                         f"expected {TOTAL_CAT!r}")

    tiers = {}
    order = []
    prov = dist = local = None
    prov_n = dist_n = local_n = 0
    key = level = name = None

    for r in grid[4:]:
        if _label(r[C_NEPAL]) == "NEPAL":
            key, level, name = "NP", "country", "Nepal"
            prov = dist = local = None
            continue
        if _label(r[C_PROV]):
            prov_n += 1
            dist_n = 0
            prov = f"NP-{prov_n:d}"
            key, level, name = prov, "province", _label(r[C_PROV])
            dist = local = None
            continue
        if _label(r[C_DIST]):
            dist_n += 1
            local_n = 0
            dist = f"{prov}-{dist_n:02d}"
            key, level, name = dist, "district", _label(r[C_DIST])
            local = None
            continue
        if _label(r[C_LOCAL]):
            label = _label(r[C_LOCAL])
            if label.upper() == "INSTITUTIONAL":
                # One per district, carrying no local level of its own. Given the district's
                # id with an `-I` suffix so it is obviously not a 754th palika.
                local = f"{dist}-I"
                level = "institutional"
                name = f"{tiers[dist]['name']} (institutional)"
            else:
                local_n += 1
                local = f"{dist}-{local_n:02d}"
                level = "local"
                name = label
            key = local
            continue
        if _label(r[C_SEX]) != "Total":
            continue                       # Male / Female rows; the sex split is not drawn
        if key is None:
            raise SystemExit(f"a Total row before any area label: {r[:7]}")
        if key in tiers:
            raise SystemExit(f"duplicate Total row for {key} ({name})")
        vals = {TOTAL_CAT: r[C_TOTAL]}
        for i, cat in enumerate(RELIGIONS):
            vals[cat] = r[C_FIRST_RELIGION + i]
        for cat, v in vals.items():
            if not isinstance(v, (int, float)) or v != int(v) or v < 0:
                raise SystemExit(f"{key} ({name}) {cat}: {v!r} is not a count")
            vals[cat] = int(v)
        tiers[key] = {"level": level, "name": name, "prov": prov, "dist": dist, **vals}
        order.append(key)

    wb_nat = list(wb[SHEET_NATIONAL].iter_rows(values_only=True))
    wb.close()

    # ---- the national sheet, read independently of everything above
    national = {}
    for row in wb_nat:
        lab = _label(row[0])
        if lab in ("Total", "Total "):
            national[TOTAL_CAT] = int(row[1])
        elif lab in RELIGIONS:
            national[lab] = int(row[1])

    rows = []
    for key in order:
        t = tiers[key]
        if t["level"] not in ("local", "institutional"):
            continue                        # parents are the check, not the data
        note = f"level={t['level']}; district={tiers[t['dist']]['name']}"
        if t["level"] == "institutional":
            note += ("; NSO tabulates the institutional population at DISTRICT level only "
                     "-- no local level is given, and it is not part of any local level's "
                     "figure (spec §3.7)")
        for cat in [TOTAL_CAT] + RELIGIONS:
            rows.append({
                "geo_id": key,
                "geo_level": t["level"],
                "geo_name": t["name"],
                "source_category": cat,
                "count": t[cat],
                "basis": BASIS,
                "year": YEAR,
                "source_id": SOURCE_ID,
                "note": note + ("; universe total, not a religion category"
                                if cat == TOTAL_CAT else ""),
            })
    return rows, tiers, national


def check(rows, tiers, national):
    ok = True
    by_level = {}
    for k, t in tiers.items():
        by_level.setdefault(t["level"], []).append(k)

    for level, want in (("province", EXPECTED_PROVINCES),
                        ("district", EXPECTED_DISTRICTS),
                        ("local", EXPECTED_LOCAL),
                        ("institutional", EXPECTED_INSTITUTIONAL)):
        got = len(by_level.get(level, []))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {level:<14} {got:>5} units (expected {want})")

    good = tiers["NP"][TOTAL_CAT] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national universe {tiers['NP'][TOTAL_CAT]:,} "
          f"(expected {NATIONAL:,})")

    # ---- 1. the ten religions sum to the row total, on EVERY row of every tier.
    cells = [k for k in tiers
             if sum(tiers[k][c] for c in RELIGIONS) != tiers[k][TOTAL_CAT]]
    ok &= not cells
    print(f"  {'OK ' if not cells else 'BAD'} the 10 religions sum to Total on all "
          f"{len(tiers)} rows ({len(cells)} failures) {cells[:4]}")
    print("      an EXACT partition -- NSO publishes no `Other`, no `Not stated` and no")
    print("      residual of any kind, so 100% of the census is in a named category.")

    # ---- 2. the nesting, on all eleven columns. This is the check that catches a local
    # level attached to the wrong district, which every national identity would survive.
    for child, parent, field in (("local+institutional", "district", "dist"),
                                 ("district", "province", "prov"),
                                 ("province", "country", None)):
        agg = {}
        kids = ([k for k in tiers if tiers[k]["level"] in ("local", "institutional")]
                if field == "dist" else
                [k for k in tiers if tiers[k]["level"] == child])
        for k in kids:
            p = tiers[k][field] if field else "NP"
            a = agg.setdefault(p, dict.fromkeys([TOTAL_CAT] + RELIGIONS, 0))
            for c in [TOTAL_CAT] + RELIGIONS:
                a[c] += tiers[k][c]
        bad = [(p, c, v[c], tiers[p][c]) for p, v in agg.items()
               for c in [TOTAL_CAT] + RELIGIONS if v[c] != tiers[p][c]]
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} {child} sums to its {parent} on all 11 "
              f"columns, {len(agg)} parents ({len(bad)} failures)")
        for p, c, s, w in bad[:4]:
            print(f"        {p} ({tiers[p]['name']}) {c}: {s:,} vs {w:,}")

    # ---- 3. the second sheet. Shares no cell with the first and is written differently:
    # `Bahai` is printed there as `0` in the PERCENTAGE column, so only column B is used.
    bad = [(c, national.get(c), tiers["NP"][c]) for c in [TOTAL_CAT] + RELIGIONS
           if national.get(c) != tiers["NP"][c]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the `Nepal` sheet agrees on all 11 national "
          f"figures ({len(bad)} failures)")
    for c, a, b in bad[:4]:
        print(f"        {c}: sheet says {a}, units give {b:,}")

    # ---- 4. what is drawn and what is not
    inst = sum(tiers[k][TOTAL_CAT] for k in by_level.get("institutional", []))
    drawn = NATIONAL - inst
    print(f"\n  institutional (NOT drawn, district-level only): {inst:,} "
          f"({inst / NATIONAL:.2%})")
    print(f"  drawn on 753 local levels: {drawn:,} ({drawn / NATIONAL:.2%}), "
          f"{drawn / EXPECTED_LOCAL:,.0f} people per unit")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in [TOTAL_CAT] + RELIGIONS:
        n = tiers["NP"][cat]
        mark = "  <- universe" if cat == TOTAL_CAT else ""
        print(f"    {n:>12,}  {100.0 * n / NATIONAL:6.2f}%  {cat}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, tiers, national = read()
    check(rows, tiers, national)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
