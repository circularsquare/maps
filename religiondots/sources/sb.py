"""Solomon Islands — SINSO, 2019 Census, Report Vol 2 Basic Tables, Table P8.3.

Reads (or fetches) data/raw/sb/ and writes data/normalized/sb.csv.

**RELIGION BY WARD, AND THE TABLE RECONCILES TO THE PERSON.** P8.3 is *Total population by
ward and religious denomination*: **183 wards**, seventeen denominations plus a total, and the
183 ward rows sum to the printed national row **exactly, on all eighteen columns**. So do the
ten province rows. Nothing here needs a tolerance, which is unusual enough on this map to be
worth saying (compare Vanuatu, §9bg §4, whose published table misses its own totals by up to
two).

720,956 people over 183 wards is **3,940 each**, the finest tier of any Pacific country here.

**TWO CHURCHES THAT NOTHING ELSE ON THIS MAP COUNTS.** The **Church of Melanesia** is 32.2% of
the country, the Anglican province of Melanesia and the largest body here. The **South Sea
Evangelical Church** is 17.3% and is the church of the Malaita labour trade: it grew out of the
Queensland Kanaka Mission, which evangelised Solomon Islanders working the Queensland cane
fields, and came home with them. **Malaita is 28.1% South Sea Evangelical and Isabel is 0.4%.**

**AND ONE FOUNDED HERE.** The **Christian Fellowship Church**, 16,179 people, is Silas Eto's
church, which broke from the Methodist mission in New Georgia in 1960. **It is 13.6% of Western
Province and essentially zero everywhere else** — 13,629 of its 16,179 members are in Western.

Usage:
    python sources/sb.py --fetch    two PDFs from statistics.gov.sb (~13 MB)
    python sources/sb.py            normalise from data/raw/sb/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sb")
OUT = os.path.join(ROOT, "data", "normalized", "sb.csv")

SOURCE_ID = "sb_nphc_2019"
YEAR = 2019
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# SINSO runs the same WP File Download plugin as Fiji and PNG; `id=0` on its AJAX route
# returns the whole 330-file library (reference: sources.md §11ab).
BASE = "https://statistics.gov.sb"
VOL2_NAME = "sb_2019_basic_tables_vol2.pdf"
VOL2_URL = (BASE + "/download/60/solomon-islands-2019-population-and-housing-census-"
            "national-report_volume-1-and-2/1207/"
            "solomon-islands-2019-census-report-vol-2_basic-tables_operations.pdf")
VOL1_NAME = "sb_2019_national_report_vol1.pdf"
VOL1_URL = (BASE + "/download/60/solomon-islands-2019-population-and-housing-census-"
            "national-report_volume-1-and-2/1208/"
            "solomon-islands-2019-population-and-housing-census_national-report-vol-1-2.pdf")

COLS = ["Total", "Church of Melanesia", "Roman Catholic",
        "South Sea Evangelical Church", "Seventh Day Adventist", "United Church",
        "Christian Fellowship Church", "Christian OutReach Church", "Pentecostal",
        "Jehovah's Witness", "Bahai Faith", "Assembly of God", "Muslim",
        "Baptist Church", "Other religions", "Custom Beliefs or Animism",
        "No Religion or Atheism", "Religion Faith/Refuse to Answer"]
TOTAL_CAT = "Total"
RESIDUALS = ["Religion Faith/Refuse to Answer"]
DRAWN = [c for c in COLS[1:] if c not in RESIDUALS]

# The ten province rows, which open each block of wards. `Honiara City Council` is the
# capital and is a province in its own right.
PROVINCE_NAMES = {"Choiseul", "Western", "Isabel", "Central", "Rennell-Bellona",
                  "Guadalcanal", "Malaita", "Makira-Ulawa", "Temotu",
                  "Honiara City Council"}

LABEL_X = 155.0
NUM = re.compile(r"^[\d][\d,]*$")

NATIONAL = 720_956
EXPECTED_WARDS = 183
EXPECTED_PROVINCES = 10

# Vol 1's Table 8.3.1, typeset separately from Vol 2, for the four largest bodies and the
# 1999/2009 series it also prints.
VOL1_TABLE831 = {
    "Church of Melanesia": 232_041, "Roman Catholic": 144_078,
    "South Sea Evangelical Church": 124_506, "Seventh Day Adventist": 83_452,
    "United Church": 66_915, "Christian Fellowship Church": 16_179,
    "Christian OutReach Church": 5_582,
}


def fetch():
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
    for name, url in ((VOL2_NAME, VOL2_URL), (VOL1_NAME, VOL1_URL)):
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print("already have", dest)
            continue
        print("GET", url)
        r = requests.get(url, headers=ua, timeout=1800, verify=False)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise SystemExit(f"SINSO returned something that is not a PDF for {name} "
                             f"({len(r.content):,} bytes)")
        if b"%%EOF" not in r.content[-4096:]:
            raise SystemExit(f"{name} has no %%EOF trailer -- truncated at source")
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(dest):,} bytes")


def _visual_rows(page, ytol=3.0):
    words = [w for w in page.get_text("words") if w[4].strip()]
    rows = []
    for x0, y0, x1, y1, txt, *_ in words:
        for r in rows:
            if abs(r[0] - y0) <= ytol:
                r[1].append((x0, x1, txt))
                break
        else:
            rows.append([y0, [(x0, x1, txt)]])
    for r in rows:
        r[1].sort(key=lambda t: t[0])
    rows.sort(key=lambda r: r[0])
    return rows


def _page_rows(doc, pno):
    """P8.3's data rows on one page, as (label, [(right_edge, token)]).

    **TWO BOUNDS ARE REQUIRED AND WITHOUT THEM THE LAST PAGE IS SILENTLY WRONG.** Page 163
    carries the tail of P8.3 *and* the whole of P8.4 (ethnic group by province, 13 columns).
    P8.4 has its own `Province` header, so taking the last one on the page lands on the wrong
    table; and its rows smear the right-edge distribution so that no column split separates.
    """
    page = doc[pno - 1]
    rows = _visual_rows(page)
    title_y = min((y for y, cells in rows
                   if any(t.startswith("P8.3") for x0, x1, t in cells)), default=0.0)
    stop_y = min((y for y, cells in rows
                  if any(re.match(r"^P8\.(?!3)\d", t) for x0, x1, t in cells)
                  and y > title_y), default=float("inf"))
    head_y = None
    for y, cells in rows:
        if (title_y < y < stop_y
                and any(t == "Province" and x0 < LABEL_X for x0, x1, t in cells)):
            head_y = y
            break
    if head_y is None:
        return []

    data = []
    for y, cells in rows:
        if y <= head_y + 3.0 or y >= stop_y:
            continue
        label = " ".join(t for x0, x1, t in cells if x0 < LABEL_X).strip()
        nums = [(x1, t) for x0, x1, t in cells
                if x0 >= LABEL_X and (NUM.match(t) or t == "-")]
        if not label or len(nums) < 5:
            continue
        data.append((label, nums))
    return data


def _split_columns(rows, ncols, pno):
    """Column centres, by cutting the sorted right edges at the ncols-1 largest gaps.

    The numbers are right-aligned, but the layout is NOT the same from page to page -- each
    page sizes its columns to its own widest figure -- so a fixed x map fails and pooling the
    pages merges neighbours. A per-page gap THRESHOLD fails too, because the last page holds
    only twelve wards and with that few samples the within-column spread swallows the
    between-column gaps. What is known for certain is the column COUNT, so cut on rank.
    """
    edges = sorted(x1 for _, nums in rows for x1, _ in nums)
    gaps = sorted(((edges[i + 1] - edges[i], i) for i in range(len(edges) - 1)),
                  reverse=True)
    between, within = gaps[ncols - 2][0], gaps[ncols - 1][0]
    if between <= within * 1.5:
        raise SystemExit(
            f"page {pno}: columns are not separable -- smallest between-column gap "
            f"{between:.1f} against largest within-column gap {within:.1f}")
    cuts = sorted(i for _, i in gaps[:ncols - 1])
    groups, start = [], 0
    for cut in cuts + [len(edges) - 1]:
        groups.append(edges[start:cut + 1])
        start = cut + 1
    return [sum(g) / len(g) for g in groups], between, within


def read():
    import fitz

    path = os.path.join(RAW, VOL2_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    doc = fitz.open(path)
    if doc.page_count == 0:
        raise SystemExit(f"{path}: PyMuPDF reports 0 pages")
    pages = [i + 1 for i in range(doc.page_count)
             if "P8.3:" in doc[i].get_text() and "Province" in doc[i].get_text()]

    rows = []
    for p in pages:
        got = _page_rows(doc, p)
        if not got:
            continue
        centres, between, within = _split_columns(got, len(COLS), p)
        print(f"  page {p}: {len(got):>3} rows, {len(COLS)} columns "
              f"(gap between {between:.0f} vs within {within:.0f})")
        for label, nums in got:
            vals = [None] * len(COLS)
            for x1, t in nums:
                j = min(range(len(COLS)), key=lambda k: abs(centres[k] - x1))
                if vals[j] is not None:
                    raise SystemExit(f"page {p} row {label!r}: two values in column {j}")
                vals[j] = 0 if t == "-" else int(t.replace(",", ""))
            if any(v is None for v in vals):
                raise SystemExit(f"page {p} row {label!r}: a column came out empty")
            rows.append((label, vals))
    doc.close()

    # Walk in printed order: a province row opens a block and its wards follow. SINSO's own
    # ward id is the province number with the ward number zero-padded to two -- Choiseul
    # ward 1 is `101`, Honiara ward 1 is `1001` -- which is what COD-AB carries as SINSO_WID.
    national, provinces, wards = None, [], []
    cur = None
    for label, vals in rows:
        m = re.match(r"^(\d+)\s+(.*)$", label)
        code, name = (m.group(1), m.group(2).strip()) if m else ("", label.strip())
        if name.lower() == "total":
            national = vals
            continue
        if name in PROVINCE_NAMES:
            cur = (code, name)
            provinces.append((code, name, vals))
            continue
        if cur is None:
            raise SystemExit(f"ward row {label!r} appears before any province row")
        wards.append((f"{int(cur[0])}{int(code):02d}", name, cur[0], cur[1], vals))
    if national is None:
        raise SystemExit("no `Total` row found in P8.3")
    return national, provinces, wards


def build(wards):
    rows = []
    for wid, name, pcode, pname, vals in wards:
        for cat, n in zip(COLS[1:], vals[1:]):
            note = f"level=ward; province={pname}"
            if cat in RESIDUALS:
                note += "; §3.5 residual, not drawn"
            rows.append({"geo_id": wid, "geo_level": "ward", "geo_name": name,
                         "source_category": cat, "count": n, "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})
        rows.append({"geo_id": wid, "geo_level": "ward", "geo_name": name,
                     "source_category": TOTAL_CAT, "count": vals[0], "basis": BASIS,
                     "year": YEAR, "source_id": SOURCE_ID,
                     "note": f"level=ward; province={pname}; universe total, not a "
                             "religion category"})
    return rows


def check(national, provinces, wards, rows):
    ok = True

    good = len(wards) == EXPECTED_WARDS
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} {len(wards)} wards (expected "
          f"{EXPECTED_WARDS})")
    good = len(provinces) == EXPECTED_PROVINCES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(provinces)} provinces: "
          + ", ".join(p[1] for p in provinces))

    ids = [w[0] for w in wards]
    good = len(set(ids)) == len(ids)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} all {len(ids)} ward ids are distinct")

    good = national[0] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {national[0]:,} "
          f"(expected {NATIONAL:,})")

    # ---- the arithmetic, and it is exact ----
    d = sum(national[1:]) - national[0]
    ok &= d == 0
    print(f"  {'OK ' if not d else 'BAD'} the seventeen categories sum to the national "
          f"total exactly ({d:+,})")

    for label, rowset in (("provinces", [p[2] for p in provinces]),
                          ("wards", [w[4] for w in wards])):
        s = [sum(r[i] for r in rowset) for i in range(len(COLS))]
        diff = [a - b for a, b in zip(s, national)]
        good = not any(diff)
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} the {len(rowset)} {label} sum to the national "
              f"row on {sum(1 for x in diff if not x)}/{len(COLS)} columns exactly")

    # every ward's own row must partition its own total
    bad = [w for w in wards if sum(w[4][1:]) != w[4][0]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} every ward's categories sum to its own total "
          f"({len(bad)} failures) {[(w[1], w[4][0], sum(w[4][1:])) for w in bad[:4]]}")

    # ---- witness: Volume 1's Table 8.3.1, typeset separately ----
    print("\n  the WITNESS — Volume 1's Table 8.3.1, a different volume of the same census:")
    bad = [(c, national[COLS.index(c)], w) for c, w in VOL1_TABLE831.items()
           if national[COLS.index(c)] != w]
    ok &= not bad
    print(f"    {'OK ' if not bad else 'BAD'} all {len(VOL1_TABLE831)} of its 2019 figures "
          f"match P8.3 exactly ({len(bad)} failures)")
    for c, g, w in bad[:6]:
        print(f"        {c}: Vol 2 {g:,} vs Vol 1 {w:,}")

    drawn = sum(national[COLS.index(c)] for c in DRAWN)
    resid = sum(national[COLS.index(c)] for c in RESIDUALS)
    print(f"\n  drawn {drawn:,} of {national[0]:,} — {100.0 * drawn / national[0]:.2f}%. "
          f"The {resid:,} not drawn are {', '.join(RESIDUALS)} (§3.5).")

    print("\n  the drawn categories, national:")
    for c in sorted(DRAWN, key=lambda c: -national[COLS.index(c)]):
        v = national[COLS.index(c)]
        print(f"    {v:>8,}  {100.0 * v / national[0]:6.2f}%  {c}")

    # ---- what the fine tier is for ----
    print("\n  the mission spheres, by province:")
    idx = {c: COLS.index(c) for c in COLS}
    print(f"    {'':<22}{'CoM':>7}{'RC':>7}{'SSEC':>7}{'SDA':>7}{'United':>8}{'CFC':>7}")
    for code, name, v in provinces:
        t = v[0]
        print(f"    {name[:22]:<22}"
              + "".join(f"{100.0 * v[idx[c]] / t:6.1f}%" for c in
                        ("Church of Melanesia", "Roman Catholic",
                         "South Sea Evangelical Church", "Seventh Day Adventist"))
              + f"{100.0 * v[idx['United Church']] / t:7.1f}%"
              + f"{100.0 * v[idx['Christian Fellowship Church']] / t:6.1f}%")

    cfc = idx["Christian Fellowship Church"]
    top = sorted(((w[4][cfc] / w[4][0], w) for w in wards if w[4][0] > 0), reverse=True)
    print("\n  `Christian Fellowship Church` was founded on New Georgia and stayed there:")
    for share, w in top[:5]:
        print(f"    {w[1]:<24} {w[3]:<12} {w[4][cfc]:>6,} of {w[4][0]:>6,}  "
              f"{100 * share:5.1f}%")

    cb = idx["Custom Beliefs or Animism"]
    top = sorted(((w[4][cb] / w[4][0], w) for w in wards if w[4][0] > 0), reverse=True)
    print("\n  `Custom Beliefs or Animism` is 0.57% nationally and concentrated:")
    for share, w in top[:5]:
        print(f"    {w[1]:<24} {w[3]:<12} {w[4][cb]:>6,} of {w[4][0]:>6,}  "
              f"{100 * share:5.1f}%")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    national, provinces, wards = read()
    rows = build(wards)
    check(national, provinces, wards, rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
