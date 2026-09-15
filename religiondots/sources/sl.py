"""Sierra Leone — 2015 Population and Housing Census, religion by district.

Reads (or fetches) data/raw/sl/sl_2015_phc_thematic_report_on_pop_structure_and_pop_distribution.pdf
and writes data/normalized/sl.csv. `sources/sl.md` is the write-up; `sources/sl_geo.py` builds the
fourteen 2015 districts and the grid.

## THE TABLE

Statistics Sierra Leone, *Sierra Leone 2015 Population and Housing Census: Thematic Report on
Population Structure and Population Distribution* (October 2017), **Table 5.3, "Religious
composition of household population in Sierra Leone by regions and districts"**, PDF pp.37-38:
the nation, the four regions and the fourteen districts, six answers, one decimal.

    Christianity  Islam  Bahai  Traditional  Other  No Religion

Nothing finer is published. The National Analytical Report prints religion nationally only
(Table 3.25, denominations; Table 4.6, households by the head's religion); the other thematic
reports and the Census Atlas were not found to cross religion with chiefdom (sources/sl.md §1).

## THE TABLE IS PERCENTAGES, SO THE COUNTS ARE THIS MODULE'S ARITHMETIC

    shares        Table 5.3, district x religion, one decimal            this PDF, pp.37-38
    denominators  Table 2.2, total population by district, counts        this PDF, p.17
    universe      household population 7,076,119 of 7,092,113            National Analytical
                                                                          Report, Tables 4.1, 4.6

Each district's six shares are divided by their own sum (99.9 to 100.1 as printed) and applied to
its total population times 7,076,119 / 7,092,113. No per-religion rescale follows, because no
national count by religion exists in persons: Table 3.25 is percentages too, and Table 4.6 counts
the people living in households by the HEAD's religion, which is a different quantity.

The 15,994 people enumerated in institutions are in no religion table, and the office does not
print them by district, so the household share is taken as uniform. At 0.23% nationally it moves
no district by more than a fraction of a percent of its size.

## THE NATIONAL ROW MISPRINTS BAHAI AS 0.5

Table 5.3's `Sierra Leone` row sums to 100.5. Every district row prints Bahai as 0.0 or 0.1 and
the population-weighted district figure is 0.04; the analytical report's Table 3.25 prints 0.0.
check() pins the misprint rather than correcting it, since nothing drawn reads the national row.

## THE QUESTIONNAIRE

The 2015 household form (IPUMS enumeration materials, `enum_form_sl2015a.pdf`, p.1) asks P05
*What is (NAME's) religion?*, "Write the code of the religion, use code list". The code list
(`Census_2015_tools/2015-slphc_codes_list_web.pdf`, Wayback 20201114085519) gives eleven codes:
01 Catholic, 02 Anglican, 03 Methodist, 04 SDA, 05 Pentecostal, 06 Other Christian, 07 Islam,
08 Bahai, 09 Traditional, 10 Other, 11 No Religion. Table 5.3 is those codes with the six
Christian ones added together, in code order. There is no not-stated code, so any blank answer is
inside one of the six columns, most plausibly `Other`.

Usage:
    python sources/sl.py --fetch    one ~4.2 MB PDF from statistics.sl (Wayback as fallback)
    python sources/sl.py            normalise from data/raw/sl/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sl")
OUT = os.path.join(ROOT, "data", "normalized", "sl.csv")
sys.path.insert(0, HERE)

from fetch_checks import FetchCheckError, check_body, digest   # noqa: E402  shared, not copied

SOURCE_ID = "sl_phc2015_thematic_pop_structure_t53"
YEAR = 2015
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

_PATH = ("www.statistics.sl/images/StatisticsSL/Documents/Census/2015/"
         "sl_2015_phc_thematic_report_on_pop_structure_and_pop_distribution.pdf")
URL = "https://" + _PATH
WAYBACK = "https://web.archive.org/web/20190410152831id_/https://" + _PATH
PDF = os.path.join(RAW, os.path.basename(_PATH))
# The live file on 2026-09-15 and the Wayback capture of 2019-04-10 are the same bytes; this is
# the CDX digest of that capture.
PIN_DIGEST = "H4DQRV6Q6CHUWXAOK43BI4NRANWBZ7J2"
PIN_SIZE = 4_192_879
PAGES = 54

# 0-based page indices.
PAGE_PROSE_HH = 14    # PDF p15, "99.8 per cent of the population enumerated in households"
PAGE_T22 = 16         # PDF p17, Table 2.2 (continued), district totals
PAGE_T32 = 19         # PDF p20, Table 3.2, national total
PAGE_T53 = (36, 37)   # PDF pp37-38, Table 5.3 and its continuation
PAGE_T54 = 37         # PDF p38, Table 5.4, region totals

CATS = ["Christianity", "Islam", "Bahai", "Traditional", "Other", "No Religion"]

# Table 5.3, transcribed, and asserted equal to what is parsed off the page.
T53_NATIONAL = (21.9, 77.0, 0.5, 0.1, 0.7, 0.3)
T53_REGIONS = {
    "Eastern":      (29.0, 69.4, 0.1, 0.1, 1.2, 0.2),
    "Northern":     (13.7, 85.1, 0.0, 0.0, 0.6, 0.5),
    "Southern":     (19.2, 79.9, 0.0, 0.0, 0.6, 0.3),
    "Western Area": (30.1, 69.1, 0.1, 0.0, 0.6, 0.1),
}
T53 = {
    "Kailahun":           (34.0, 64.0, 0.1, 0.2, 1.4, 0.3),
    "Kenema":             (12.8, 86.7, 0.0, 0.0, 0.4, 0.1),
    "Kono":               (43.5, 54.3, 0.1, 0.2, 1.8, 0.1),
    "Bombali":            (26.9, 71.4, 0.0, 0.0, 1.2, 0.4),
    "Kambia":             (5.3, 94.1, 0.0, 0.0, 0.3, 0.3),
    "Koinadugu":          (12.7, 86.6, 0.0, 0.0, 0.4, 0.2),
    "Port Loko":          (5.9, 92.8, 0.0, 0.0, 0.5, 0.8),
    "Tonkolili":          (14.0, 84.9, 0.0, 0.1, 0.5, 0.4),
    "Bo":                 (26.9, 72.1, 0.0, 0.0, 0.7, 0.2),
    "Bonthe":             (13.1, 85.4, 0.0, 0.0, 1.4, 0.1),
    "Moyamba":            (24.9, 74.2, 0.0, 0.0, 0.4, 0.4),
    "Pujehun":            (4.8, 94.6, 0.0, 0.0, 0.1, 0.5),
    "Western Area Rural": (27.3, 72.0, 0.1, 0.0, 0.6, 0.1),
    "Western Area Urban": (31.3, 67.9, 0.1, 0.0, 0.6, 0.1),
}

# Table 2.2 (continued), 2015 total population by district.
T22 = {
    "Kailahun": 526_379, "Kenema": 609_891, "Kono": 506_100,
    "Bombali": 606_544, "Kambia": 345_474, "Koinadugu": 409_372, "Port Loko": 615_376,
    "Tonkolili": 531_435,
    "Bo": 575_478, "Bonthe": 200_781, "Moyamba": 318_588, "Pujehun": 346_461,
    "Western Area Rural": 444_270, "Western Area Urban": 1_055_964,
}
REGION_OF = {
    "Kailahun": "Eastern", "Kenema": "Eastern", "Kono": "Eastern",
    "Bombali": "Northern", "Kambia": "Northern", "Koinadugu": "Northern",
    "Port Loko": "Northern", "Tonkolili": "Northern",
    "Bo": "Southern", "Bonthe": "Southern", "Moyamba": "Southern", "Pujehun": "Southern",
    "Western Area Rural": "Western Area", "Western Area Urban": "Western Area",
}
# Table 5.4, total population by region, printed on the same page as Table 5.3's continuation.
T54_REGIONS = {"Eastern": 1_642_370, "Northern": 2_508_201, "Southern": 1_441_308,
               "Western Area": 1_500_234}
TOTAL = 7_092_113

# National Analytical Report (2015_census_national_analytical_report.pdf), Table 4.1, p.111 as
# printed (PDF p.153), and Table 4.6's total, PDF p.158: the household population, all ages.
HOUSEHOLD = 7_076_119
INSTITUTIONAL = TOTAL - HOUSEHOLD

# The same report's Table 3.25 (PDF p.137), national, transcribed and not parsed (that PDF is
# 12 MB and not fetched here). Its Christian block has a stray row: `SDA 8.0` with a sex ratio of
# 210.2, then an unlabelled row of 0.7. Without the 8.0 row the six Christian codes sum to 21.9,
# Table 5.3's Christianity, so 0.7 is SDA. Only used as a witness to the national row.
T325_CHRISTIAN = {"Catholic": 7.0, "Anglican": 1.2, "Methodist": 3.0, "SDA": 0.7,
                  "Pentecostal": 5.3, "Other Christian": 4.7}
T325_STRAY = 8.0
T325_OTHER = {"Islam": 77.0, "Bahai": 0.0, "Traditional": 0.1, "Other": 0.7,
              "No Religion": 0.3}

PCT = re.compile(r"^\d{1,3}\.\d$")
SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")


def despace(s):
    return re.sub(r"\s+", " ", str(s).translate(SPACES)).strip()


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) == PIN_SIZE:
        print("already have", PDF)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
    for url in (URL, WAYBACK):
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
        except Exception as e:                       # noqa: BLE001 - try the archive next
            print(f"  {url[:70]}...: {e}")
            continue
        try:
            check_body(body, "pdf", where=f"{url[:70]}...", pin_size=PIN_SIZE,
                       pin_digest=PIN_DIGEST)
        except FetchCheckError as e:
            print(f"  {e}")
            continue
        with open(PDF + ".part", "wb") as fh:
            fh.write(body)
        os.replace(PDF + ".part", PDF)
        print(f"wrote {PDF} ({len(body):,} bytes) from {url[:40]}")
        return
    raise SystemExit("neither statistics.sl nor the Wayback copy returned the pinned PDF")


def _lines(doc, pno):
    return [ln for ln in (despace(x) for x in doc.load_page(pno).get_text().splitlines()) if ln]


def read_t53(doc):
    """Table 5.3 off both pages: name lines (a name can wrap, `Western` / `Area Rural`), then
    six one-decimal percentages. `District` is a heading row with no numbers."""
    rows = {}
    for pno in PAGE_T53:
        lines = _lines(doc, pno)
        start = next((i for i, ln in enumerate(lines) if ln.startswith("Table 5.3")), None)
        if start is None:
            raise SystemExit(f"no `Table 5.3` caption on page index {pno}")
        hdr = next((i for i in range(start, len(lines)) if lines[i] == "No Religion"), None)
        if hdr is None:
            raise SystemExit(f"Table 5.3 on page index {pno} has no `No Religion` header")
        end = next((i for i in range(hdr, len(lines))
                    if lines[i].startswith("Table 5.4") or lines[i].startswith("Source")),
                   len(lines))
        name, nums = [], []
        for ln in lines[hdr + 1:end]:
            if PCT.match(ln):
                nums.append(float(ln))
                if len(nums) == 6:
                    rows[" ".join(name)] = tuple(nums)
                    name, nums = [], []
            else:
                if nums:
                    raise SystemExit(f"Table 5.3 row {' '.join(name)!r} broke after {nums}")
                if ln == "District":
                    continue
                name.append(ln)
    return rows


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Sierra Leone — 2015 PHC, Thematic Report on Population Structure, Table 5.3\n")
    say(doc.page_count == PAGES, f"the report is {doc.page_count} pages (expected {PAGES})")
    with open(PDF, "rb") as fh:
        d = digest(fh.read())
    say(d == PIN_DIGEST, f"the PDF's digest is the pinned {PIN_DIGEST} ({d})")

    # 1. the parsed table is the transcribed one
    parsed = read_t53(doc)
    want = {"Sierra Leone": T53_NATIONAL, **T53_REGIONS, **T53}
    say(parsed == want,
        f"Table 5.3 parsed off pp.37-38: {len(parsed)} rows (nation, 4 regions, 14 districts), "
        "identical to the transcription")
    if parsed != want:
        for k in sorted(set(parsed) | set(want)):
            if parsed.get(k) != want.get(k):
                print(f"        {k!r}: page {parsed.get(k)} transcribed {want.get(k)}")

    # 2. rows close; the national row's misprint is pinned
    rows = list(T53.values()) + list(T53_REGIONS.values())
    worst = max(abs(sum(v) - 100.0) for v in rows)
    say(worst <= 0.30 + 1e-9, f"every district and region row sums to 100 within {worst:.2f} pp "
        "(bound 0.30 = 6 cells x 0.05)")
    nat = sum(T53_NATIONAL)
    fixed = sum(T53_NATIONAL) - T53_NATIONAL[CATS.index("Bahai")]
    say(abs(nat - 100.5) < 1e-9 and abs(fixed - 100.0) < 1e-9,
        f"the national row sums to {nat:.1f}, and to {fixed:.1f} with its Bahai 0.5 read as 0.0")

    # 3. populations: printed, and adding up
    t22 = despace(" ".join(_lines(doc, PAGE_T22)))
    found = [k for k, v in T22.items() if f"{v:,}" in t22]
    say(len(found) == 14, f"all 14 district totals appear on Table 2.2's page ({len(found)})")
    t54 = despace(" ".join(_lines(doc, PAGE_T54)))
    found = [k for k, v in T54_REGIONS.items() if f"{v:,}" in t54]
    say(len(found) == 4 and f"{TOTAL:,}" in t54,
        f"the 4 region totals and {TOTAL:,} appear in Table 5.4 ({len(found)})")
    say(f"{TOTAL:,}" in despace(" ".join(_lines(doc, PAGE_T32))),
        "and the national total again in Table 3.2")
    for reg, n in T54_REGIONS.items():
        s = sum(v for k, v in T22.items() if REGION_OF[k] == reg)
        say(s == n, f"{reg}'s districts sum to {s:,} = Table 5.4's {n:,}")
    say(sum(T22.values()) == TOTAL, f"the 14 districts sum to {sum(T22.values()):,}")
    prose = despace(" ".join(_lines(doc, PAGE_PROSE_HH)))
    say(round(100.0 * HOUSEHOLD / TOTAL, 1) == 99.8 and "99.8 per cent" in prose
        and "enumerated in households" in prose,
        f"household population {HOUSEHOLD:,} is {100.0 * HOUSEHOLD / TOTAL:.2f}% of {TOTAL:,}, "
        "the report's printed 99.8 per cent")

    # 4. district shares, population-weighted, reproduce the region rows and the nation
    for reg, printed in T53_REGIONS.items():
        ds = [k for k in T53 if REGION_OF[k] == reg]
        pop = sum(T22[k] for k in ds)
        w = [sum(T53[k][i] * T22[k] for k in ds) / pop for i in range(6)]
        worst = max(abs(a - b) for a, b in zip(w, printed))
        say(worst <= 0.10 + 1e-9, f"{reg:<13} weighted "
            + " ".join(f"{x:5.2f}" for x in w) + f" against {printed}, worst {worst:.2f}")
    w = [sum(T53[k][i] * T22[k] for k in T53) / TOTAL for i in range(6)]
    ib = CATS.index("Bahai")
    worst = max(abs(a - b) for i, (a, b) in enumerate(zip(w, T53_NATIONAL)) if i != ib)
    say(worst <= 0.10 + 1e-9 and w[ib] < 0.05,
        "nation weighted " + " ".join(f"{x:5.2f}" for x in w)
        + f" against {T53_NATIONAL}: worst {worst:.2f} apart from Bahai, whose weighted "
        f"{w[ib]:.3f} rounds to the 0.0 of Table 3.25 and not the printed 0.5")

    # 5. the analytical report's national table, as a witness
    say(abs(sum(T325_CHRISTIAN.values()) - T53_NATIONAL[0]) < 1e-9
        and abs(sum(T325_CHRISTIAN.values()) + T325_STRAY - 29.9) < 1e-9,
        f"Table 3.25's six Christian codes sum to {sum(T325_CHRISTIAN.values()):.1f} = Table "
        f"5.3's Christianity once its stray {T325_STRAY} row is left out (29.9 with it)")
    same = all(T325_OTHER[c] == T53_NATIONAL[CATS.index(c)] for c in CATS[1:] if c != "Bahai")
    say(same and T325_OTHER["Bahai"] == 0.0,
        "Table 3.25's Islam, Traditional, Other and No Religion equal Table 5.3's national row; "
        "its Bahai is 0.0")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def emit():
    scale = HOUSEHOLD / TOTAL
    rows = []
    for dist, shares in T53.items():
        hh = T22[dist] * scale
        s = sum(shares)
        for c, p in zip(CATS, shares):
            n = int(round(hh * p / s))
            if n <= 0:
                continue
            rows.append({
                "geo_id": dist, "geo_level": "district", "geo_name": dist,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID,
                "note": (f"Table 5.3 pct={p:.1f} of a row summing to {s:.1f}; Table 2.2 "
                         f"pop={T22[dist]} x {scale:.6f} household share"),
            })
    return rows


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing — run: python sources/sl.py --fetch")
    doc = fitz.open(PDF)
    check(doc)
    rows = emit()

    total = sum(r["count"] for r in rows)
    print(f"\n  14 districts, {total:,} people ({total - HOUSEHOLD:+,} against the household "
          f"population {HOUSEHOLD:,}), {total / 14:,.0f} each")
    by_cat, by_dist = {}, {}
    for r in rows:
        by_cat[r["source_category"]] = by_cat.get(r["source_category"], 0) + r["count"]
        by_dist.setdefault(r["geo_name"], {})[r["source_category"]] = r["count"]
    for c in CATS:
        n = by_cat.get(c, 0)
        print(f"    {n:>11,}  {100.0 * n / total:6.3f}%  {c}")
    print(f"\n  {'district':<20}{'drawn':>11}   Christian share of the country's Christians")
    chr_tot = by_cat["Christianity"]
    for dist, d in by_dist.items():
        print(f"  {dist:<20}{sum(d.values()):>11,}   {100.0 * d['Christianity'] / chr_tot:5.1f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
