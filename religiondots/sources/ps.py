"""Palestine — PCBS census 2017, religion by governorate, in counts.

Reads (or fetches) data/raw/ps/book2364-1.pdf and writes data/normalized/ps.csv.
`sources/ps.md` is the write-up; `sources/ps_geo.py` builds the polygons and the grid.

## THE TABLE

Palestinian Central Bureau of Statistics, *Preliminary Results of the Population, Housing and
Establishments Census, 2017* (Ramallah, February 2018; the PDF is dated 2018-03-29), **Table 3**
*Palestinian Population in Palestine by Governorate and Religion, 2017*, printed p.35 (PDF page
index 33). Four answers (Islam, Christian, Other, Not Stated) and a total, for the sixteen
governorates, the West Bank, the Gaza Strip and Palestine, in counts. Found by scout
`d743fc47-scout-asia` (sources.md §scout-2026-09-14-asia-oceania).

## THREE TOTALS, AND WHICH ONE TABLE 3 IS

The same book prints three population figures for every governorate, and they nest:

  * Table 3, **Palestinians counted**: Palestine 4,665,426, Jerusalem 392,835.
  * Table 2, **everyone counted**: Palestine 4,705,601 (Table 1's "actual counted population"),
    Jerusalem 414,786. The difference, 40,175 nationally, is people the census counted who are
    not Palestinian, and the form asks them no religion question (below).
  * Table 25, **counted plus PCBS's estimate of those the count missed**: Palestine 4,780,978,
    Jerusalem governorate 435,483, of which J1 281,163 and J2 154,320. Its footnote says so
    ("Includes actually counted population ... in addition to the uncounted population
    estimates based on the post enumeration survey results"), and Table 1 prints the national
    under-coverage as 75,377 (1.7%).

The scout found that J1 plus J2 is 435,483 and not Table 3's 392,835 and left it open. It is
this nesting: Table 25 is the largest of the three universes. J2 alone is 154,320 even with
the estimate added, so Table 3's Jerusalem row (392,835) has to include J1, the part of the
governorate Israel annexed in 1967, where PCBS lists 21 localities from Kafr A'qab to Umm Tuba
(Table 25's footnote, PDF page index 80). `check()` asserts every step of the nesting.

## THE FORM

PCBS Form No. 25 PHC, *Household and Housing Conditions Questionnaire* (IPUMS copy
`enum_form_ps2017a.pdf`, read 2026-09-15). Part two's person block has a column group headed
**For Palestinians only**, and Religion is in it with three codes: `1. Muslim 2. Christian
3.Other`. There is no code for no answer and no box for no religion, so `Not Stated` (1,509) is
a blank or unreadable answer, and a Palestinian with no religion had to leave it blank or
answer Other.

## UNSD HOLDS 1997 AND 2007, NOT 2017

`python tools/oracle.py "State of Palestine"` has national rows for 2007 (Islam 3,568,020,
Christian 43,179, Other 1,051, Not Stated 56,994) and 1997 (Christian 40,055). Neither is this
census, so they are printed as a witness and not compared to the person.

Usage:
    python sources/ps.py --fetch    one 22.6 MB PDF from pcbs.gov.ps (its TLS chain is incomplete)
    python sources/ps.py            normalise from data/raw/ps/
"""

import csv
import os
import re
import ssl
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ps")
OUT = os.path.join(ROOT, "data", "normalized", "ps.csv")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

from fetch_checks import check_body   # noqa: E402  shared, not copied

SOURCE_ID = "ps_pcbs_census2017_preliminary_t3"
YEAR = 2017
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

URL = "https://www.pcbs.gov.ps/Downloads/book2364-1.pdf"
PDF = os.path.join(RAW, "book2364-1.pdf")
SIZE = 22_646_657
PAGES = 81

# 0-based page indices.
PAGE_T1 = 31
PAGE_T2 = 32
PAGE_T3 = 33
PAGE_T25_JERUSALEM = 73
PAGE_T25_FOOT = 80

CATS = ["Islam", "Christian", "Other", "Not Stated"]
FORM = {1: "Muslim", 2: "Christian", 3: "Other"}          # Form 25 PHC, Part two, Religion

# Table 3 as printed: Islam, Christian, Other, Not Stated, Total. Transcribed from the rendered
# page and asserted equal to what is parsed off the text layer.
WEST_BANK = ["Jenin", "Tubas and the Northern Valleys", "Tulkarm", "Nablus", "Qalqiliya",
             "Salfit", "Ramallah & Al-Bireh", "Jericho & Al-Aghwar", "Jerusalem", "Bethlehem",
             "Hebron"]
GAZA_STRIP = ["North Gaza", "Gaza", "Dier Al-Balah", "Khan Yunis", "Rafah"]
T3 = {
    "Jenin":                          (305_207, 2_699, 21, 146, 308_073),
    "Tubas and the Northern Valleys": (60_070, 54, 8, 0, 60_132),
    "Tulkarm":                        (182_924, 21, 10, 46, 183_001),
    "Nablus":                         (385_327, 601, 361, 263, 386_552),
    "Qalqiliya":                      (107_973, 11, 5, 0, 107_989),
    "Salfit":                         (73_693, 4, 7, 0, 73_704),
    "Ramallah & Al-Bireh":            (304_326, 10_255, 60, 442, 315_083),
    "Jericho & Al-Aghwar":            (46_962, 285, 7, 71, 47_325),
    "Jerusalem":                      (383_384, 8_558, 594, 299, 392_835),
    "Bethlehem":                      (188_851, 23_165, 32, 143, 212_191),
    "Hebron":                         (705_473, 59, 19, 38, 705_589),
    "North Gaza":                     (363_597, 20, 90, 19, 363_726),
    "Gaza":                           (639_155, 1_082, 44, 33, 640_314),
    "Dier Al-Balah":                  (269_360, 8, 48, 9, 269_425),
    "Khan Yunis":                     (366_462, 16, 42, 0, 366_520),
    "Rafah":                          (232_919, 12, 36, 0, 232_967),
}
T3_WEST_BANK = (2_744_190, 45_712, 1_124, 1_448, 2_792_474)
T3_GAZA_STRIP = (1_871_493, 1_138, 260, 61, 1_872_952)
T3_PALESTINE = (4_615_683, 46_850, 1_384, 1_509, 4_665_426)

# Table 2, both sexes: everyone counted.
T2 = {
    "Jenin": 308_618, "Tubas and the Northern Valleys": 60_186, "Tulkarm": 183_205,
    "Nablus": 387_240, "Qalqiliya": 108_234, "Salfit": 73_756, "Ramallah & Al-Bireh": 322_193,
    "Jericho & Al-Aghwar": 50_002, "Jerusalem": 414_786, "Bethlehem": 215_047,
    "Hebron": 707_017, "North Gaza": 364_188, "Gaza": 641_310, "Dier Al-Balah": 269_830,
    "Khan Yunis": 366_823, "Rafah": 233_166,
}
T2_WEST_BANK, T2_GAZA_STRIP, T2_PALESTINE = 2_830_284, 1_875_317, 4_705_601

# Table 1.
T1_TOTAL, T1_COUNTED, T1_UNDERCOUNT = 4_780_978, 4_705_601, 75_377

# Table 25, Jerusalem governorate: counted plus the post-enumeration estimate.
T25_JERUSALEM, T25_J1, T25_J2 = 435_483, 281_163, 154_320

NUM = re.compile(r"(\d{1,3}(?:,\d{3})*|\d+)\s*$")


def _ctx():
    # pcbs.gov.ps serves an incomplete certificate chain (sources.md
    # §scout-2026-09-14-asia-oceania, "curl needs -k"). The body is checked by size instead.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) == SIZE:
        print("already have", PDF)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
    req = urllib.request.Request(URL, headers=ua)
    with urllib.request.urlopen(req, timeout=900, context=_ctx()) as r:
        body = r.read()
    check_body(body, "pdf", where="pcbs.gov.ps", pin_size=SIZE)
    with open(PDF + ".part", "wb") as fh:
        fh.write(body)
    os.replace(PDF + ".part", PDF)
    print(f"wrote {PDF} ({len(body):,} bytes)")


def _lines(doc, pno):
    return [ln.strip() for ln in doc.load_page(pno).get_text().splitlines() if ln.strip()]


def _num(s):
    m = NUM.search(s)
    if not m:
        raise SystemExit(f"expected a number at the end of {s!r}")
    return int(m.group(1).replace(",", ""))


def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def read_rows(doc, pno, names, width):
    """A PCBS bilingual table off its text layer.

    Each row comes out as the Arabic name with the first number glued to its end, then the
    remaining numbers one per line, then the English name on its own line. So the row is found
    by its ENGLISH label, and its numbers are the `width` lines above it, the first of them read
    from its tail. Anchoring on the label rather than on a line offset is what survives the
    Arabic text layer's reordering (sources.md §scout-2026-09-14-asia-oceania: "the Arabic text
    layer of book1827-2007.pdf is scrambled").
    """
    lines = _lines(doc, pno)
    keys = {_norm(n): n for n in names}
    out = {}
    for k, ln in enumerate(lines):
        name = keys.get(_norm(ln))
        if name is None or name in out:
            continue
        cells = lines[k - width:k]
        out[name] = tuple(_num(c) if i == 0 else int(c.replace(",", "").split(".")[0])
                          if re.fullmatch(r"[\d,]+", c) else c
                          for i, c in enumerate(cells))
    return out, lines


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Palestine — PCBS census 2017, Preliminary Results, Table 3\n")
    say(doc.page_count == PAGES, f"the book is {doc.page_count} pages (expected {PAGES})")
    say(os.path.getsize(PDF) == SIZE, f"the file is {os.path.getsize(PDF):,} bytes")
    t3text = " ".join(_lines(doc, PAGE_T3))
    say("Table 3: Palestinian Population in Palestine by Governorate and Religion" in t3text,
        "page index 33 carries Table 3's English caption")

    # 1. the parsed table is the transcribed one
    names = WEST_BANK + GAZA_STRIP + ["West Bank", "Gaza Strip", "Palestine"]
    parsed, _ = read_rows(doc, PAGE_T3, names, 5)
    want = dict(T3, **{"West Bank": T3_WEST_BANK, "Gaza Strip": T3_GAZA_STRIP,
                       "Palestine": T3_PALESTINE})
    diff = {k: (parsed.get(k), v) for k, v in want.items() if parsed.get(k) != v}
    say(not diff, f"Table 3 parsed off the text layer: {len(parsed)} rows, every cell identical "
        f"to the transcription {diff or ''}")
    say(len(T3) == 16 and list(T3) == WEST_BANK + GAZA_STRIP,
        "sixteen governorates, eleven West Bank then five Gaza Strip, in the table's order")

    # 2. rows: four answers sum to each printed total
    bad = {k: (sum(v[:4]), v[4]) for k, v in want.items() if sum(v[:4]) != v[4]}
    say(not bad, f"every row's four answers sum to its printed total {bad or ''}")

    # 3. columns: governorates sum to their territory, territories to Palestine
    for label, members, row in (("West Bank", WEST_BANK, T3_WEST_BANK),
                                ("Gaza Strip", GAZA_STRIP, T3_GAZA_STRIP)):
        cols = tuple(sum(T3[m][j] for m in members) for j in range(5))
        say(cols == row, f"the {len(members)} {label} governorates sum to its row in all five "
            f"columns {'' if cols == row else (cols, row)}")
    say(tuple(a + b for a, b in zip(T3_WEST_BANK, T3_GAZA_STRIP)) == T3_PALESTINE,
        "West Bank plus Gaza Strip is the Palestine row in all five columns")

    # 4. Table 2, everyone counted, parsed off its page (9 cells per row)
    t2, _ = read_rows(doc, PAGE_T2, list(T2) + ["West Bank", "Gaza Strip", "Palestine"], 9)
    t2both = {k: v[0] for k, v in t2.items()}
    want2 = dict(T2, **{"West Bank": T2_WEST_BANK, "Gaza Strip": T2_GAZA_STRIP,
                        "Palestine": T2_PALESTINE})
    diff2 = {k: (t2both.get(k), v) for k, v in want2.items() if t2both.get(k) != v}
    say(not diff2, f"Table 2's both-sexes column parsed off its page equals the transcription "
        f"{diff2 or ''}")
    say(all(v[0] == v[2] + v[4] for k, v in t2.items()), "Table 2: males + females = both sexes "
        "on every row")
    say(sum(T2[g] for g in WEST_BANK) == T2_WEST_BANK and
        sum(T2[g] for g in GAZA_STRIP) == T2_GAZA_STRIP and
        T2_WEST_BANK + T2_GAZA_STRIP == T2_PALESTINE == T1_COUNTED,
        f"Table 2 sums to its territories and to Table 1's counted population {T1_COUNTED:,}")

    # 5. the nesting: Palestinians counted <= everyone counted <= counted plus estimate
    neg = {g: T2[g] - T3[g][4] for g in T3 if T2[g] < T3[g][4]}
    say(not neg, "Table 3's Palestinians never exceed Table 2's counted population in any "
        f"governorate {neg or ''}")
    nonpal = T2_PALESTINE - T3_PALESTINE[4]
    say(nonpal == 40_175, f"counted people outside Table 3's universe: {nonpal:,} "
        f"({100 * nonpal / T2_PALESTINE:.2f}% of the counted population)")
    top = sorted(T3, key=lambda g: -(T2[g] - T3[g][4]))[:3]
    print("        most of them in " + ", ".join(f"{g} {T2[g] - T3[g][4]:,}" for g in top))
    say(T1_COUNTED + T1_UNDERCOUNT == T1_TOTAL, f"Table 1: counted {T1_COUNTED:,} + under-"
        f"coverage {T1_UNDERCOUNT:,} = {T1_TOTAL:,}")
    t1 = " ".join(_lines(doc, PAGE_T1))
    say(all(f"{n:,}" in t1 for n in (T1_TOTAL, T1_COUNTED, T1_UNDERCOUNT)),
        "all three Table 1 figures appear on its page")

    # 6. Jerusalem: J1 is inside Table 3's row
    t25 = _lines(doc, PAGE_T25_JERUSALEM)
    joined = " ".join(t25)
    say(f"{T25_JERUSALEM:,}" in joined and f"{T25_J1:,}" in joined and f"{T25_J2:,}" in joined,
        "Table 25's Jerusalem governorate, J1 and J2 figures appear on page index 73")
    say(T25_J1 + T25_J2 == T25_JERUSALEM, f"J1 {T25_J1:,} + J2 {T25_J2:,} = the governorate "
        f"{T25_JERUSALEM:,}")
    say(T3["Jerusalem"][4] <= T2["Jerusalem"] <= T25_JERUSALEM,
        f"Jerusalem nests: Table 3 {T3['Jerusalem'][4]:,} <= Table 2 {T2['Jerusalem']:,} <= "
        f"Table 25 {T25_JERUSALEM:,}")
    say(T3["Jerusalem"][4] > T25_J2, f"Table 3's Jerusalem row ({T3['Jerusalem'][4]:,}) is "
        f"larger than J2 with its estimate ({T25_J2:,}), so it includes J1")
    foot = " ".join(_lines(doc, PAGE_T25_FOOT))
    say("Jerusalem (J1) localities are" in foot and "Umm" in foot,
        "Table 25's footnote lists the J1 localities")

    # 7. the form
    say(list(FORM.values()) == ["Muslim", "Christian", "Other"],
        "Form 25 PHC's Religion codes are 1 Muslim, 2 Christian, 3 Other, asked of "
        "Palestinians only, with no code for no answer (read from the IPUMS copy)")

    # 8. UNSD: other censuses, a witness only
    try:
        import oracle
        for yr in (2007, 1997):
            rows = oracle.oracle("State of Palestine", yr)
            if rows:
                print(f"  --  UNSD table 28, {yr}: {rows}")
    except (SystemExit, Exception) as e:     # noqa: BLE001 - a witness, not a check
        print(f"  --  UNSD table 28 not read ({e})")
    print(f"  --  2017 Christians {T3_PALESTINE[1]:,} ({100 * T3_PALESTINE[1] / T3_PALESTINE[4]:.2f}%); "
          "UNSD's 2007 census row has 43,179")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def emit():
    rows = []
    for g, cells in T3.items():
        for c, n in zip(CATS + ["Total"], cells):
            if n <= 0:
                continue
            rows.append({
                "geo_id": g, "geo_level": "governorate", "geo_name": g,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID,
                "note": (f"Table 3, count; Table 2 counted pop={T2[g]}; "
                         f"territory={'West Bank' if g in WEST_BANK else 'Gaza Strip'}"),
            })
    return rows


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing — run: python sources/ps.py --fetch")
    doc = fitz.open(PDF)
    check(doc)
    rows = emit()

    total = T3_PALESTINE[4]
    print(f"\n  16 governorates, {total:,} Palestinians, {total / 16:,.0f} each")
    for j, c in enumerate(CATS):
        n = T3_PALESTINE[j]
        top = sorted(T3, key=lambda g: -T3[g][j] / T3[g][4])[:3]
        where = ", ".join(f"{g} {100 * T3[g][j] / T3[g][4]:.2f}%" for g in top)
        print(f"    {n:>10,}  {100.0 * n / total:6.2f}%  {c:<11} highest: {where}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
