"""Rwanda — NISR, RPHC-5 2022, the thirty DISTRICT PROFILES, Table 2.2 in each.

Reads (or fetches) data/raw/rw/ and writes data/normalized/rw.csv.

**THE QUEUE ROW PRICED THIS COUNTRY AT FIVE PROVINCES AND IT IS THIRTY DISTRICTS.**
sources.md §11p read the RPHC-5 *Social-cultural characteristics* thematic report, which is
where religion lives in the thematic series and which works at province. That report is not
the finest thing NISR published. In May 2025 the office released a **district profile per
district**, thirty separate 80-140 page PDFs on `statistics.gov.rw/district-statistics/<province>`,
and **every one carries the religion question as its Table 2.2** — the same eleven
categories, counts and percentages, split Total / Urban / Rural. 13.2M over 30 units is
441k each, against 2.6M each at province. This is Botswana's shape exactly (§9bu): the
national report stops at the nation and the per-district booklet series, published outside
the report set the sweep tested, carries the fine geography.

**ELEVEN CATEGORIES, AND ONE OF THEM IS A NAMED CHURCH.** `ADEPR` is the Association des
Églises de Pentecôte du Rwanda, and at **2,820,813 people, 21.3%, it is the second-largest
religious answer in the country** after Catholicism. The UNSD Demographic Yearbook prints
the identical figure under the generic label `Pentecostal`, which is how NISR described it
to the UN; the district profiles name the church. Nothing else on this map has a single
denomination counted as a census category at a fifth of a country.

**THE PARSE IS A PLAIN LINE READ.** The text layer emits the row label and then its three
counts and three percentages, one per line, in column order. It is still checked rather
than trusted: the eleven labels are asserted in order in each of the thirty files, and the
printed percentages are reproduced from the counts on every cell, which is the check that
catches a column landing in the wrong place.

**THE RECONCILIATION HAS AN OUTSIDE WITNESS.** Every district's Table 2.2 reprints the
national row, and the thirty districts sum to it: 13,246,394 / 3,701,245 / 9,545,149. That
is internal. The outside check is `tools/oracle.py Rwanda` — the UNSD Demographic Yearbook
table 28, which is NISR's own return to the UN and an entirely separate publication from
these booklets. Its eleven 2022 figures are held in `UNSD` below and must reproduce to the
person from the thirty district tables.

**THE GAP IS `Not stated`, 17,785 people, 0.13%** — the smallest §3.5 residual of any
census on this map that has one at all.

Two access notes. The thirty RPHC-5 profiles live under `/sites/default/files/2025-05/` and
are named for the district with no suffix, except that **Nyabihu is at
`/2025-10/Nyabihu_RPHC2022.pdf`** — a later folder and a different filename, the one file a
pattern-guessing fetch misses. Three district names are shouted in the filename (`BURERA`,
`BUGESERA`, `KIREHE`) and the rest are title case. The `/2025-06/` files on the same pages
are the RPHC-4 (2012) profiles and must not be read for this.

Usage:
    python sources/rw.py --fetch    thirty PDFs, ~120 MB, a couple of minutes
    python sources/rw.py            normalise from data/raw/rw/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "rw")
OUT = os.path.join(ROOT, "data", "normalized", "rw.csv")

SOURCE_ID = "rw_rphc5_2022"
YEAR = 2022
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASE = "https://statistics.gov.rw"

# district -> (province as the booklets print it, path on statistics.gov.rw).
# Read off the five /district-statistics/<province> pages rather than guessed; see the
# module docstring on Nyabihu, which is the one file no pattern reaches.
PROFILES = {
    "Nyarugenge":  ("City of Kigali",    "/sites/default/files/2025-05/Nyarugenge.pdf"),
    "Gasabo":      ("City of Kigali",    "/sites/default/files/2025-05/Gasabo.pdf"),
    "Kicukiro":    ("City of Kigali",    "/sites/default/files/2025-05/Kicukiro.pdf"),
    "Nyanza":      ("Southern Province", "/sites/default/files/2025-05/Nyanza.pdf"),
    "Gisagara":    ("Southern Province", "/sites/default/files/2025-05/Gisagara.pdf"),
    "Nyaruguru":   ("Southern Province", "/sites/default/files/2025-05/Nyaruguru.pdf"),
    "Huye":        ("Southern Province", "/sites/default/files/2025-05/Huye.pdf"),
    "Nyamagabe":   ("Southern Province", "/sites/default/files/2025-05/Nyamagabe.pdf"),
    "Ruhango":     ("Southern Province", "/sites/default/files/2025-05/Ruhango.pdf"),
    "Muhanga":     ("Southern Province", "/sites/default/files/2025-05/Muhanga.pdf"),
    "Kamonyi":     ("Southern Province", "/sites/default/files/2025-05/Kamonyi.pdf"),
    "Karongi":     ("Western Province",  "/sites/default/files/2025-05/Karongi.pdf"),
    "Rutsiro":     ("Western Province",  "/sites/default/files/2025-05/Rutsiro.pdf"),
    "Rubavu":      ("Western Province",  "/sites/default/files/2025-05/Rubavu.pdf"),
    "Nyabihu":     ("Western Province",  "/sites/default/files/2025-10/Nyabihu_RPHC2022.pdf"),
    "Ngororero":   ("Western Province",  "/sites/default/files/2025-05/Ngororero.pdf"),
    "Rusizi":      ("Western Province",  "/sites/default/files/2025-05/Rusizi.pdf"),
    "Nyamasheke":  ("Western Province",  "/sites/default/files/2025-05/Nyamasheke.pdf"),
    "Rulindo":     ("Northern Province", "/sites/default/files/2025-05/Rulindo.pdf"),
    "Gakenke":     ("Northern Province", "/sites/default/files/2025-05/Gakenke.pdf"),
    "Musanze":     ("Northern Province", "/sites/default/files/2025-05/Musanze.pdf"),
    "Burera":      ("Northern Province", "/sites/default/files/2025-05/BURERA.pdf"),
    "Gicumbi":     ("Northern Province", "/sites/default/files/2025-05/Gicumbi.pdf"),
    "Rwamagana":   ("Eastern Province",  "/sites/default/files/2025-05/Rwamagana.pdf"),
    "Nyagatare":   ("Eastern Province",  "/sites/default/files/2025-05/Nyagatare.pdf"),
    "Gatsibo":     ("Eastern Province",  "/sites/default/files/2025-05/Gatsibo.pdf"),
    "Kayonza":     ("Eastern Province",  "/sites/default/files/2025-05/Kayonza.pdf"),
    "Kirehe":      ("Eastern Province",  "/sites/default/files/2025-05/KIREHE.pdf"),
    "Ngoma":       ("Eastern Province",  "/sites/default/files/2025-05/Ngoma.pdf"),
    "Bugesera":    ("Eastern Province",  "/sites/default/files/2025-05/BUGESERA.pdf"),
}
EXPECTED_DISTRICTS = 30

# In the order every one of the thirty tables prints them, top to bottom.
CATEGORIES = [
    "Catholic",
    "ADEPR",
    "Protestant",
    "Adventist",
    "Other Christians",
    "Muslim",
    "Jehovah witness",
    "Traditional/Animist",
    "Other religion",
    "No Religion",
    "Not stated",
]
TOTAL_CAT = "Total"

# The national row, reprinted at the head of all thirty tables.
NATIONAL = (13_246_394, 3_701_245, 9_545_149)

# UNSD Demographic Yearbook table 28, Rwanda 2022, Total — `python tools/oracle.py Rwanda`.
# NISR's own return to the UN and a wholly separate publication from the district profiles,
# so agreeing with it to the person is evidence about the READ and not only about the
# arithmetic. The UN prints `Pentecostal` where the booklets print the church's name.
UNSD = {
    "Catholic": 5_286_003,
    "ADEPR": 2_820_813,               # UNSD: `Pentecostal`
    "Protestant": 1_928_741,
    "Adventist": 1_612_482,
    "Other Christians": 553_174,
    "Muslim": 265_317,
    "Jehovah witness": 93_131,        # UNSD: `Jehovah Witness`
    "Traditional/Animist": 2_112,     # UNSD: `Animist`
    "Other religion": 264_319,        # UNSD: `Other Religions`
    "No Religion": 402_517,
    "Not stated": 17_785,             # UNSD: `Not Specified`
}

NUMISH = re.compile(r"^\d[\d,]*(?:\.\d+)?$")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
    for district, (_, path) in PROFILES.items():
        dest = os.path.join(RAW, district + ".pdf")
        if os.path.exists(dest) and os.path.getsize(dest) > 500_000:
            print("already have", dest)
            continue
        print("GET", BASE + path)
        r = requests.get(BASE + path, timeout=600, headers=ua)
        r.raise_for_status()
        # §5a: HTTP 200 is not a download. Assert size AND type.
        if r.content[:5] != b"%PDF-":
            raise SystemExit(f"{district}: not a PDF -- starts {r.content[:16]!r}, "
                             f"{len(r.content):,} bytes")
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(dest):,} bytes")


def _fold(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _num(tok):
    return float(tok.replace(",", "")) if "." in tok else int(tok.replace(",", ""))


_TITLE = re.compile(r"Religious\s+Affiliation", re.I)

# **THE PARSE IS GEOMETRIC AND IT HAS TO BE.** A plain line read works on 29 of the 30
# booklets and is wrong on Gakenke, where `Traditional/Animist` has 51 people, all of them
# rural, and NISR prints its URBAN cell BLANK rather than as a zero. In the text layer that
# row is `51 / 51 / 0.01 / 0.01` and a reader taking the first three tokens as the three
# counts hands the district 0.01 traditionalists and shifts the percentages one column
# left. Nothing downstream would notice: the row is 51 people. So columns are cut on
# position, using the `Catholic` row -- which is full in every district -- as the template.
# [[reference_pdf_table_geometry]].
_ROW_TOL = 3.0


def _rows(page):
    """The page's words grouped into visual rows: [(label, [(x_centre, value), ...]), ...]."""
    ws = sorted(page.get_text("words"), key=lambda w: (round(w[1], 1), w[0]))
    out, cur, cy = [], [], None
    for x0, y0, x1, _y1, txt, *_rest in ws:
        if cy is not None and abs(y0 - cy) > _ROW_TOL:
            out.append(sorted(cur))
            cur = []
        cur.append((round((x0 + x1) / 2.0, 2), txt))
        cy = y0 if cy is None or abs(y0 - cy) > _ROW_TOL else cy
    if cur:
        out.append(sorted(cur))

    rows = []
    for line in out:
        i = 0
        while i < len(line) and not NUMISH.match(line[i][1]):
            i += 1
        label = " ".join(t for _, t in line[:i])
        vals = [(x, _num(t)) for x, t in line[i:] if NUMISH.match(t)]
        rows.append((label, vals))
    return rows


def _cells(vals, anchors, where):
    """Assign a row's figures to the six columns the Catholic row defines. None = blank."""
    out = [None] * len(anchors)
    for x, v in vals:
        j = min(range(len(anchors)), key=lambda k: abs(anchors[k] - x))
        if out[j] is not None:
            raise SystemExit(f"{where}: two figures land in column {j} ({out[j]}, {v}) -- "
                             "Table 2.2 has been re-typeset and the columns must be recut")
        out[j] = v
    return out


def read_one(district, province):
    import fitz

    p = os.path.join(RAW, district + ".pdf")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    doc = fitz.open(p)
    if doc.page_count == 0:
        raise SystemExit(f"{p}: PyMuPDF reports zero pages -- truncated at source "
                         "([[reference_pdf_truncated_at_source]])")

    # The title is not a constant: most booklets say "... by Religious Affiliation and
    # residence areas" and Rwamagana says "... by Area of Residence and Religious
    # Affiliation". And the PROSE above the table names the categories too, so a text test
    # alone can match the wrong page. The page is identified by having a `Catholic` row
    # with all six figures on it, which only the table has.
    hits = []
    for n in range(doc.page_count):
        t = doc[n].get_text()
        if "ADEPR" not in t or not _TITLE.search(" ".join(t.split())):
            continue
        rows = _rows(doc[n])
        if any(_fold(lab) == "catholic" and len(v) == 6 for lab, v in rows):
            hits.append((n, rows))
    if len(hits) != 1:
        raise SystemExit(f"{district}: {len(hits)} pages carry Table 2.2, expected exactly "
                         f"one (pages {[h[0] + 1 for h in hits]})")
    page_no = hits[0][0]
    # Table 2.2 breaks across a page in several booklets, always with its header repeated
    # at the top of the continuation. The next page is appended unconditionally; a booklet
    # whose table is complete simply never reads that far.
    rows = list(hits[0][1])
    if page_no + 1 < doc.page_count:
        rows += _rows(doc[page_no + 1])
    doc.close()

    where = f"{district} p{page_no + 1}"

    # ANCHOR ON `Catholic`, NOT ON THE FIRST `Rwanda`. Table 2.1 (nationality) shares the
    # page in several booklets and one of ITS column headers is the word `Rwanda`, so the
    # obvious anchor lands in the wrong table and reads `Foreigners` as a figure.
    cat = [i for i, (lab, v) in enumerate(rows) if _fold(lab) == "catholic" and len(v) == 6]
    if len(cat) != 1:
        raise SystemExit(f"{where}: {len(cat)} full `Catholic` rows, expected one")
    ci = cat[0]
    anchors = [x for x, _ in rows[ci][1]]

    # The eleven categories must follow in order. Anything between them that is not one of
    # the eleven -- the folio, the repeated header -- is skipped, and a category out of
    # place stops the run rather than relabelling the map.
    want = {_fold(c): c for c in CATEGORIES}
    got, seen = {}, []
    for lab, vals in rows[ci:]:
        key = _fold(lab)
        if key not in want:
            continue
        if want[key] in got:
            break
        seen.append(want[key])
        got[want[key]] = _cells(vals, anchors, f"{where} {want[key]!r}")
        if len(seen) == len(CATEGORIES):
            break
    if seen != CATEGORIES:
        raise SystemExit(f"{where}: read categories {seen}, expected {CATEGORIES} -- NISR "
                         "has changed Table 2.2's row list and taxonomy/rw2022.py must be "
                         "revisited")

    # The rows above `Catholic`: the district's own total, and -- in most but not all
    # booklets -- the province and national rows reprinted above it.
    head = {}
    for name in (f"{district} district", province, "Rwanda"):
        idx = next((i for i in range(ci - 1, max(-1, ci - 5), -1)
                    if _fold(rows[i][0]) == _fold(name)), None)
        head[name] = None if idx is None else _cells(rows[idx][1], anchors, f"{where} {name}")
    if head[f"{district} district"] is None:
        raise SystemExit(f"{where}: no `{district} district` row above Table 2.2's "
                         "Catholic row")
    return page_no + 1, head, got


def read():
    per = {}
    for district, (province, _) in PROFILES.items():
        page, head, got = read_one(district, province)

        nat = head["Rwanda"]
        if nat is not None and tuple(int(v) for v in nat[:3]) != NATIONAL:
            raise SystemExit(f"{district}: the reprinted national row is {nat[:3]}, "
                             f"expected {list(NATIONAL)} -- not RPHC-5 2022")

        total = head[f"{district} district"][:3]
        if any(v is None for v in total):
            raise SystemExit(f"{district}: the district total row has a blank cell {total}")
        prov = head[province]

        cats = {}
        for c in CATEGORIES:
            cells = got[c]
            if cells[0] is None:
                raise SystemExit(f"{district}/{c}: the Total column is blank")
            # A blank count cell is a printed zero, not a missing figure -- Gakenke's
            # `Traditional/Animist` urban cell. `Urban + Rural == Total` checks it.
            cats[c] = [int(v or 0) for v in cells[:3]]

        per[district] = {
            "page": page,
            "province": province,
            "has_national": nat is not None,
            "province_row": None if prov is None else [
                None if v is None else int(v) for v in prov[:3]],
            "total": [int(v) for v in total],
            "cats": cats,
            "pct": {c: got[c][3:] for c in CATEGORIES},
        }
    return per


def check(per):
    ok = True

    good = len(per) == EXPECTED_DISTRICTS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(per)} district profiles read "
          f"(expected {EXPECTED_DISTRICTS})")

    bad = [d for d, v in per.items()
           if sum(v["cats"][c][0] for c in CATEGORIES) != v["total"][0]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 11 categories sum to the district total on "
          f"all {len(per)} districts ({len(bad)} failures) {bad[:4]}")

    bad = []
    for d, v in per.items():
        for c in CATEGORIES + ["__total__"]:
            t, u, r = v["total"] if c == "__total__" else v["cats"][c]
            if u + r != t:
                bad.append((d, c, u + r, t))
    ok &= not bad
    n = len(per) * (len(CATEGORIES) + 1)
    print(f"  {'OK ' if not bad else 'BAD'} Urban + Rural == Total on all {n} cells "
          f"({len(bad)} failures)")
    for d, c, s, t in bad[:5]:
        print(f"        {d}/{c}: {s:,} vs {t:,}")

    # The printed percentages are the check on the READ. Every other identity here holds
    # inside one column whichever line a figure came off; this one does not.
    bad, nudged, cells = [], [], 0
    for d, v in per.items():
        for c in CATEGORIES:
            pct = v["pct"][c]
            for k in range(min(3, len(pct))):
                base = v["total"][k]
                if not base or pct[k] is None:
                    continue
                cells += 1
                want = 100.0 * v["cats"][c][k] / base
                dp = len(str(pct[k]).split(".")[1]) if "." in str(pct[k]) else 0
                err = abs(want - pct[k])
                if err > 1.0 * 10 ** -dp + 1e-9:
                    bad.append((d, c, k, pct[k], round(want, 4)))
                elif err > 0.5 * 10 ** -dp + 1e-9:
                    nudged.append((d, c, k, pct[k], round(want, 4)))
    ok &= not bad
    # The bar is ONE unit in the last printed place, not half of one. NISR rounds a
    # handful of cells the wrong way and those are its arithmetic rather than this read;
    # a column landing one place left would be out by whole percentage points, so the
    # loosened bar still catches everything it is here to catch.
    print(f"  {'OK ' if not bad else 'BAD'} the printed percentages reproduce from the "
          f"counts on all {cells:,} cells to within a unit in the last place "
          f"({len(bad)} failures, {len(nudged)} rounded the wrong way by NISR)")
    for d, c, k, p, w in (bad + nudged)[:5]:
        print(f"        {d}/{c} col{k}: printed {p} vs {w}")

    tot = [sum(v["total"][k] for v in per.values()) for k in range(3)]
    good = tuple(tot) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 30 districts sum to the national row "
          f"{tuple(f'{t:,}' for t in tot)}")

    # Each booklet also reprints its own province row, from a different line of the table.
    prov = {}
    for d, v in per.items():
        prov.setdefault(v["province"], []).append(v)
    bad, printed = [], 0
    for p, vs in prov.items():
        rows = [v["province_row"] for v in vs if v["province_row"] is not None]
        printed += len(rows)
        for k in range(3):
            got = {r[k] for r in rows if r[k] is not None}
            if len(got) > 1:
                bad.append((p, k, "the booklets disagree", sorted(got)))
            elif got and sum(v["total"][k] for v in vs) != got.pop():
                bad.append((p, k, sum(v["total"][k] for v in vs),
                            [r[k] for r in rows][0]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} each province's districts sum to the province "
          f"row, on the {printed} booklets that reprint one ({len(bad)} failures)")
    for row in bad[:5]:
        print(f"        {row}")
    nat_rows = sum(1 for v in per.values() if v["has_national"])
    print(f"      {nat_rows} of {len(per)} booklets also reprint the national row and all "
          f"{nat_rows} give\n      13,246,394 / 3,701,245 / 9,545,149. Gakenke, Kirehe "
          "and Musanze print neither\n      and start straight at the district row, which "
          "is why both are optional rather than\n      asserted.")

    # The outside witness.
    bad = []
    for c in CATEGORIES:
        s = sum(v["cats"][c][0] for v in per.values())
        if s != UNSD[c]:
            bad.append((c, s, UNSD[c]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} all 11 national totals reproduce the UNSD "
          f"Demographic Yearbook to the person ({len(bad)} failures)")
    for c, s, w in bad[:5]:
        print(f"        {c}: {s:,} vs UNSD {w:,}")
    print("      the Yearbook is NISR's return to the UN and a separate publication from "
          "these\n      booklets, so this is a check on the read and not on the "
          "arithmetic (§9's rule).")

    print(f"\n  categories, national, from the thirty tables:")
    for c in CATEGORIES:
        s = sum(v["cats"][c][0] for v in per.values())
        print(f"    {s:>11,}  {100.0 * s / NATIONAL[0]:6.2f}%  {c}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def build_rows(per):
    rows = []
    # geo_id is minted alphabetically and is deliberately NOT NISR's district code, so
    # nothing downstream can quietly assume the two are the same string (Benin's lesson).
    for i, district in enumerate(sorted(per), start=1):
        v = per[district]
        gid = f"RW{i:02d}"
        note = f"level=district; province={v['province']}"
        for c in CATEGORIES:
            rows.append({"geo_id": gid, "geo_level": "district", "geo_name": district,
                         "source_category": c, "count": v["cats"][c][0], "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})
        rows.append({"geo_id": gid, "geo_level": "district", "geo_name": district,
                     "source_category": TOTAL_CAT, "count": v["total"][0], "basis": BASIS,
                     "year": YEAR, "source_id": SOURCE_ID,
                     "note": note + "; universe total, not a religion category"})
    return rows


def main():
    if "--fetch" in sys.argv:
        fetch()
    per = read()
    check(per)
    rows = build_rows(per)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
