"""Seychelles: NBS, Population and Housing Census 2010, Table 3 of the district supplement.

Reads (or fetches) data/raw/sc/ and writes data/normalized/sc.csv.

**THE 2010 CENSUS PRINTS RELIGION FOR EVERY DISTRICT, IN THE WRITE-IN CATEGORIES.** The
supplement *Population and Housing Census 2010: Supplement Statistical Tables (ALL
DISTRICTS)* repeats thirteen tables once for each of the 26 districts in alphabetical order,
and Table 3 of each is *Population by religion and sex*. NBS post-coded the religion question
in 2010 ("due to a rise in demand for more disaggregated data", report §2.7): the form had
boxes for Roman Catholic, Anglican, Adventist, Muslim, Baha'i, Hindu and no religion and a
line to write anything else, and the supplement keeps every write-in, **57 labels across the
districts**. UNSD table 28 holds only the 2002 return (11 categories) and nothing from 2010.

**THE PARSE CLOSES THREE WAYS.** Every row's female and male add to its total; every
district's rows add to its Total row AND to the total of Table 1 (population by age) on the
page before, which is an independent table; and the 26 districts add, label by label, to the
main report's national Table 2.9. Table 2.9 prints 31 rows where the districts print 57,
because the national table folds 24 small write-ins into four of its rows; `T29_FOLD` spells
out the folds and `check` asserts each one to the person.

**THE REPORT'S DISTRICT TABLE HAS ONE WRONG ROW, AND THE SUPPLEMENT CLOSES IT.** Table 2.3
(population by district) prints `Other Islands` as 576 and a total of 90,945, but its 26 rows
sum to 90,479. The supplement prints Other Islands as 1,042 on both its age and its religion
table, which is 466 more, exactly the shortfall; every other district agrees. `check` asserts
exactly that.

**WHY 2010 AND NOT 2022.** The 2022 report prints religion by district too (Table B4.1), but
folded to six answers plus `Unable to classify` and `Missing`, where `Other` lumps the Baha'is,
the Buddhists and no religion together. **11.5% of its 102,612 are `Missing`**, because
institutional households got a short form without the question, and NBS says in the same
report that the census missed a significant part of the population (its mid-2022 estimate is
119,878). 2010 has 4.8% `Not stated`. So 2010 is drawn and B4.1 is read as a witness on the
district pattern; `check_2022` prints it.

The table is read off the PDF text layer, which gives one cell per line: a label (wrapped
onto two lines when long), then Female, Male and Total. Rows vary by district because a
district lists only what someone in it answered.

Usage:
    python sources/sc.py --fetch    three PDFs, ~33 MB (IHSN catalogue 4079, and NBS for 2022)
    python sources/sc.py            normalise from data/raw/sc/
"""

import csv
import math
import os
import random
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

RAW = os.path.join(ROOT, "data", "raw", "sc")
OUT = os.path.join(ROOT, "data", "normalized", "sc.csv")

SOURCE_ID = "sc_phc_2010"
YEAR = 2010
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126 Safari/537.36"}

# local name -> (url, exact size in bytes). The IHSN catalogue copies are NBS's own PDFs;
# nbs.gov.sc's downloads tree lists only the 2022 census volumes (checked 2026-09-14).
FILES = {
    "phc2010_supplement.pdf": ("https://catalog.ihsn.org/catalog/4079/download/55082", 17_533_419),
    "phc2010_report.pdf": ("https://catalog.ihsn.org/catalog/4079/download/55081", 5_093_475),
    "phc2022_report.pdf": ("https://www.nbs.gov.sc/downloads/"
                           "1555-seychelles-population-and-housing-census-2022/download",
                           10_525_457),
}

NATIONAL = 90_945

# The supplement's district order (its preface), each with the ISO 3166-2:SC code that
# sources/sc_geo.py joins on. `Other Islands` has no ISO code.
DISTRICTS = {
    "Anse Aux Pins": "SC-01", "Anse Boileau": "SC-02", "Au Cap": "SC-04",
    "Anse Etoile": "SC-03", "Anse Royale": "SC-05", "Bel Air": "SC-09",
    "Baie Lazare": "SC-06", "Belombre": "SC-10", "Baie Sainte Anne": "SC-07",
    "Beau Vallon": "SC-08", "Cascade": "SC-11", "English River": "SC-16",
    "Glacis": "SC-12", "Grand Anse Mahe": "SC-13", "Grand Anse Praslin": "SC-14",
    "La Digue": "SC-15", "Les Mamelles": "SC-24", "Mont Buxton": "SC-17",
    "Mont Fleuri": "SC-18", "Plaisance": "SC-19", "Port Glaud": "SC-21",
    "Pointe Larue": "SC-20", "Roche Caiman": "SC-25", "Saint Louis": "SC-22",
    "Takamaka": "SC-23", "Other Islands": "SC-OI",
}

# Report Table 2.9, `Population by religious affiliation, 2010`, page 37 of the PDF.
TABLE_2_9 = {
    "Roman Catholic": 69_277, "Anglican": 5_585, "Hindu": 2_174, "Islam": 1_459,
    "Pentecostal Assembly": 1_333, "7th Day Adventist": 1_128, "No Religion": 840,
    "Assembly of God": 831, "Jehovah's Witness": 683, "Born Again Christian": 605,
    "Baha'i": 522, "Redeemed Christian Church": 394, "Buddhist": 331,
    "Christians Unspecified Denomination": 322, "Christian Community Fellowship": 241,
    "Other Christian": 112, "Nazarite Christian": 109, "Other non-Christian": 108,
    "Orthodox": 97, "Neo Apostolic": 69, "Taoism": 62, "Grace and Peace": 52,
    "Deeper Life": 50, "Church of Christ": 42, "Christ Holiness Church": 40,
    "Christian Church of England": 29, "Baptist": 28, "New Testament Church": 27,
    "Methodist": 24, "Protestant Christian": 23, "Latin Catholic": 20, "Not stated": 4_328,
}

# How Table 2.9 folds the supplement's labels. A Table 2.9 row not listed here is the
# supplement label of the same name. These were found by arithmetic, not assumed: each fold
# is the only set of the small labels that closes its row, and `check` asserts all of them.
T29_FOLD = {
    "Jehovah's Witness": ["Jehovah Witness"],
    "Christian Community Fellowship": ["Christian Community Fellowship",
                                       "Christian Life Fellowship"],
    "Other Christian": ["Other Christian", "End-Time-Bride Tabernacle", "United Pentecost",
                        "Peniel Tabernacle", "Lutheran", "End Time Message",
                        "Seychelles Believers International", "New Born Christian",
                        "Dutch Reform Church", "United Christian Church", "Grace Assembly",
                        "Full Gospel Assembly", "Presbyterian"],
    "Other non-Christian": ["Other", "Rastafarian", "Zoroastrian", "Tamil", "Jain",
                            "Pantheist", "Jew", "Meditate", "Sai Baba devotee", "Atheist",
                            "Bobo Chanty", "Padayachi", "Agnostic"],
}

# Report Table 2.3, `Population and household distribution and density by district, 2010`,
# page 28 of the PDF: population and area in km2 as printed.
TABLE_2_3 = {
    "Anse Aux Pins": (3_850, 2.5), "Anse Boileau": (4_011, 12.0), "Anse Etoile": (4_717, 6.0),
    "Anse Royale": (4_168, 7.2), "Au Cap": (4_233, 8.3), "Baie Lazare": (3_608, 12.1),
    "Baie Sainte Anne": (4_876, 28.5), "Beau Vallon": (4_120, 4.5), "Bel Air": (2_857, 4.5),
    "Belombre": (3_708, 9.4), "Cascade": (4_267, 10.6), "English River": (4_196, 2.3),
    "Glacis": (3_833, 6.9), "Grand Anse Mahe": (3_106, 15.7),
    "Grand Anse Praslin": (3_727, 22.3), "La Digue": (2_761, 36.4),
    "Les Mamelles": (2_667, 1.7), "Mont Buxton": (3_089, 1.2), "Mont Fleuri": (3_419, 5.6),
    "Other Islands": (576, 179.6), "Plaisance": (3_781, 3.4), "Pointe Larue": (3_071, 3.5),
    "Port Glaud": (2_572, 26.6), "Roche Caiman": (3_232, 1.7), "Saint Louis": (3_209, 1.4),
    "Takamaka": (2_825, 14.2),
}
# Table 2.3 prints `La Digue (& Inner Islands)`; the supplement prints `La Digue` for the same
# 2,761 people. The row it gets wrong, and by how much (see the docstring).
TABLE_2_3_MISPRINT = ("Other Islands", 466)

NUM = re.compile(r"^\d+$")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, (url, size) in FILES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"  have {name} ({size:,} bytes)")
            continue
        print("GET", url)
        r = requests.get(url, headers=UA, timeout=900)
        r.raise_for_status()
        body = r.content
        # [[reference_pdf_truncated_at_source]]: pin the exact size and the trailer.
        if not body.startswith(b"%PDF") or b"%%EOF" not in body[-4096:]:
            raise SystemExit(f"{name}: {len(body):,} bytes, not a whole PDF")
        if len(body) != size:
            raise SystemExit(f"{name}: {len(body):,} bytes, pinned at {size:,}; the file has "
                             "been reissued and the tables need re-reading")
        tmp = dest + ".part"                              # [[reference_wb_truncates]]
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, dest)
        print(f"  {len(body):,} bytes")


def _open(name, pages):
    import fitz

    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing; run `python sources/sc.py --fetch`")
    doc = fitz.open(path)
    if doc.page_count != pages:
        raise SystemExit(f"{name} has {doc.page_count} pages, expected {pages}")
    return doc


def _lines(page):
    return [l.strip() for l in page.get_text().split("\n") if l.strip()]


def _rows(lines, start):
    """Label, then three integers, repeated until `Total`. -> ({label: (f, m, t)}, total row)."""
    rows, j = {}, start
    while j < len(lines):
        # A long label wraps onto a second line (`Christians Unspecified` / `Denomination`),
        # so every non-numeric line up to the first figure is one label.
        parts = []
        while j < len(lines) and not NUM.match(lines[j]):
            parts.append(lines[j])
            j += 1
        if not parts:
            raise SystemExit(f"a figure where a label should be, at line {j}: {lines[j:j + 4]}")
        if len(parts) > 2:
            raise SystemExit(f"a label spanning {len(parts)} lines: {parts}")
        label = " ".join(parts)
        vals = lines[j:j + 3]
        if len(vals) < 3 or not all(NUM.match(v) for v in vals):
            raise SystemExit(f"row {label!r} is not followed by three figures: {vals}")
        f, m, t = (int(v) for v in vals)
        if f + m != t:
            raise SystemExit(f"row {label!r}: female {f} + male {m} != total {t}")
        if label == "Total":
            return rows, (f, m, t)
        if label in rows:
            raise SystemExit(f"row {label!r} appears twice")
        rows[label] = (f, m, t)
        j += 3
    raise SystemExit("ran off the page without a Total row")


def read_supplement():
    """-> {district: {label: total}}, every in-table identity and the age total asserted."""
    doc = _open("phc2010_supplement.pdf", 313)
    out, age_total = {}, {}
    for i in range(doc.page_count):
        lines = _lines(doc[i])
        if not lines or lines[0] not in DISTRICTS:
            continue
        d = lines[0]
        # Table 2 (nationality) is printed ABOVE Table 1 on some pages, so the first `Total`
        # on the page can be Table 2's header. Anchor on Table 1's last age group instead.
        if any(l.startswith("Table 1:") for l in lines):
            k = lines.index("65+")
            if lines[k + 4] != "Total" or not all(NUM.match(v) for v in lines[k + 5:k + 8]):
                raise SystemExit(f"{d}: Table 1's Total row is not where expected on page "
                                 f"{i + 1}: {lines[k:k + 8]}")
            f, m, t = (int(v) for v in lines[k + 5:k + 8])
            if f + m != t:
                raise SystemExit(f"{d}: Table 1 Total {f} + {m} != {t}")
            age_total[d] = t
        # Table 3's caption is misprinted as `marital status` on Beau Vallon's page, so the
        # anchor is the header cell `Religion` and a Roman Catholic row, not the caption.
        if "Religion" in lines and "Roman Catholic" in lines:
            k = lines.index("Religion")
            start = lines.index("Male", k if lines[k + 1] == "Sex" else 0) + 1
            if d in out:
                raise SystemExit(f"{d}: a second religion table on page {i + 1}")
            rows, (f, m, t) = _rows(lines, start)
            s = sum(v[2] for v in rows.values())
            if s != t:
                raise SystemExit(f"{d}: rows sum to {s:,}, Total row says {t:,}")
            # One label carries a stray leading apostrophe, `'7th Day Adventist`, from a
            # spreadsheet cell forced to text. The label is otherwise kept verbatim.
            out[d] = {lab.lstrip("'"): v[2] for lab, v in rows.items()}
    doc.close()

    if sorted(out) != sorted(DISTRICTS):
        raise SystemExit(f"religion tables found for {len(out)} districts; missing "
                         f"{sorted(set(DISTRICTS) - set(out))}")
    for d, rows in out.items():
        if sum(rows.values()) != age_total.get(d):
            raise SystemExit(f"{d}: religion table totals {sum(rows.values()):,}, the age "
                             f"table {age_total.get(d)}")
    print(f"  26 district religion tables read; each equals its own age table's total")
    return out


def read_table_2_9():
    """The national table, off the report page, against the transcription in TABLE_2_9."""
    doc = _open("phc2010_report.pdf", 227)
    lines = _lines(doc[36])
    doc.close()
    if "Table 2.9: Population by religious affiliation, 2010" not in lines:
        raise SystemExit("report page 37 is not Table 2.9")
    j = lines.index("%") + 1
    got = {}
    while lines[j] != "Total":
        parts = []
        while not re.match(r"^[\d,]+$", lines[j]):
            parts.append(lines[j])
            j += 1
        label = " ".join(parts).replace("’", "'")
        got[label] = int(lines[j].replace(",", ""))
        j += 2                                               # the count, then the percentage
    if int(lines[j + 1].replace(",", "")) != NATIONAL:
        raise SystemExit(f"Table 2.9 total reads {lines[j + 1]}")
    if got != TABLE_2_9:
        diff = {k: (got.get(k), TABLE_2_9.get(k)) for k in set(got) | set(TABLE_2_9)
                if got.get(k) != TABLE_2_9.get(k)}
        raise SystemExit(f"Table 2.9 off the page disagrees with TABLE_2_9: {diff}")
    if sum(got.values()) != NATIONAL:
        raise SystemExit(f"Table 2.9 rows sum to {sum(got.values()):,}")
    return got


def check(sup):
    t29 = read_table_2_9()
    national = {}
    for rows in sup.values():
        for lab, n in rows.items():
            national[lab] = national.get(lab, 0) + n
    if sum(national.values()) != NATIONAL:
        raise SystemExit(f"the districts sum to {sum(national.values()):,}, not {NATIONAL:,}")

    folded = {lab for labs in T29_FOLD.values() for lab in labs}
    seen = set()
    for row, want in t29.items():
        labs = T29_FOLD.get(row, [row])
        missing = [l for l in labs if l not in national]
        if missing:
            raise SystemExit(f"Table 2.9 {row!r} folds {missing}, which no district prints")
        got = sum(national[l] for l in labs)
        if got != want:
            raise SystemExit(f"Table 2.9 {row!r} is {want:,}; its districts sum to {got:,}")
        seen.update(labs)
    stray = sorted(set(national) - seen)
    if stray:
        raise SystemExit(f"district labels in no Table 2.9 row: {stray}")
    print(f"  the districts sum to Table 2.9 in all {len(t29)} rows, with "
          f"{len(folded) - len(T29_FOLD) + 1} small labels folded into four of them")

    bad, gap = [], 0
    for d, rows in sup.items():
        s, (printed, _) = sum(rows.values()), TABLE_2_3[d]
        if s != printed:
            bad.append((d, s - printed))
    if bad != [TABLE_2_3_MISPRINT]:
        raise SystemExit(f"supplement against Table 2.3: {bad}, expected only "
                         f"{TABLE_2_3_MISPRINT}")
    short = NATIONAL - sum(p for p, _ in TABLE_2_3.values())
    if short != TABLE_2_3_MISPRINT[1]:
        raise SystemExit(f"Table 2.3's rows are {short:,} short of its total, not "
                         f"{TABLE_2_3_MISPRINT[1]}")
    print(f"  Table 2.3 agrees on 25 districts; its Other Islands row is {short} short, "
          "which is exactly its rows' shortfall against its own total")
    return national


def read_b41():
    """2022 report Table B4.1 -> [(label, [total, cath, angl, islam, hindu, xoth, uncl, oth, miss])]."""
    doc = _open("phc2022_report.pdf", 301)
    lines = _lines(doc[117])
    doc.close()
    if not any(l.startswith("Table B4.1: Population in all households by religion") for l in lines):
        raise SystemExit("2022 report page 118 is not Table B4.1")
    j = lines.index("Missing") + 1
    cell = re.compile(r"^[\d,]+$|^-$")
    rows, label = [], "Seychelles"
    while j < len(lines):
        if cell.match(lines[j]):
            vals = [0 if v == "-" else int(v.replace(",", "")) for v in lines[j:j + 9]]
            if len(vals) != 9 or sum(vals[1:]) != vals[0]:
                raise SystemExit(f"B4.1 row {label!r} does not close: {vals}")
            rows.append((label, vals))
            j += 9
            label = None
        else:
            label = lines[j] if label is None else label + " " + lines[j]
            j += 1
    return rows


def _spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


def check_2022(sup):
    """The same districts twelve years on, on the four answers both censuses print alike.

    A witness on the pattern, not on the level: the 2022 table folds everything small, and
    the two sets of district boundaries differ in three places (Ile Perseverance became a
    district of its own, and `La Digue & Inner Islands` and `Outer Islands` cut the islands
    differently from 2010's `La Digue` and `Other Islands`). Those are left out.
    """
    b41 = dict((lab, v) for lab, v in read_b41())
    b41 = {("Grand Anse Mahe" if lab == "Grand Anse Mahé" else
            "Baie Sainte Anne" if lab == "Baie Ste Anne" else lab): v for lab, v in b41.items()}
    same = [d for d in DISTRICTS if d not in ("English River", "La Digue", "Other Islands")]
    missing = [d for d in same if d not in b41]
    if missing:
        raise SystemExit(f"B4.1 has no row for {missing}")
    rng = random.Random(0)
    print(f"  2022 witness, {len(same)} districts whose boundaries did not move, share of "
          "those who answered:")
    pairs = {"Catholic": (["Roman Catholic", "Latin Catholic"], 1),
             "Anglican": (["Anglican"], 2), "Islam": (["Islam"], 3), "Hindu": (["Hindu"], 4)}
    for name, (labs, col) in pairs.items():
        a, b = [], []
        for d in same:
            s10 = sup[d]
            ans10 = sum(s10.values()) - s10.get("Not stated", 0)
            a.append(sum(s10.get(l, 0) for l in labs) / ans10)
            v = b41[d]
            b.append(v[col] / (v[0] - v[8] - v[6]))
        rho = _spearman(a, b)
        null = []
        for _ in range(2000):
            sh = b[:]
            rng.shuffle(sh)
            null.append(_spearman(a, sh))
        p = sum(1 for x in null if x >= rho) / len(null)
        print(f"    {name:<9} rho {rho:+.2f}  (one-sided permutation p {p:.3f})  "
              f"2010 {min(a):.1%}-{max(a):.1%}, 2022 {min(b):.1%}-{max(b):.1%}")


def normalise():
    sup = read_supplement()
    national = check(sup)
    check_2022(sup)

    rows = []
    for d, labels in sup.items():
        for lab, n in labels.items():
            if n == 0:
                continue
            rows.append({"geo_id": DISTRICTS[d], "geo_level": "district", "geo_name": d,
                         "source_category": lab, "count": n, "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID,
                         "note": "PHC 2010 Supplement Statistical Tables, Table 3"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)
    ns = national["Not stated"]
    print(f"\nwrote {OUT}")
    print(f"  {len(rows)} rows, {len(sup)} districts, {len(national)} labels, "
          f"{sum(r['count'] for r in rows):,} people; not stated {ns:,} ({ns / NATIONAL:.2%})")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    normalise()
