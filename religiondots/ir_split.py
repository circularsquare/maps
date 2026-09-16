"""Iran: split the 1395 census `Muslim` column into Sunni by Masaili's province estimates.

Writes data/normalized/ir_split.csv: every province's census `مسلمان` (Muslim) count again, as a
Sunni row where Masaili gives one plus the census's own `مسلمان` for the rest, summing to the census
figure to the person. countries/ir.py::_ir_counts swaps these in for ir.csv's Muslim rows.

WHY THIS EXISTS
---------------
Iran draws 79.6M Muslims from one census column, and the census publishes no sect. Anita,
2026-09-16 (ask/RULINGS.md), after the §14 flag on Sistan and Baluchestan and Kurdistan and the
vetting in sources/branches.md ("Masaili, vetted"): *"lets draw masaili. he seems good."* The brief
was queue.md, "Iran's Sunni/Shia split from Masaili"; sources/ir.md §9 has the record.

THE SOURCE
----------
Mehdi Masaili, *Atlas-e Towsifi-ye Ahl-e Sonnat-e Iran* (Tehran: Amirkabir, 1402 / 2023), p. 59,
reprinted as Table 1 of Hosseinpour and Masaili, *Haft Aseman* 26(88), printed pp. 77-78:
`haftasman.urd.ac.ir/article_218087_735cc0a11dd4bd43e466ede88c6286bc.pdf` (open, 106 pages,
1,383,553 bytes), kept as data/raw/ir/haftaseman_26-88_hosseinpour_masaili.pdf. The book itself was
not seen. Each row is the province's 1395 census population, the author's Sunni percentage of it
("library and first-hand field research", one person, no source list) and their product, rounded to
the thousand (Kermanshah to the 500):

    province                 population   Sunni   Sunni people
    Sistan and Baluchestan    2,775,014    64%     1,776,000
    Kurdistan                 1,603,011    82%     1,312,000
    West Azerbaijan           3,265,219    35%     1,142,000
    Golestan                  1,868,819    38%       710,000
    Hormozgan                 1,776,415    35%       621,000
    Kermanshah                1,952,434    26%       507,500
    Razavi Khorasan           6,434,501     5%       322,000
    Fars                      4,851,274     4%       194,000
    Gilan                     2,530,696     7%       177,000
    South Khorasan              768,898    15%       115,000
    North Khorasan              863,092    10%        86,000
    Kerman                    3,164,718     2%        63,000
    Bushehr                   1,163,400     5%        58,000
    Ardabil                   1,270,420     2%        25,000
    Tehran and the central provinces                 500,000
    Iran                                           7,608,500

`check_source()` re-reads every cell off the PDF's text layer, and `_check_table()` asserts each
population against Table 3-18's province total and each name against the census's Persian name, so
neither a transcription slip nor a row on the wrong province can pass quietly.

THE CALLS, argued in sources/ir.md §9 and taxonomy/ir2016.py REVIEW
------------------------------------------------------------------
* THE PRINTED COUNT IS DRAWN, not the percentage times the census's Muslims. The count is the
  figure the author published and it sums to his total; the percentage is an integer and carries
  less. The Sunni row is taken out of the census `مسلمان` column, never added beside it.
* THE REST STAYS ON `islam`, as the census's own `مسلمان`, `measured`, just smaller (spec §2.7a,
  in_split.py's REMAINDER). Masaili names Sunnis and prints no Shia figure. The remainder is mostly
  Twelver Shia, but a Shia layer made of it would be a number nobody published, would carry the
  low lean he states (a Sunni author puts the Kurds about two million higher, so the excess lands
  on "Shia" in exactly the Kurdish provinces), and nothing could contradict it.
* THE 500,000 LUMP names no province. It goes to Tehran and Alborz in proportion to their census
  Muslims, by largest remainder: Tehran is named, Alborz was Tehran province until 2010 and Karaj is
  part of the same city region, and Soltani (2015) counts Alborz with Tehran. The other central
  provinces (Qom, Markazi, Isfahan, Qazvin, Semnan, Yazd) get none of it; nothing read names a
  Sunni community in any of them. See LUMP_PROVINCES.
* PROVINCES WITH NO ROW (15, Khuzestan and Isfahan among them) keep every Muslim on `islam`.

TIER: `derived`, AND THE ROLL-UP IS THE REASON
----------------------------------------------
The census counted these Muslims in each province; only which branch comes from Masaili. So the
Sunni rows are `derived` with `parent_column=مسلمان`, and taxonomy/ir2016.py COLUMNS rolls them back
to `islam` when a reader turns inferred dots off (spec §7a-i-1). `modelled` would make the viewer
delete 7.6M counted Muslims instead. Basis is `estimate` on the Sunni rows (spec §3.1: a compiler's
judgement may split a category, never add to one).

Usage:
    python ir_split.py             build data/normalized/ir_split.csv
    python ir_split.py --dry-run   report the numbers, write nothing
"""

import os
import re
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
NORM = os.path.join(HERE, "data", "normalized")
SRC_CENSUS = os.path.join(NORM, "ir.csv")
PDF = os.path.join(HERE, "data", "raw", "ir", "haftaseman_26-88_hosseinpour_masaili.pdf")
PDF_BYTES = 1_383_553
PDF_PAGES = (4, 5)                     # page indices; printed pp. 77 and 78
OUT = os.path.join(NORM, "ir_split.csv")

COL = "مسلمان"
SOURCE_ID = "ir_masaili_2023_atlas_p59"
LABEL = "Muslim: Sunni (Masaili 2023)"
LABEL_LUMP = "Muslim: Sunni, Tehran and the central provinces (Masaili 2023)"

# (ir.csv geo_id, name as printed, 1395 population as printed, Sunni %, Sunni people as printed)
TABLE = [
    ("Sistan and Baluchestan", "سیستان و بلوچستان", 2_775_014, 64, 1_776_000),
    ("Kurdistan",              "کردستان",           1_603_011, 82, 1_312_000),
    ("West Azerbaijan",        "آذربایجان غربی",    3_265_219, 35, 1_142_000),
    ("Golestan",               "گلستان",            1_868_819, 38,   710_000),
    ("Hormozgan",              "هرمزگان",           1_776_415, 35,   621_000),
    ("Kermanshah",             "کرمانشاه",          1_952_434, 26,   507_500),
    ("Razavi Khorasan",        "خراسان رضوی",       6_434_501,  5,   322_000),
    ("Fars",                   "فارس",              4_851_274,  4,   194_000),
    ("Gilan",                  "گیلان",             2_530_696,  7,   177_000),
    ("South Khorasan",         "خراسان جنوبی",        768_898, 15,   115_000),
    ("North Khorasan",         "خراسان شمالی",        863_092, 10,    86_000),
    ("Kerman",                 "کرمان",             3_164_718,  2,    63_000),
    ("Bushehr",                "بوشهر",             1_163_400,  5,    58_000),
    ("Ardabil",                "اردبیل",            1_270_420,  2,    25_000),
]
LUMP = 500_000                          # row 15, اهل‌سنت تهران و استان‌های مرکزی ایران
TOTAL = 7_608_500                       # the table's foot, جمعیت اهل‌سنت در کل کشور
# Which provinces take the lump, and why (module docstring, sources/ir.md §9). Persian names are
# asserted against ir.csv like TABLE's.
LUMP_PROVINCES = {"Tehran": "تهران", "Alborz": "البرز"}

NOTE_SUNNI = ("level=leaf; derivation=compiler_estimate; structure={sid}; "
              "structure_geo=province; share_of_population={pct}%; printed={printed}; "
              "parent_column=" + COL)
NOTE_LUMP = ("level=leaf; derivation=compiler_estimate; structure={sid}; "
             "structure_geo=lump:Tehran and the central provinces; lump=500000 spread over "
             "Tehran and Alborz by census Muslims; parent_column=" + COL)
NOTE_REST = ("level=leaf; cat=" + COL + "; derivation=exact_single_child; branch not named "
             "(Masaili names Sunnis only; the rest is mostly Twelver Shia and no source counts it)")
NOTE_UNSPLIT = ("level=leaf; cat=" + COL + "; derivation=exact_single_child; no Sunni row in "
                "Masaili's table for this province")
OUT_COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis",
               "year", "source_id", "tier", "note"]

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٬", "01234567890123456789,")


def _fold(name):
    """Persian name key: Arabic yeh and kaf folded, no spaces or ZWNJ, and the لا ligature that
    the PDF's text layer writes as ال (`گیالن` for `گیلان`) folded the same way on both sides."""
    s = str(name).replace("ي", "ی").replace("ك", "ک").replace("‌", "")
    s = "".join(s.split())
    return s.replace("لا", "ال")


def _num(s):
    return int(s.replace(",", ""))


def check_source():
    """Re-read Table 1 off the PDF's text layer and compare every cell with TABLE, LUMP, TOTAL."""
    if not os.path.exists(PDF):
        raise SystemExit(f"missing {PDF}; the URL is in this file's docstring")
    size = os.path.getsize(PDF)
    if size != PDF_BYTES:
        raise SystemExit(f"{PDF} is {size:,} bytes, expected {PDF_BYTES:,}")
    import fitz
    doc = fitz.open(PDF)
    if doc.page_count != 106:
        raise SystemExit(f"PDF has {doc.page_count} pages, expected 106 (truncated?)")
    text = "\n".join(doc[i].get_text() for i in PDF_PAGES).translate(_DIGITS)
    text = text.replace("‌", "")
    if "جدول شماره" not in text or "(مسائلی" not in text:
        raise SystemExit("Table 1's caption not found on PDF page indices 4-5")
    rows = re.findall(r"(\d{1,2})([^\d%]+?)(\d{1,3}(?:,\d{3})+)\s*%(\d{1,2})\s*"
                      r"(\d{1,3}(?:,\d{3})+)", text)
    got = [(int(r), _fold(n), _num(p), int(s), _num(c)) for r, n, p, s, c in rows]
    want = [(i + 1, _fold(name), pop, pct, n) for i, (_, name, pop, pct, n) in enumerate(TABLE)]
    if got != want:
        raise SystemExit("Table 1 on the PDF does not match TABLE:\n"
                         + "\n".join(f"  pdf  {g}\n  here {w}" for g, w in zip(got, want)
                                     if g != w)
                         + f"\n  ({len(got)} rows parsed, {len(want)} expected)")
    m = re.search(r"15\s*اهلسنت\s*تهران\s*و\s*استانهای\s*مرکزی\s*ایران\s*(\d{1,3}(?:,\d{3})+)",
                  text)
    if not m or _num(m.group(1)) != LUMP:
        raise SystemExit(f"row 15 (Tehran and the central provinces) reads "
                         f"{m.group(1) if m else 'nothing'}, expected {LUMP:,}")
    t = re.search(r"(\d{1,3}(?:,\d{3})+)\s*:\s*جمعیت\s*اهلسنت\s*در\s*کل\s*کشور", text)
    if not t or _num(t.group(1)) != TOTAL:
        raise SystemExit(f"the table's national total reads {t.group(1) if t else 'nothing'}, "
                         f"expected {TOTAL:,}")
    print(f"  OK Table 1 re-read off the PDF: 14 province rows, the {LUMP:,} lump and the "
          f"{TOTAL:,} total all match")


def _check_table(census):
    """Every row against the census: names, populations, arithmetic, and the Muslim ceiling."""
    totals = census.groupby("geo_id")["count"].sum()
    names = census.drop_duplicates("geo_id").set_index("geo_id")["geo_name"]
    muslims = census[census["source_category"] == COL].set_index("geo_id")["count"]
    if len(totals) != 31 or len(muslims) != 31:
        raise SystemExit(f"ir.csv has {len(totals)} provinces and {len(muslims)} Muslim rows, "
                         "expected 31 each -- re-run sources/ir.py")
    for geo, name, pop, pct, n in TABLE:
        if geo not in totals.index:
            raise SystemExit(f"{geo!r} is not a geo_id in ir.csv")
        if _fold(names[geo]) != _fold(name):
            raise SystemExit(f"{geo}: ir.csv's Persian name is {names[geo]!r}, Table 1's is "
                             f"{name!r} (a row joined to the wrong province?)")
        if int(totals[geo]) != pop:
            raise SystemExit(f"{geo}: Table 1 prints population {pop:,}, Table 3-18's province "
                             f"total is {int(totals[geo]):,}")
        # An integer percentage carries +-0.5% of the population, and the count is rounded to
        # the thousand or the 500.
        if abs(pop * pct / 100 - n) > pop * 0.005 + 500:
            raise SystemExit(f"{geo}: {pct}% of {pop:,} is {pop * pct / 100:,.0f}, the table "
                             f"prints {n:,}")
        if n > int(muslims[geo]):
            raise SystemExit(f"{geo}: {n:,} Sunnis exceed the census's {int(muslims[geo]):,} "
                             "Muslims")
    for geo, name in LUMP_PROVINCES.items():
        if _fold(names[geo]) != _fold(name):
            raise SystemExit(f"{geo}: ir.csv's Persian name is {names[geo]!r}, expected {name!r}")
        if geo in {t[0] for t in TABLE}:
            raise SystemExit(f"{geo} has its own row and is also in LUMP_PROVINCES")
    if sum(t[4] for t in TABLE) + LUMP != TOTAL:
        raise SystemExit(f"the 14 rows and the lump sum to {sum(t[4] for t in TABLE) + LUMP:,}, "
                         f"the table prints {TOTAL:,}")
    print("  OK 14 names and populations equal Table 3-18's; every count is its percentage of "
          "the population; rows + lump = total; no Sunni figure exceeds its census Muslims")
    return muslims


def _largest_remainder(weights, total):
    """Integer split of `total` in proportion to `weights`, summing to `total` exactly."""
    s = sum(weights)
    exact = [total * w / s for w in weights]
    base = [int(x) for x in exact]
    order = sorted(range(len(exact)), key=lambda i: -(exact[i] - base[i]))
    for i in order[:total - sum(base)]:
        base[i] += 1
    return base


def build():
    check_source()
    census = pd.read_csv(SRC_CENSUS, low_memory=False, keep_default_na=False, na_values=[""])
    census["count"] = census["count"].astype("int64")
    muslims = _check_table(census)

    sunni = {geo: (n, NOTE_SUNNI.format(sid=SOURCE_ID, pct=pct, printed=n), LABEL)
             for geo, _, _, pct, n in TABLE}
    lump_geos = list(LUMP_PROVINCES)
    for geo, n in zip(lump_geos, _largest_remainder([int(muslims[g]) for g in lump_geos], LUMP)):
        sunni[geo] = (n, NOTE_LUMP.format(sid=SOURCE_ID), LABEL_LUMP)

    mus = census[census["source_category"] == COL]
    rows = []
    for r in mus.itertuples(index=False):
        base = dict(geo_id=r.geo_id, geo_level=r.geo_level, geo_name=r.geo_name, year=r.year)
        m = int(r.count)
        if r.geo_id not in sunni:
            rows.append(dict(base, source_category=COL, count=m, basis=r.basis,
                             source_id=r.source_id, tier="measured", note=NOTE_UNSPLIT))
            continue
        n, note, label = sunni[r.geo_id]
        rows.append(dict(base, source_category=label, count=n, basis="estimate",
                         source_id=SOURCE_ID, tier="derived", note=note))
        rows.append(dict(base, source_category=COL, count=m - n, basis=r.basis,
                         source_id=r.source_id, tier="measured", note=NOTE_REST))
    out = pd.DataFrame(rows, columns=OUT_COLUMNS)
    _check(out, muslims)
    return out


def _check(out, muslims):
    per = out.groupby("geo_id")["count"].sum()
    if set(per.index) != set(muslims.index) or (per.reindex(muslims.index) - muslims).abs().max():
        raise SystemExit("ir_split does not conserve each province's census Muslims")
    if (out["count"] < 0).any():
        raise SystemExit("a remainder went negative")
    sun = out[out["tier"] == "derived"]
    if int(sun["count"].sum()) != TOTAL:
        raise SystemExit(f"{int(sun['count'].sum()):,} Sunnis drawn, the table prints {TOTAL:,}")
    total = int(muslims.sum())
    print(f"  OK every province's rows sum to its census Muslims, {total:,} in all; "
          f"{TOTAL:,} Sunni drawn")

    print("\n  province                    Muslims      Sunni  of Muslims   on islam")
    s = sun.groupby("geo_id")["count"].sum()
    for geo in s.sort_values(ascending=False).index:
        mm = int(muslims[geo])
        print(f"  {geo:<24}{mm:>11,}{int(s[geo]):>11,}{100 * s[geo] / mm:>11.1f}%"
              f"{mm - int(s[geo]):>11,}")
    rest = int(muslims.drop(s.index).sum())
    print(f"  {'15 provinces, no row':<24}{rest:>11,}{0:>11,}{0:>11.1f}%{rest:>11,}")
    print(f"  {'Iran':<24}{total:>11,}{TOTAL:>11,}{100 * TOTAL / total:>11.2f}%"
          f"{total - TOTAL:>11,}")


def main():
    out = build()
    if "--dry-run" in sys.argv:
        print("  --dry-run: nothing written")
        return
    tmp = OUT + ".part"
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"  wrote {OUT} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
