"""Uzbekistan, Population and Agriculture Census 2026: native language by region -> data/normalized/uz.csv.

    python sources/uz_census.py [--fetch]

THE TABLE. *Preliminary Results of the Population and Agriculture Census of the Republic of
Uzbekistan, 2026* (National Statistics Committee, Tashkent 2026), "Distribution of the population
by main language of communication (mother tongue), by region", printed p.71 of the English edition
(PDF p.73), p.51 of the Uzbek edition (PDF p.72, "Hududlar kesimida aholining asosiy muloqot tili
(ona tili) bo'yicha taqsimlanishi"). One answer per person: questionnaire item 13, "Ваш родной
язык?", which the enumerators' instruction defines as the language learned in childhood and mainly
used in daily life (religiondots data/raw/uz/uz_census2026_instruction_ru.txt, line 1374). For the
nation and its 14 regions it prints the population and eight columns:

    Uzbek, Karakalpak, Kazakh, Tajik, Kyrgyz, Russian, Turkmen, other

The English edition heads the columns "Uzbeks", "Karakalpaks"... (copied from the ethnicity
table); the Uzbek edition prints the language names (o'zbek, qoraqalpoq, qozoq, tojik, qirg'iz,
rus, turkman, boshqa), and the title is a language title in both. THE GRAIN IS THE REGION: the
preliminary volume prints nothing below it, and the final results are still to come.

THE POPULATION is the census's de jure one at 15 January 2026, 39,047,321, which includes the
2,090,953 counted at their usual residence while temporarily away (table on printed p.8).

CHECKS, all asserted:
  1. every row's eight columns sum to its population; the 14 regions sum to the national row in
     every column; the national population is 39,047,321;
  2. the English and Uzbek editions print the same 15 x 9 numbers (a reprint, so it catches a
     misread, not an office error);
  3. A SECOND TABLE OF THE SAME CENSUS: each region's population equals the ethnic composition
     table's (English PDF p.64). Language against nationality per region is then printed, not
     asserted: they are different questions.

RAW FILES. The English edition is religiondots' copy (`religiondots/data/raw/uz/`), read in
place, read-only. The Uzbek edition is aholi.stat.uz's, in languagedots/data/raw/uz/. `--fetch`
downloads whichever is missing; stat.uz's certificate chain is incomplete, so the fetch does not
verify it (as religiondots' uz.py, which uses curl -k).
"""
import os
import re
import ssl
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD  # noqa: E402

RAW = ROOT / "data" / "raw" / "uz"
RD_RAW = RD / "data" / "raw" / "uz"
EN_NAME = "uz_census2026_results_en.pdf"
EN_URL = "https://stat.uz/img/news/english_natija_merged-2_p42445.pdf"
UZ_NAME = "uz_census2026_results_uz.pdf"
UZ_URL = "https://aholi.stat.uz/images/kitob-uzb_p83506.pdf"
OUT = ROOT / "data" / "normalized" / "uz.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

NATIONAL = 39_047_321
LANGS = ["Uzbek", "Karakalpak", "Kazakh", "Tajik", "Kyrgyz", "Russian", "Turkmen", "other"]
UZ_HEAD = ["oʻzbek", "qoraqalpoq", "qozoq", "tojik", "qirgʻiz", "rus", "turkman", "boshqa"]
ETHNIC = ["Uzbeks", "Karakalpaks", "Kazakhs", "Tajiks", "Kyrgyz", "Russians", "Turkmens", "Other"]
# row label (English, Uzbek) -> geo_id, the COD-AB p-code religiondots' uz_lookup.csv keys on.
# The census rows run in the office's region order (SOATO 1735, 1703 ... 1726).
REGIONS = [
    ("Rep. of Karakalpakstan", "Qoraqalpogʻiston Resp.", "UZ35", "Karakalpakstan"),
    ("Andijan Region", "Andijon viloyati", "UZ03", "Andijan"),
    ("Bukhara Region", "Buxoro viloyati", "UZ06", "Bukhara"),
    ("Jizzakh Region", "Jizzax viloyati", "UZ08", "Jizzakh"),
    ("Kashkadarya Region", "Qashqadaryo viloyati", "UZ10", "Kashkadarya"),
    ("Navoi Region", "Navoiy viloyati", "UZ12", "Navoi"),
    ("Namangan Region", "Namangan viloyati", "UZ14", "Namangan"),
    ("Samarkand Region", "Samarqand viloyati", "UZ18", "Samarkand"),
    ("Surkhandarya Region", "Surxondaryo viloyati", "UZ22", "Surkhandarya"),
    ("Syrdarya Region", "Sirdaryo viloyati", "UZ24", "Syrdarya"),
    ("Tashkent Region", "Toshkent viloyati", "UZ27", "Tashkent region"),
    ("Fergana Region", "Fargʻona viloyati", "UZ30", "Fergana"),
    ("Khorezm Region", "Xorazm viloyati", "UZ33", "Khorezm"),
    ("Tashkent city", "Toshkent shahri", "UZ26", "Tashkent city"),
]
NATION = ("Republic of Uzbekistan", "Oʻzbekiston Respublikasi")


def path(name):
    for d in (RAW, RD_RAW):
        if (d / name).exists():
            return d / name
    return None


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for name, url in ((EN_NAME, EN_URL), (UZ_NAME, UZ_URL)):
        if path(name):
            print(f"  have {path(name)}")
            continue
        data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600,
                                      context=ctx).read()
        if not data.rstrip().endswith(b"%%EOF"):
            raise SystemExit(f"{url}: no %%EOF trailer, truncated at source?")
        dst = RAW / name
        dst.with_suffix(".part").write_bytes(data)
        os.replace(dst.with_suffix(".part"), dst)
        print(f"  got  {dst} ({len(data):,} bytes)")


def _count(s):
    t = re.sub(r"\s", "", s)
    if not t.isdigit():
        raise SystemExit(f"expected a count, found {s!r}")
    return int(t)


def _lines(pdf, page):
    import fitz
    doc = fitz.open(pdf)
    return [ln.strip() for ln in doc[page].get_text().splitlines() if ln.strip()]


def read(pdf, page, title, head, labels, ncol):
    """{label: [population, col1..]} for the nation and the 14 regions of one printed table."""
    ls = _lines(pdf, page)
    if title not in ls:
        raise SystemExit(f"{pdf.name} PDF p.{page + 1} is not {title!r}")
    if head and ls[ls.index(head[0]):ls.index(head[0]) + len(head)] != head:
        raise SystemExit(f"{pdf.name} p.{page + 1}: column heads are not {head}")
    out = {}
    for i, ln in enumerate(ls):
        if ln in labels:
            if ln in out:
                raise SystemExit(f"{pdf.name}: {ln} read twice")
            out[ln] = [_count(x) for x in ls[i + 1:i + 1 + ncol]]
    missing = [x for x in labels if x not in out]
    if missing:
        raise SystemExit(f"{pdf.name} p.{page + 1}: rows not found {missing}")
    return out


def main():
    if "--fetch" in sys.argv or not (path(EN_NAME) and path(UZ_NAME)):
        fetch()
    en_labels = [NATION[0]] + [r[0] for r in REGIONS]
    uz_labels = [NATION[1]] + [r[1] for r in REGIONS]
    en = read(path(EN_NAME), 72, "Distribution of the population by main language of "
              "communication (mother tongue), by region", None, en_labels, 9)
    uz = read(path(UZ_NAME), 71, "Hududlar kesimida aholining asosiy muloqot tili (ona tili) "
              "boʻyicha taqsimlanishi", UZ_HEAD, uz_labels, 9)

    # 1. rows sum, regions sum to the nation
    for lab, v in en.items():
        if sum(v[1:]) != v[0]:
            raise SystemExit(f"{lab}: languages sum to {sum(v[1:]):,}, population {v[0]:,}")
    for j, col in enumerate(["population"] + LANGS):
        s = sum(en[r[0]][j] for r in REGIONS)
        if s != en[NATION[0]][j]:
            raise SystemExit(f"{col}: regions sum to {s:,}, the nation {en[NATION[0]][j]:,}")
    if en[NATION[0]][0] != NATIONAL:
        raise SystemExit(f"national population {en[NATION[0]][0]:,}, expected {NATIONAL:,}")
    print(f"language table: 15 rows x 9 columns; every row sums, the 14 regions sum to the "
          f"nation in every column, {NATIONAL:,}")

    # 2. the two editions agree
    for (e, u) in [NATION] + [(r[0], r[1]) for r in REGIONS]:
        if en[e] != uz[u]:
            raise SystemExit(f"editions disagree on {e}: {en[e]} against {uz[u]}")
    print("  the English and Uzbek editions print the same 135 numbers")

    # 3. the ethnic composition table: same population per region; language against nationality
    eth = read(path(EN_NAME), 63, "Ethnic composition of the population, by region", ETHNIC,
               en_labels, 9)
    for lab in en_labels:
        if eth[lab][0] != en[lab][0]:
            raise SystemExit(f"{lab}: population {en[lab][0]:,} in the language table, "
                             f"{eth[lab][0]:,} in the ethnic composition table")
        if sum(eth[lab][1:]) != eth[lab][0]:
            raise SystemExit(f"{lab}: ethnic groups do not sum")
    print("  every region's population equals the ethnic composition table's (p.64)")
    print("\n  language minus nationality, thousands (same order: " + ", ".join(LANGS) + ")")
    for lab in en_labels:
        d = [(a - b) / 1000 for a, b in zip(en[lab][1:], eth[lab][1:])]
        print(f"    {lab:24s}" + " ".join(f"{x:8.1f}" for x in d))

    rows = []
    for e, _u, gid, name in REGIONS:
        for lang, n in zip(LANGS, en[e][1:]):
            rows.append(dict(geo_id=gid, unit_name=name, source_category=lang, count=n,
                             tier="measured"))
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, OUT)
    nat = en[NATION[0]]
    print(f"\nwrote {OUT} ({len(df)} rows, {df['count'].sum():,} people)")
    for lang, n in zip(LANGS, nat[1:]):
        print(f"  {lang:11s}{n:12,} {100 * n / nat[0]:6.2f}%")


if __name__ == "__main__":
    main()
