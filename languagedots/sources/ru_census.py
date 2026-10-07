"""Russia, All-Russian Population Census 2020 (held October-November 2021): native language by
federal subject, urban and rural -> data/normalized/ru.csv.

    python sources/ru_census.py [--fetch]

THE TABLE. Rosstat, *Itogi VPN-2020*, Volume 5 "Natsionalnyi sostav i vladenie yazykami", Table 6
"Naselenie po rodnomu yazyku" (Tom5_tab6_VPN-2020.xlsx, published 31.12.2022). One sheet for the
Russian Federation and one for each of the 85 subjects the census covered, plus "Arkhangelsk
without the AO" and "Tyumen without the AOs". Each sheet: the people who stated a native language
(urban + rural, urban, rural), then one row per language group the census coded (about 190
nationally; a sheet prints only the languages present there), then "other answers" and "native
language not stated". ONE ANSWER PER PERSON: the language rows sum exactly to "stated" (checked).

CHECKS, all asserted:
  1. every sheet: the language rows plus "other answers" sum to "stated"; urban + rural = total;
     Tyumen, the one sheet printing its population, has stated + not stated = population;
  2. the 85 subject sheets sum to the Russian Federation sheet, language by language (6 people
     reshuffled between three tiny rows, totals exact; tolerated up to 10); Arkhangelsk
     without the AO + Nenets = Arkhangelsk, Tyumen without the AOs + Khanty-Mansi + Yamal-Nenets =
     Tyumen, language by language;
  3. A SECOND TABLE OF THE SAME CENSUS: Table 7 (nationality by native language, same volume,
     Tom5_tab7_VPN-2020.xlsx) carries an "all population" column per language per subject; it
     equals Table 6's urban + rural on every (subject, language) both tables print.

WHAT IS WRITTEN. geo_level `subject` rows for the 85 subjects with area urban / rural (and total):
the 83 of religiondots' units keyed by ISO 3166-2 (geoBoundaries ADM1), and the Republic of Crimea
and Sevastopol keyed by their ISO 3166-2 codes UA-43 and UA-40, which the map draws from this
census (Anita, 2026-10-05; sources/ru.md). Arkhangelsk and Tyumen are the "without AO" sheets, as
their AOs are units of their own.
`source_category` is the census's language label as printed, stripped. "Не указавшие родной язык"
is written too (the gap), and taxonomy/ru2021.py resolves it to None.

Rosstat's certificate is issued by the Russian national CA, which the usual bundles do not carry,
so the fetch does not verify it.
"""
import os
import ssl
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "ru"
OUT = ROOT / "data" / "normalized" / "ru.csv"
BASE = "https://rosstat.gov.ru/storage/mediabank/"
FILES = ["Tom5_tab6_VPN-2020.xlsx", "Tom5_tab7_VPN-2020.xlsx", "Tom5_Spisok_yazykov.doc",
         "Tom5_met_VPN-2020.pdf"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

STATED = "Указавшие родной язык"
OTHER = "Указавшие другие ответы"
NOT_STATED = "Не указавшие родной язык"
POP = "Мужчины и женщины"     # the whole population; printed on a few sheets only

# Rosstat sheet name (stripped) -> ISO 3166-2. None: a parent whose parts are units of their own.
SHEETS = {
    "Белгородская область": "RU-BEL", "Брянская область": "RU-BRY",
    "Владимирская область": "RU-VLA", "Воронежская область": "RU-VOR",
    "Ивановская область": "RU-IVA", "Калужская область": "RU-KLU",
    "Костромская область": "RU-KOS", "Курская область": "RU-KRS",
    "Липецкая область": "RU-LIP", "Московская область": "RU-MOS",
    "Орловская область": "RU-ORL", "Рязанская область": "RU-RYA",
    "Смоленская область": "RU-SMO", "Тамбовская область": "RU-TAM",
    "Тверская область": "RU-TVE", "Тульская область": "RU-TUL",
    "Ярославская область": "RU-YAR", "г. Москва": "RU-MOW",
    "Республика Карелия": "RU-KR", "Республика Коми": "RU-KO",
    "Архангельская область": None, "Архангельская область без АО": "RU-ARK",
    "Ненецкий автономный округ": "RU-NEN", "Вологодская область": "RU-VLG",
    "Калининградская область": "RU-KGD", "Ленинградская область": "RU-LEN",
    "Мурманская область": "RU-MUR", "Новгородская область": "RU-NGR",
    "Псковская область": "RU-PSK", "г. Санкт-Петербург": "RU-SPE",
    "Республика Адыгея": "RU-AD", "Республика Калмыкия": "RU-KL",
    "Республика Крым": "UA-43", "Краснодарский край": "RU-KDA",
    "Астраханская область": "RU-AST", "Волгоградская область": "RU-VGG",
    "Ростовская область": "RU-ROS", "г. Севастополь": "UA-40",
    "Республика Дагестан": "RU-DA", "Республика Ингушетия": "RU-IN",
    "Кабардино-Балкарская Республика": "RU-KB", "Карачаево-Черкесская Республика": "RU-KC",
    "РСО-Алания": "RU-SE", "Чеченская Республика": "RU-CE",
    "Ставропольский край": "RU-STA", "Республика Башкортостан": "RU-BA",
    "Республика Марий Эл": "RU-ME", "Республика Мордовия": "RU-MO",
    "Республика Татарстан": "RU-TA", "Удмуртская Республика": "RU-UD",
    "Чувашская Республика": "RU-CU", "Пермский край": "RU-PER",
    "Кировская область": "RU-KIR", "Нижегородская область": "RU-NIZ",
    "Оренбургская область": "RU-ORE", "Пензенская область": "RU-PNZ",
    "Самарская область": "RU-SAM", "Саратовская область": "RU-SAR",
    "Ульяновская область": "RU-ULY", "Курганская область": "RU-KGN",
    "Свердловская область": "RU-SVE", "Тюменская область": None,
    "Тюменская область без АО": "RU-TYU", "Ханты-Мансийский АО - Югра": "RU-KHM",
    "Ямало-Ненецкий АО": "RU-YAN", "Челябинская область": "RU-CHE",
    "Республика Алтай": "RU-AL", "Республика Тыва": "RU-TY",
    "Республика Хакасия": "RU-KK", "Алтайский край": "RU-ALT",
    "Красноярский край": "RU-KYA", "Иркутская область": "RU-IRK",
    "Кемеровская область - Кузбасс": "RU-KEM", "Новосибирская область": "RU-NVS",
    "Омская область": "RU-OMS", "Томская область": "RU-TOM",
    "Республика Бурятия": "RU-BU", "Республика Саха (Якутия)": "RU-SA",
    "Забайкальский край": "RU-ZAB", "Камчатский край": "RU-KAM",
    "Приморский край": "RU-PRI", "Хабаровский край": "RU-KHA",
    "Амурская область": "RU-AMU", "Магаданская область": "RU-MAG",
    "Сахалинская область": "RU-SAK", "Еврейская автономная область": "RU-YEV",
    "Чукотский автономный округ": "RU-CHU",
}
NATIONAL = "Российская Федерация"
WITHOUT_AO = {"Архангельская область без АО", "Тюменская область без АО"}
PARTS = {"Архангельская область": ["Архангельская область без АО", "Ненецкий автономный округ"],
         "Тюменская область": ["Тюменская область без АО", "Ханты-Мансийский АО - Югра",
                               "Ямало-Ненецкий АО"]}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for f in FILES:
        p = RAW / f
        if p.exists() and p.stat().st_size > 10_000:
            continue
        req = urllib.request.Request(BASE + f, headers=UA)
        body = urllib.request.urlopen(req, context=ctx, timeout=300).read()
        if len(body) < 10_000:
            raise SystemExit(f"{f}: only {len(body)} bytes; Rosstat may be refusing")
        p.write_bytes(body)
        print(f"  {f}  {len(body):,} bytes")


def num(v):
    if v is None or v == "-" or v == "–":
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    raise ValueError(f"not a number: {v!r}")


def read_t6():
    """{sheet: {label: (total, urban, rural)}}, labels stripped; STATED and NOT_STATED included."""
    wb = openpyxl.load_workbook(RAW / "Tom5_tab6_VPN-2020.xlsx", read_only=True)
    out = {}
    for name in wb.sheetnames:
        rows = {}
        for r in wb[name].iter_rows(values_only=True):
            if r[0] is None or r[1] is None:
                continue
            lab = " ".join(str(r[0]).split())
            if lab in rows:
                raise SystemExit(f"t6 {name}: label twice: {lab}")
            rows[lab] = (num(r[1]), num(r[2]), num(r[3]))
        out[name.strip()] = rows
    return out


def read_t7():
    """{sheet: {label: all-population}} from Table 7's first numeric column."""
    wb = openpyxl.load_workbook(RAW / "Tom5_tab7_VPN-2020.xlsx", read_only=True)
    out = {}
    for name in wb.sheetnames:
        rows = {}
        for r in wb[name].iter_rows(values_only=True):
            if r[0] is None or r[1] is None or isinstance(r[1], str):
                continue
            rows[" ".join(str(r[0]).split())] = num(r[1])
        out[name.strip()] = rows
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    t6 = read_t6()
    names = set(t6) - {NATIONAL}
    if names != set(SHEETS):
        raise SystemExit(f"sheet names changed: new {sorted(names - set(SHEETS))}, "
                         f"gone {sorted(set(SHEETS) - names)}")

    # 1. each sheet adds up
    n_pop = 0
    for name, rows in t6.items():
        st = rows[STATED]
        langs = {k: v for k, v in rows.items() if k not in (STATED, NOT_STATED, POP)}
        if POP in rows:
            n_pop += 1
            for i in range(3):
                if st[i] + rows.get(NOT_STATED, (0, 0, 0))[i] != rows[POP][i]:
                    raise SystemExit(f"{name}: stated + not stated != population (col {i})")
        for i in range(3):
            s = sum(v[i] for v in langs.values())
            if s != st[i]:
                raise SystemExit(f"{name}: languages sum to {s:,}, stated is {st[i]:,} (col {i})")
        for k, v in rows.items():
            if v[1] + v[2] != v[0]:
                raise SystemExit(f"{name} {k}: urban + rural {v[1] + v[2]:,} != {v[0]:,}")
    print(f"  check 1: {len(t6)} sheets, languages + other answers = stated; urban + rural = total; "
          f"stated + not stated = population on the {n_pop} sheets that print it")

    # 2. subjects sum to the federation; parents = parts
    top = [n for n in SHEETS if n not in WITHOUT_AO and n not in ("Ненецкий автономный округ",
           "Ханты-Мансийский АО - Югра", "Ямало-Ненецкий АО")]
    # 85 subjects, but Arkhangelsk's and Tyumen's sheets already hold their three AOs
    if len(top) != 82:
        raise SystemExit(f"expected 82 top-level sheets, have {len(top)}")

    def summed(sheets):
        acc = {}
        for n in sheets:
            for k, v in t6[n].items():
                if k == POP:
                    continue
                acc[k] = acc.get(k, 0) + v[0]
        return acc
    nat = {k: v[0] for k, v in t6[NATIONAL].items() if k != POP}
    s85 = summed(top)
    diff = {k: (nat.get(k, 0), s85.get(k, 0)) for k in set(nat) | set(s85)
            if nat.get(k, 0) != s85.get(k, 0)}
    # The federation sheet differs from its subjects by 6 people on three rows (Maori 6 in the
    # subjects, folded into "other answers" 4 and "Gabonese" 2 nationally); totals are exact.
    # Small reshufflings like that are tolerated, anything bigger is not.
    if nat[STATED] != s85[STATED] or nat[NOT_STATED] != s85[NOT_STATED] or \
            any(abs(a - b) > 10 for a, b in diff.values()):
        raise SystemExit(f"82 top-level sheets != federation: {list(diff.items())[:8]}")
    shuffled = ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(diff.items()))
    for parent, parts in PARTS.items():
        a, b = {k: v[0] for k, v in t6[parent].items() if k != POP}, summed(parts)
        bad = {k: (a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b) if a.get(k, 0) != b.get(k, 0)}
        if a[STATED] != b[STATED] or any(abs(x - y) > 10 for x, y in bad.values()):
            raise SystemExit(f"{parent} != its parts on {sorted(bad)[:8]}")
        if bad:
            print(f"    {parent} vs its parts: {bad} (same reshuffle)")
    print(f"  check 2: the 85 subjects = Russian Federation on {len(nat) - len(diff)} of "
          f"{len(nat)} rows, totals exact (federation/subjects: {shuffled}); "
          "Arkhangelsk and Tyumen = their parts")

    # 3. Table 7 agrees
    t7 = read_t7()
    n_cmp = 0
    for name, rows in t6.items():
        r7 = t7.get(name)
        if r7 is None:
            # Table 7's sheet names differ in spacing, and two are abbreviated
            alias = {"Ханты-Мансийский АО - Югра": "ХМАО", "Ямало-Ненецкий АО": "ЯНАО"}
            cand = [k for k in t7 if k.replace(" ", "") == alias.get(name, name).replace(" ", "")]
            if len(cand) != 1:
                raise SystemExit(f"Table 7 has no sheet for {name}")
            r7 = t7[cand[0]]
        for k, v in rows.items():
            if k in (NOT_STATED, POP):
                continue
            if k in r7:
                if r7[k] != v[0]:
                    raise SystemExit(f"Table 7 disagrees: {name} {k}: {r7[k]:,} vs {v[0]:,}")
                n_cmp += 1
    if n_cmp < 3000:
        raise SystemExit(f"only {n_cmp} cells compared against Table 7; its layout changed?")
    print(f"  check 3: Table 7 equals Table 6 on all {n_cmp:,} (subject, language) cells both print")

    out = []
    for name, iso in SHEETS.items():
        if iso is None:
            continue
        level = "subject"
        for k, v in t6[name].items():
            if k in (STATED, POP):
                continue
            for area, c in zip(("total", "urban", "rural"), v):
                out.append((iso, level, name, area, k, c))
    for k, v in t6[NATIONAL].items():
        if k not in (STATED, POP):
            out.append(("RU", "country", NATIONAL, "total", k, v[0]))
    df = pd.DataFrame(out, columns=["geo_id", "geo_level", "geo_name", "area",
                                    "source_category", "count"])
    df = df[df["count"] > 0]
    df["tier"] = "measured"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")

    sub = df[(df["geo_level"] == "subject") & (df["area"] == "total")]
    drawn = sub[sub["source_category"] != NOT_STATED]["count"].sum()
    gap = sub[sub["source_category"] == NOT_STATED]["count"].sum()
    cr = sub[sub["geo_id"].str.startswith("UA-")]
    cr_drawn = cr[cr["source_category"] != NOT_STATED]["count"].sum()
    print(f"  wrote {OUT.name}: {sub['geo_id'].nunique()} subjects, {drawn:,} with a native "
          f"language, {gap:,} not stated ({100 * gap / (drawn + gap):.1f}%); of them Crimea + "
          f"Sevastopol {cr_drawn:,} with a native language, {cr['count'].sum() - cr_drawn:,} not stated")


if __name__ == "__main__":
    main()
