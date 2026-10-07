"""Tajikistan: 2010 census native language by nationality, and a NATIONALITY MODEL placing it by region.

    python sources/tj_census.py --fetch     copy or download Volume III, then build
    python sources/tj_census.py             build from data/raw/tj/

Writes data/normalized/tj.csv (the census's national table, `measured`) and
data/normalized/tj_model.csv (the five regions, every row `modelled`, which countries/tj.py draws).
sources/tj.md is the record.

THE SOURCE. Population and Housing Census 2010, Volume III, *Natsionalny sostav, vladenie yazykami
i grazhdanstvo naseleniya Respubliki Tadzhikistan* (Agency on Statistics under the President, 2012),
537 pages with a text layer, on the Wayback Machine (stat.tj's own link is dead). The 2020 census's
Volume III is still an empty heading on stat.tj (checked 2026-10-05). The table used is
"Distribution of the republic's population by sex, nationality and native language" (pp.12-56):
for each of the 92 nationalities plus "other" and "not stated", how many consider their native
language (a) the language of their own nationality, or another nationality's: (b) Tajik,
(c) Russian, (d) other languages. Printed for the whole country, then urban and rural.
No volume prints native language for a region or district.

CHECKS. Every row's four columns sum to its total; the rows sum to the printed total row, column
by column; each row's total equals the nationality table on pp.7-11; urban plus rural equals the
total, cell by cell (a second printing of the same table); the seven groups printed by region
(pp.108-115) sum to their national rows.

THE MODEL (as sources/az_model.py, ask 006). Volume III prints nationality by region only for
Tajiks, Uzbeks, Russians, Kyrgyz, Turkmens, Tatars and Kazakhs (pp.108-115); each region's 2010
population (religiondots' tj_lookup.csv, read-only, which Volume I's 2010 column confirms) less
those seven is its `other` column. Then:
  1. the seven groups: their own 2010 count in each region;
  2. the European and other nationalities of non-Muslim heritage (Ukrainians, Germans, Koreans,
     Armenians ... and "other" and "not stated"): on the Russians' regional distribution, the
     placement religiondots' Tajikistan uses (sources/tj.py there);
  3. every other nationality (the Uzbek tribal groups Lakai, Kongrat, Durmen and the rest, Arabs,
     Afghans, Lyuli, Turks, Uyghurs ...): the rest of each region's `other` column, pro rata;
  4. each nationality's people in a region split over native languages by THAT NATIONALITY'S
     NATIONAL SPLIT.
Step 3's remainder is asserted non-negative in every region. Every model row is `modelled`.
"""
import argparse
import os
import re
import shutil
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
from rdlink import RD, RD_GEO  # noqa: E402

RAW = ROOT / "data" / "raw" / "tj"
VOL3 = RAW / "census2010_vol3.pdf"
RD_VOL3 = RD / "data" / "raw" / "tj" / "census2010_vol3.pdf"
VOL3_URL = ("https://web.archive.org/web/20131014054442id_/http://www.stat.tj/ru/img/"
            "526b8592e834fcaaccec26a22965ea2b_1355501132.pdf")
LOOKUP = RD_GEO / "tj" / "tj_lookup.csv"
OUT = ROOT / "data" / "normalized" / "tj.csv"
OUT_MODEL = ROOT / "data" / "normalized" / "tj_model.csv"

TOTAL_2010 = 7_564_502
TOTAL_ROW = (7564502, 7498938, 27809, 7851, 29904)      # p.12, "both sexes", as printed

# The 92 nationalities, "other" and "not stated", in print order (pp.12-17 follow pp.7-11):
# (Russian label as printed, English, placement class). Class "own": placed on its own regional
# count (the seven groups); "ru": on the Russians' regional distribution; "rest": on the rest of
# each region's `other` column. The "ru" set is religiondots' non-Muslim-heritage list.
NATS = [
    ("Таджики", "Tajik", "own"), ("Узбеки", "Uzbek", "own"), ("Русские", "Russian", "own"),
    ("Татары", "Tatar", "own"), ("Кыргызы", "Kyrgyz", "own"), ("Украинцы", "Ukrainian", "ru"),
    ("Немцы", "German", "ru"), ("Туркмены", "Turkmen", "own"), ("Корейцы", "Korean", "ru"),
    ("Казахи", "Kazakh", "own"), ("Евреи", "Jew", "ru"), ("Осетины", "Ossetian", "ru"),
    ("Белорусы", "Belarusian", "ru"), ("Татары крымские", "Crimean Tatar", "rest"),
    ("Татары сибирские", "Siberian Tatar", "rest"), ("Башкиры", "Bashkir", "rest"),
    ("Армяне", "Armenian", "ru"), ("Мордва", "Mordvin", "ru"),
    ("Евреи среднеазиатские", "Central Asian Jew", "ru"), ("Азербайджанцы", "Azerbaijani", "rest"),
    ("Чуваши", "Chuvash", "ru"), ("Афганцы", "Afghan", "rest"), ("Цыгане", "Lyuli (Roma)", "rest"),
    ("Лакцы", "Lak", "rest"), ("Болгары", "Bulgarian", "ru"), ("Грузины", "Georgian", "ru"),
    ("Молдаване", "Moldovan", "ru"), ("Турки (османы)", "Turk", "rest"), ("Поляки", "Pole", "ru"),
    ("Удмурты", "Udmurt", "ru"), ("Марийцы", "Mari", "ru"), ("Греки", "Greek", "ru"),
    ("Уйгуры", "Uyghur", "rest"), ("Литовцы", "Lithuanian", "ru"),
    ("Персы (иране)", "Persian (Iranian)", "rest"), ("Даргинцы", "Dargin", "rest"),
    ("Латыши", "Latvian", "ru"), ("Лезгины", "Lezgin", "rest"), ("Арабы", "Arab", "rest"),
    ("Кабардинцы", "Kabardian", "rest"), ("Аварцы", "Avar", "rest"), ("Караимы", "Karaim", "ru"),
    ("Каракалпаки", "Karakalpak", "rest"), ("Буряты", "Buryat", "ru"), ("Коми", "Komi", "ru"),
    ("Эстонцы", "Estonian", "ru"), ("Чеченцы", "Chechen", "rest"), ("Кумыки", "Kumyk", "rest"),
    ("Ингуши", "Ingush", "rest"), ("Черкесы", "Circassian", "rest"), ("Хакасы", "Khakas", "ru"),
    ("Финны", "Finn", "ru"), ("Коми-пермяки", "Komi-Permyak", "ru"),
    ("Табасараны", "Tabasaran", "rest"), ("Китайцы", "Chinese", "ru"), ("Курды", "Kurd", "rest"),
    ("Карачаевцы", "Karachay", "rest"), ("Абхазы", "Abkhaz", "ru"), ("Болкарцы", "Balkar", "rest"),
    ("Абазины", "Abaza", "rest"), ("Австрийцы", "Austrian", "ru"), ("Американцы", "American", "ru"),
    ("Румыны", "Romanian", "ru"), ("Англичане", "English", "ru"), ("Ненцы", "Nenets", "ru"),
    ("Вьетнамцы", "Vietnamese", "ru"), ("Голландцы", "Dutch", "ru"), ("Испанцы", "Spaniard", "ru"),
    ("Карелы", "Karelian", "ru"), ("Словаки", "Slovak", "ru"), ("Французы", "French", "ru"),
    ("Итальянцы", "Italian", "ru"), ("Японцы", "Japanese", "ru"), ("Дунгане", "Dungan", "rest"),
    ("Коряки", "Koryak", "ru"), ("Венгры", "Hungarian", "ru"), ("Агулы", "Agul", "rest"),
    ("Тофалары", "Tofalar", "ru"), ("Чуванцы", "Chuvan", "ru"), ("Ногайцы", "Nogai", "rest"),
    ("Минги", "Ming", "rest"), ("Дурмены", "Durmen", "rest"), ("Лакайцы", "Lakai", "rest"),
    ("Конграты", "Kongrat", "rest"), ("Катаганы", "Katagan", "rest"), ("Юзы", "Yuz", "rest"),
    ("Барлосы", "Barlos", "rest"), ("Семизы", "Semiz", "rest"), ("Кесамиры", "Kesamir", "rest"),
    ("Народы Индии и Пакистана", "peoples of India and Pakistan", "ru"),
    ("Другие национальности", "other nationalities", "ru"),
    ("Национальность в переписном листе не указана", "nationality not stated", "ru"),
]
COLS = ("own", "Tajik", "Russian", "other languages")
COLS_ALL = ("total",) + COLS
COL_SLACK = 5
NOTE = dict(tajik_pct=84.4, uzbek_pct=11.9, kyrgyz_pct=0.8, russian_pct=0.5, lakai=60392,
            kongrat=37831, six_tribes=19386)

# Volume III's regional tables (pp.108-115), in print order, and the label that opens each block
# (as religiondots' sources/tj.py reads them).
REGION_LABELS = [("TJ-GB", r"Горно\s*-\s*Бадахшанская\s+Автономная\s+область"),
                 ("TJ-SU", r"Согдийская\s+область"),
                 ("TJ-KT", r"Хатлонская\s+область"),
                 ("TJ-DU", r"г\.\s*Душанбе"),
                 ("TJ-RA", r"Города\s+и\s+районы\s+республиканского\s+подчинения")]
REGION_GROUPS = ("Таджики", "Узбеки", "Русские", "Кыргызы", "Туркмены", "Татары", "Казахи")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if VOL3.exists():
        return
    if RD_VOL3.exists():
        shutil.copyfile(RD_VOL3, VOL3)
        print(f"copied {RD_VOL3} -> {VOL3}")
        return
    req = urllib.request.Request(VOL3_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    if not data.startswith(b"%PDF"):
        raise SystemExit(f"{VOL3_URL} is not a PDF")
    VOL3.write_bytes(data)
    print(f"downloaded {VOL3}")


def open_vol3():
    import fitz
    if not VOL3.exists():
        raise SystemExit(f"{VOL3} missing; run with --fetch")
    doc = fitz.open(VOL3)
    if doc.page_count != 537:
        raise SystemExit(f"{VOL3}: {doc.page_count} pages, expected 537 (truncated?)")
    return doc


VALUE = re.compile(r"^(\d+|-)$")


def read_section(doc, first, start_label, stop_label):
    """Rows of the native-language table from the row labelled `start_label` (on printed page
    `first`) until the row labelled `stop_label` (the next section's total).
    Returns [(label text, (total, own, tj, ru, other))]."""
    rows = read_rows(doc, first, stop_label)
    starts = [i for i, (lab, _v) in enumerate(rows) if start_label in lab]
    if len(starts) != 1:
        raise SystemExit(f"p.{first}: {len(starts)} rows labelled {start_label!r}")
    return rows[starts[0]:]


def read_rows(doc, first, stop_label):
    rows, label, vals = [], [], []
    page = first
    while True:
        text = doc[page - 1].get_text()
        lines = text.splitlines()
        cut = max(i for i, l in enumerate(lines) if l.strip() == "другие языки")
        for line in lines[cut + 1:]:
            s = line.strip()
            if not s:
                continue
            if VALUE.match(s):
                vals.append(0 if s == "-" else int(s))
                if len(vals) == 5:
                    lab = re.sub(r"[.…]+", " ", " ".join(label))
                    lab = re.sub(r"\s+", " ", lab).strip()
                    rows.append((lab, tuple(vals)))
                    label, vals = [], []
                    if stop_label in lab and len(rows) > 1:
                        return rows[:-1]
            else:
                if vals:
                    raise SystemExit(f"p.{page}: label {s!r} inside a row of numbers")
                label.append(s)
        page += 1
        if page > first + 20:
            raise SystemExit(f"no {stop_label!r} row within 20 pages of p.{first}")


def check_section(rows, name, expect_total=None):
    """The total row first, then one row per nationality (matched by its Russian label, longest
    match winning, so "Татары крымские" is not "Татары"); returns {en: values}."""
    def norm(s):
        return re.sub(r"[\s\-]", "", s)

    tot_lab, tot = rows[0]
    body = rows[1:]
    if len(body) != len(NATS):
        raise SystemExit(f"{name}: {len(body)} rows, expected {len(NATS)}")
    out, swapped = {}, False
    for lab, v in body:
        lab = lab.replace("Абазинцы", "Абазины")       # the urban table's spelling
        hits = [(len(norm(ru)), en) for ru, en, _c in NATS if norm(ru) in norm(lab)]
        if not hits:
            raise SystemExit(f"{name}: row {lab!r} matches no nationality")
        en = max(hits)[1]
        if en in out:
            raise SystemExit(f"{name}: {en} read twice ({lab!r})")
        if v[0] != sum(v[1:]):
            raise SystemExit(f"{name}, {en}: {v[1:]} sum to {sum(v[1:])}, total {v[0]}")
        out[en] = v
    for j in range(5):
        s = sum(v[j] for v in out.values())
        if s != tot[j]:
            # the volume's own caveat: "small discrepancies between totals and the sum of their
            # parts" (p.2); allowed up to COL_SLACK people and printed
            if abs(s - tot[j]) > COL_SLACK:
                raise SystemExit(f"{name}: column {j} sums to {s:,}, total row {tot[j]:,}")
            print(f"  !! {name}: column {COLS_ALL[j]} sums to {s:,}, the printed total row says "
                  f"{tot[j]:,} ({s - tot[j]:+d})")
            swapped = True
    if expect_total is not None and tot != expect_total:
        raise SystemExit(f"{name}: total row {tot}, expected {expect_total}")
    print(f"  witness: {name}: {len(out)} rows, each summing to its own total; the columns sum to "
          f"the printed total row ({tot[0]:,})" + (" except as above" if swapped else ""))
    return out


def read_regions(doc):
    """{unit: {Russian group label: 2010 count}} for the seven groups printed by region."""
    text = "\n".join(doc[i].get_text() for i in range(107, 115))       # printed pp.108-115
    marks = [(0, "national")]
    for unit, pat in REGION_LABELS:
        m = re.search(pat, text)
        if not m:
            raise SystemExit(f"Volume III p.108-115: no block for {unit}")
        marks.append((m.start(), unit))
    if sorted(p for p, _u in marks) != [p for p, _u in marks]:
        raise SystemExit("Volume III regional blocks are not in print order")
    out = {u: {} for _p, u in marks}
    for m in re.finditer(r"\n(" + "|".join(REGION_GROUPS) + r")\s*\n[^\d]*?Оба пола\s*\n\s*(\d+)",
                         text):
        unit = [u for p, u in marks if p <= m.start()][-1]
        if m.group(1) in out[unit]:
            raise SystemExit(f"{unit}: {m.group(1)} read twice")
        out[unit][m.group(1)] = int(m.group(2))
    for u, d in out.items():
        if set(d) != set(REGION_GROUPS):
            raise SystemExit(f"{u}: read {sorted(d)}, expected the seven groups")
    for g in REGION_GROUPS:
        s = sum(out[u][g] for u, _p in REGION_LABELS)
        if s != out["national"][g]:
            raise SystemExit(f"{g}: regions sum to {s:,}, national row {out['national'][g]:,}")
    print("  witness: the seven groups printed by region sum to their national rows")
    return {u: out[u] for u, _p in REGION_LABELS}


def label_of(en, col):
    return f"own language: {en}" if col == "own" else col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    doc = open_vol3()

    total = check_section(read_section(doc, 12, "Оба пола", "Мужчины"), "total (pp.12-17)",
                          TOTAL_ROW)
    urban = check_section(read_section(doc, 27, "Городское население", "Мужчины"),
                          "urban (pp.27-32)")
    rural = check_section(read_section(doc, 42, "Аҳолии деҳот", "Мужчины"),  # "Сельскоее" sic
                          "rural (pp.42-47)")
    # Urban plus rural is the table used: it equals the printed total row in every column, where
    # the national block's own rows miss it by a 2-person Tajik/Russian swap (Chuvans).
    summed = {en: tuple(u + r for u, r in zip(urban[en], rural[en])) for en in total}
    bad = {en: (total[en], v) for en, v in summed.items() if v != total[en]}
    for en, (t, v) in bad.items():
        print(f"  !! {en}: the national block prints {t}, urban plus rural give {v}; using the latter")
    if set(bad) != {"Chuvan"}:
        raise SystemExit(f"urban plus rural differ from the total for {sorted(bad)}, expected Chuvan")
    for j in range(5):
        if sum(v[j] for v in summed.values()) != TOTAL_ROW[j]:
            raise SystemExit(f"urban plus rural: column {COLS_ALL[j]} misses the printed total")
    print("  witness: urban plus rural equals the national block in every other cell, and its "
          "columns sum exactly to the printed total row")
    total = summed
    if sum(v[0] for v in total.values()) != TOTAL_2010:
        raise SystemExit("rows do not sum to the 2010 census total")

    # national table -> tj.csv
    rows = []
    for (ru, en, _c) in NATS:
        for col, n in zip(COLS, total[en][1:]):
            if n > 0:
                rows.append(("TJ", "nation", "Tajikistan", en, label_of(en, col), n, "measured"))
    nat = pd.DataFrame(rows, columns=["geo_id", "geo_level", "geo_name", "nationality",
                                      "source_category", "count", "tier"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nat.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(nat)} rows, {int(nat['count'].sum()):,} people")

    # the model -> tj_model.csv
    lut = pd.read_csv(LOOKUP)
    if len(lut) != 5 or int(lut["pop2010"].sum()) != TOTAL_2010:
        raise SystemExit(f"{LOOKUP} is not the five regions at {TOTAL_2010:,}")
    pop10 = dict(zip(lut["geo_id"], lut["pop2010"].astype(int)))
    name = dict(zip(lut["geo_id"], lut["name"]))
    reg = read_regions(doc)
    en_of = {ru: en for ru, en, _c in NATS}
    for g in REGION_GROUPS:
        if sum(reg[u][g] for u in pop10) != total[en_of[g]][0]:
            raise SystemExit(f"{g}: regional sum differs from the language table's total")
    other10 = {u: pop10[u] - sum(reg[u].values()) for u in pop10}
    ru_sum = sum(reg[u]["Русские"] for u in pop10)
    place = {}                                   # en -> {unit: people}
    for ru, en, c in NATS:
        n = total[en][0]
        if c == "own":
            place[en] = {u: float(reg[u][ru]) for u in pop10}
        elif c == "ru":
            place[en] = {u: n * reg[u]["Русские"] / ru_sum for u in pop10}
    rest_room = {u: other10[u] - sum(place[en][u] for ru, en, c in NATS if c == "ru")
                 for u in pop10}
    for u, r in rest_room.items():
        if r < 0:
            raise SystemExit(f"{name[u]}: the 'ru' groups exceed the region's other column")
    rest_total = sum(total[en][0] for ru, en, c in NATS if c == "rest")
    if abs(sum(rest_room.values()) - rest_total) > 0.5:
        raise SystemExit(f"rest room {sum(rest_room.values()):,.1f} != rest groups {rest_total:,}")
    for ru, en, c in NATS:
        if c == "rest":
            place[en] = {u: total[en][0] * rest_room[u] / rest_total for u in pop10}
    print("  the rest of each region's `other` column (Uzbek tribal groups, Arabs, Afghans ...):")
    for u in pop10:
        print(f"    {name[u]:<40} other {other10[u]:>7,}  of which rest {rest_room[u]:>9,.0f}")

    mrows = []
    for ru, en, c in NATS:
        v = total[en]
        if v[0] == 0:
            continue
        for u, people in place[en].items():
            for col, n in zip(COLS, v[1:]):
                if n > 0 and people > 0:
                    mrows.append((u, "unit", name[u], en, label_of(en, col), people * n / v[0],
                                  "modelled"))
    mod = pd.DataFrame(mrows, columns=["geo_id", "geo_level", "geo_name", "nationality",
                                       "source_category", "count", "tier"])
    by_u = mod.groupby("geo_id")["count"].sum()
    for u in pop10:
        if abs(by_u[u] - pop10[u]) > 0.5:
            raise SystemExit(f"{name[u]}: model holds {by_u[u]:,.1f}, census {pop10[u]:,}")
    a1 = mod.groupby("source_category")["count"].sum()
    a2 = nat.groupby("source_category")["count"].sum()
    if (a1.reindex(a2.index) - a2).abs().max() > 0.5:
        raise SystemExit("model's national totals differ from the census table")
    print("  witness: the model reproduces every region's 2010 population and every national cell")
    mod.to_csv(OUT_MODEL, index=False, encoding="utf-8")
    print(f"wrote {OUT_MODEL}: {len(mod)} rows")

    # the figures note_public quotes
    lang = {"own language: Tajik": "Tajik", "Tajik": "Tajik", "own language: Uzbek": "Uzbek",
            "own language: Russian": "Russian", "Russian": "Russian",
            "own language: Kyrgyz": "Kyrgyz", "own language: Turkmen": "Turkmen"}
    k = mod.assign(lang=mod["source_category"].map(lang).fillna(mod["source_category"]))
    print("\n  national, native language:")
    t = k.groupby("lang")["count"].sum().sort_values(ascending=False)
    for l, n in t.head(14).items():
        print(f"    {l:<40} {n:>10,.0f}  {n / TOTAL_2010:.2%}")
    print("\n  by region, top languages:")
    for u in pop10:
        s = k[k["geo_id"] == u].groupby("lang")["count"].sum().sort_values(ascending=False)
        print(f"    {name[u]:<40} " + ", ".join(f"{l} {n / pop10[u]:.1%}" for l, n in s.head(5).items()))

    # countries/tj.py note_public quotes these; asserted so the note cannot drift from the data
    tribes = ("Durmen", "Katagan", "Yuz", "Ming", "Kesamir", "Semiz")
    got = dict(tajik_pct=round(100 * t["Tajik"] / TOTAL_2010, 1),
               uzbek_pct=round(100 * t["Uzbek"] / TOTAL_2010, 1),
               kyrgyz_pct=round(100 * t["Kyrgyz"] / TOTAL_2010, 1),
               russian_pct=round(100 * t["Russian"] / TOTAL_2010, 1),
               lakai=total["Lakai"][1], kongrat=total["Kongrat"][1],
               six_tribes=sum(total[x][1] for x in tribes))
    print(f"\n  note_public's figures: {got}")
    if got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
