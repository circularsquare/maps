"""Kyrgyzstan, Population and Housing Census 2022: native language by rayon -> data/normalized/kg.csv.

    python sources/kg_census.py [--fetch]

THE TABLE. *Perepis' naseleniya i zhilishchnogo fonda Kyrgyzskoy Respubliki 2022 goda*, Book III,
"Regiony KR", one volume per oblast and for the cities of Bishkek and Osh (National Statistical
Committee, December 2023 to January 2024), table 3.4, "Raspredelenie naseleniya otdel'nykh
etnicheskikh grupp po rodnomu yazyku". Census question 10.1, "Vash rodnoy yazyk" (your native
language), one answer per person. For every rayon and city of oblast significance, and for the
oblast, the table prints the whole population and a handful of ethnic groups, each split over:

    the language of one's own ethnic group, Kyrgyz, Russian, [Uzbek], [Dungan], other

The bracketed columns are printed only in some volumes (Uzbek in Batken, Jalal-Abad, Osh oblast,
Osh city and Naryn; Dungan in Naryn), and Osh oblast prints Uzbek before Russian. Which ethnic
groups get a row varies by unit: the largest few, so a rayon shows two to eight and the oblast
row more. Bishkek's volume also prints its four city districts; Osh city prints urban and rural
parts (not used).

WHAT A ROW BECOMES. For each unit: one record per listed group and column, with `nationality` the
group and `source_category` the column ("own language: <group>" for the own-language column), and
one record per column for the unit's REMAINDER, the people of every group the unit does not list
(the whole-population row minus the listed rows, column by column). The remainder's own-language
cell is "own language: other groups": the census knows the language, this table does not say it.

THE GRAIN. 52 rayons and cities of oblast significance across the seven oblasts (`unit`), Osh
city (`city`), Bishkek's four districts (`district`) and Bishkek itself (`city`, kept for the
check; countries/kg.py draws the districts or the city, whichever the geography carries), and the
seven oblast rows (`oblast`, not drawn).

CHECKS, all asserted:
  1. every row's columns sum to its whole-population figure, and each group's own column is the
     one printed as "-" in its language's column (pins the column order of every volume);
  2. the units of each oblast sum to the oblast row, column by column (Bishkek's districts to the
     city);
  3. no unit's remainder is negative in any column;
  4. A SECOND PUBLICATION OF THE SAME CENSUS: Book II (national tables, xlsx annex) table 3.8
     prints the same split for every oblast and the country (with an Uzbek column everywhere and
     no Dungan column, so "other" is compared on those terms). Each oblast's whole-population row
     and each group row both volumes print are compared; the differences are printed and must be
     under 0.01% of the oblast. They come to 0.0024%: Batken 8 people between own language and
     other (Book II agrees with Batken's units), Chui 7 between the same two, Issyk-Kul's Tatar
     row 13. The nine regions sum to Book II's national row, 6,936,156.

PRINTING FAULTS handled, each asserted where it is handled: Batken's oblast row (8 people, above);
Naryn's oblast block (the Dungan column double-counted inside "other" on the oblast row, and the
group rows under it garbled, so dropped: nothing is drawn from an oblast row); stray "-" lines
between labels in Osh oblast; Jalal-Abad numbers printed without thousands spaces; page numbers
at the foot of some pages.

RESULT. The remainder's own language ("own language: other groups", groups a unit does not list)
is 7,071 people nationally, 0.10%; every other cell names its language or is the "other" column.

RAW FILES. data/raw/kg/book3_<region>.pdf (nine volumes) and book2_annex.xlsx, from
stat.gov.kg/media/publicationarchive/ (the archive list on the Book I publication page).
"""
import os
import re
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "kg"
OUT = ROOT / "data" / "normalized" / "kg.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
BASE = "https://stat.gov.kg/media/publicationarchive/"
# Book III, one volume per region (stat.gov.kg/ru/publications/perepis-...-kniga-i-.../, the
# publication's archive list), December 2023 to January 2024.
BOOKS = {
    "batken": "dfe4b339-3f9c-4bb7-887e-96e55e15dde0.pdf",
    "jalalabad": "ad52f290-cb14-4ff1-9179-d0ad0f38b09d.pdf",
    "issykkul": "8b892242-eaa9-446d-94b2-7ba7aadcb340.pdf",
    "naryn": "e790d125-ecd1-471e-b31b-0eb9dc7c4894.pdf",
    "osh_oblast": "f76c9a54-0edf-4cfd-91df-8d01379a128b.pdf",
    "talas": "d79c0e75-073d-4a50-92fe-3c735b1a2dfc.pdf",
    "chui": "390cc065-9ead-4ac7-9f6f-2ddfa38a8cbe.pdf",
    "bishkek": "2a645aed-5154-4518-b114-dc97239a3a83.pdf",
    "osh_city": "6d075a2f-51ec-4cec-bf59-455c631ce39b.pdf",
}
# Book II (national tables) and its xlsx annex, for the second-publication check.
BOOK2 = {"book2.pdf": "7bae4592-495e-4703-8093-c15b5e991731.pdf",
         "book2_annex.xlsx": "f96281e2-cbbb-4126-94af-31b317346085.xlsx"}

# Column order of table 3.4 in each volume, after the whole-population figure. Asserted by check 1.
K, R, U, D, O = "Kyrgyz", "Russian", "Uzbek", "Dungan", "other languages"
COLS = {
    "batken": ["own", K, R, U, O],
    "jalalabad": ["own", K, R, U, O],
    "issykkul": ["own", K, R, O],
    "naryn": ["own", K, R, U, D, O],
    "osh_oblast": ["own", K, U, R, O],
    "talas": ["own", K, R, O],
    "chui": ["own", K, R, O],
    "bishkek": ["own", K, R, O],
    "osh_city": ["own", K, R, U, O],
}
LANGS = [K, R, U, D, O]
# the oblast-level row's label in each volume (Book II's label after it)
REGION = {"batken": ("Баткенская область", "Баткенская область"),
          "jalalabad": ("Джалал-Абадская область", "Джалал-Абадская область"),
          "issykkul": ("Иссык-Кульская область", "Иссык-Кульская область"),
          "naryn": ("Нарынская область", "Нарынская область"),
          "osh_oblast": ("Ошская область", "Ошская область"),
          "talas": ("Таласская область", "Таласская область"),
          "chui": ("Чуйская область", "Чуйская область"),
          "bishkek": ("г.Бишкек", "г. Бишкек"),
          "osh_city": ("г.Ош", "г. Ош")}
NATIONAL = 6_936_156
# Batken's oblast row in Book III prints own language 548,760 and other 954; its six units sum to
# 548,752 and 962, which is what Book II table 3.8 prints for the oblast. 8 people moved between
# two columns of one printed total; the units are kept as printed.
MISPRINT = {("batken", "own"): 8, ("batken", O): -8}

# Ethnic-group row labels as printed (lower case, spaces collapsed) -> one spelling per group.
GROUPS = {
    "кыргызы": "Kyrgyz", "узбеки": "Uzbek", "русские": "Russian", "дунгане": "Dungan",
    "таджики": "Tajik", "уйгуры": "Uyghur", "казахи": "Kazakh", "турки": "Turk",
    "азербайджанцы": "Azerbaijani", "татары": "Tatar", "курды": "Kurd", "туркмены": "Turkmen",
    "корейцы": "Korean", "корейлер": "Korean",        # Osh city prints the Kyrgyz plural
    "украинцы": "Ukrainian", "немцы": "German", "калмыки": "Kalmyk", "лезгины": "Lezgin",
    "даргинцы": "Dargin", "чеченцы": "Chechen", "агулы": "Agul", "карачаевцы": "Karachay",
    "балкарцы": "Balkar", "кумыки": "Kumyk", "аварцы": "Avar", "белорусы": "Belarusian",
    "молдаване": "Moldovan", "цыгане": "Roma", "болгары": "Bulgarian", "китайцы": "Chinese",
    "народы индии и пакистана": "peoples of India and Pakistan",
    "пакистанцы и индийцы": "peoples of India and Pakistan",
    "пакистанцы и индусы": "peoples of India and Pakistan",
    "другие": "other groups",                          # Naryn prints a row for the rest
}
# The language whose column a group's own language is (check 1: printed "-" there).
OWN_COL = {"Kyrgyz": K, "Russian": R, "Uzbek": U, "Dungan": D}

NUM = re.compile(r"^(\d+(?:[  ]\d{3})*|-|–)$")    # Jalal-Abad prints some without spaces
JUNK = re.compile(r"^(\(человек\)|в том числе.*|из них.*|все население|все|население|язык своей|"
                  r"своей|национальности|этнической|группы|этнической группы|кыргызский|русский|"
                  r"узбекский|дунганский|продолжение табл.*|.*3\.4.*|родному языку|"
                  r"родным языком|по владению|\d+)$", re.I)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    files = {f"book3_{k}.pdf": v for k, v in BOOKS.items()} | BOOK2
    for name, uid in files.items():
        dst = RAW / name
        if dst.exists():
            print(f"  have {dst.name}")
            continue
        data = urllib.request.urlopen(urllib.request.Request(BASE + uid, headers=UA),
                                      timeout=600).read()
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        print(f"  got  {dst.name} ({len(data):,} bytes)")


def norm(s):
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()


def read_book(book):
    """[(label, kind, {col: value})] for table 3.4 in printed order; kind is unit or group."""
    import fitz

    doc = fitz.open(RAW / f"book3_{book}.pdf")
    pages = [i for i, p in enumerate(doc)
             if re.search(r"3\.4\.? ?Распределение населения отдельных|Продолжение табл\. ?3\.4",
                          p.get_text())]
    cols = COLS[book]
    width = 1 + len(cols)
    rows, run, labels = [], [], []

    def flush():
        vals = [0 if v in ("-", "–") else int(v.replace(" ", "").replace(" ", ""))
                for v in run]
        named = [s for s in labels if s.lower() != "другие" and not JUNK.match(s)]
        if named:
            lab = norm(" ".join(named))
        elif labels and labels[-1].lower() == "другие":
            lab = "другие"
        else:
            raise SystemExit(f"{book}: a run of numbers with no label after {rows[-1][0]!r}")
        kind = "group" if lab.lower() in GROUPS else "unit"
        if len(vals) != width:                      # a label is followed by exactly one row
            raise SystemExit(f"{book}: {lab!r} has {len(vals)} values, expected {width}: {run}")
        rows.append((lab, kind, dict(zip(["total"] + cols, vals))))

    for i in pages:
        lines = [norm(s) for s in doc[i].get_text().split("\n")]
        lines = [s for s in lines if s]
        if lines and re.fullmatch(r"\d+", lines[0]):
            lines = lines[1:]                       # the printed page number, at the top
        if len(lines) > 1 and re.fullmatch(r"\d+", lines[-1]) and not NUM.match(lines[-2]):
            lines = lines[:-1]                      # ... or alone after the footer
        for s in lines:
            if NUM.match(s):
                run.append(s)
                continue
            if run and len(run) < width and all(v in ("-", "–") for v in run):
                run = []                            # a stray dash between labels (Osh oblast)
            if run:
                flush()
                run, labels = [], []
            labels.append(s)
        if run:
            flush()
            run, labels = [], []
    if book == "naryn":
        # Naryn's oblast block (the oblast row and its group rows, before the first rayon) prints
        # the Dungan column AND keeps Dungan inside "other", as Book II (which has no Dungan
        # column) does: those rows overrun their total by exactly the Dungan cell. The rayon
        # blocks do not. Take Dungan out of "other" on the oblast row. The oblast block's GROUP
        # rows are garbled beyond that (Uzbeks overrun by 2, Dungans fall 1 short, and the Kazakh
        # row is Naryn city's, 26 against Book II's 186), so they are dropped: nothing is drawn
        # from an oblast row, and Book II prints the oblast's groups for check 4 anyway.
        v = rows[0][2]
        if sum(v[c] for c in COLS[book]) != v["total"] + v[D]:
            raise SystemExit("naryn: the oblast row no longer overruns by the Dungan cell")
        v[O] -= v[D]
        second = next(i for i, r in enumerate(rows) if i > 0 and r[1] == "unit")
        rows = rows[:1] + rows[second:]
    return rows


def main():
    if "--fetch" in sys.argv or not all((RAW / f"book3_{b}.pdf").exists() for b in BOOKS):
        fetch()

    out, oblast_rows = [], {}
    for book in BOOKS:
        rows = read_book(book)
        cols = COLS[book]
        # 1. rows sum; own column pinned by the "-" in the group's own language's column
        for lab, kind, v in rows:
            if sum(v[c] for c in cols) != v["total"]:
                raise SystemExit(f"{book} {lab}: columns sum to {sum(v[c] for c in cols):,}, "
                                 f"printed {v['total']:,}")
            g = GROUPS.get(lab.lower())
            if kind == "group" and g in OWN_COL and OWN_COL[g] in cols and v[OWN_COL[g]] != 0 \
                    and v["total"] > 0:
                raise SystemExit(f"{book} {lab}: {OWN_COL[g]} column is {v[OWN_COL[g]]}, not '-'")
        # split into blocks: a unit row, then its group rows
        blocks = []
        for lab, kind, v in rows:
            if kind == "unit":
                blocks.append([lab, v, []])
            else:
                if not blocks:
                    raise SystemExit(f"{book}: group row {lab!r} before any unit")
                blocks[-1][2].append((GROUPS[lab.lower()], v))
        region, _b2 = REGION[book]
        if norm(blocks[0][0]).replace(" ", "") != region.replace(" ", ""):
            raise SystemExit(f"{book}: first block is {blocks[0][0]!r}, expected {region!r}")
        oblast_rows[book] = blocks[0]
        sub = [b for b in blocks[1:] if "(без" in b[0] or "население" in b[0].lower()]
        units = [b for b in blocks[1:] if b not in sub]
        print(f"{book:<11} {len(rows):>4} rows, {len(units)} units"
              + (f" (+{len(sub)} sub-rows not used: {[b[0] for b in sub]})" if sub else ""))
        # 2. units sum to the region
        level = "unit"
        if book == "bishkek":
            level = "district"
        for c in ["total"] + cols:
            # Osh city has no units below it; its urban and rural parts are checked instead
            s = sum(b[1][c] for b in (units or sub))
            if s + MISPRINT.get((book, c), 0) != blocks[0][1][c]:
                raise SystemExit(f"{book} {c}: units sum to {s:,}, region row {blocks[0][1][c]:,}")
        if book in ("bishkek", "osh_city"):       # the city itself is also a unit of the map
            out += records(book, "city", blocks[0], cols)
        else:
            out += records(book, "oblast", blocks[0], cols)
        for b in units:
            out += records(book, level, b, cols)
    print("check 1: every row sums and every own-language column is where the '-' says;\n"
          "check 2: the units of every region sum to its row; check 3: no negative remainder")

    # 4. Book II table 3.8
    b2 = pd.read_excel(RAW / "book2_annex.xlsx", sheet_name="3.8", header=None)
    b2 = b2.iloc[:, :7]
    b2.columns = ["label", "total", "own", K, R, U, O]
    b2 = b2[b2["total"].apply(lambda x: isinstance(x, (int, float)) and x == x)]
    b2["label"] = b2["label"].astype(str).map(norm)
    cur, b2rows = None, {}
    for _i, r in b2.iterrows():
        lab = r["label"]
        if lab.lower() in GROUPS:
            b2rows[(cur, GROUPS[lab.lower()])] = r
        else:
            cur = lab
            b2rows[(cur, None)] = r
    nat = b2rows[("Кыргызская Республика", None)]
    if int(nat["total"]) != NATIONAL:
        raise SystemExit(f"Book II national {nat['total']}")
    if sum(oblast_rows[b][1]["total"] for b in BOOKS) != NATIONAL:
        raise SystemExit("the nine regions do not sum to 6,936,156")
    worst = 0.0
    for book in BOOKS:
        lab, v, groups = oblast_rows[book]
        key = REGION[book][1]
        pairs = [(None, v)] + groups
        for g, gv in pairs:
            r = b2rows.get((key, g))
            if r is None:
                continue
            def b2v(c):
                return 0 if r[c] in ("-", None) or r[c] != r[c] else int(r[c])
            for c in ["total", "own", K, R, U, O]:
                if c == U and U not in COLS[book]:
                    continue
                b3 = gv[c]
                v2 = b2v(c)
                if c == O:
                    # Book II prints Uzbek everywhere and Dungan nowhere: a volume without an
                    # Uzbek column holds Uzbek in "other", and Naryn's Dungan column is in
                    # Book II's "other"
                    b3 += gv.get(D, 0)
                    if U not in COLS[book]:
                        v2 += b2v(U)
                if v2 != b3:
                    d = abs(v2 - b3) / v["total"]
                    worst = max(worst, d)
                    print(f"  Book II vs III: {book} {g or 'all'} {c}: {v2:,} vs {b3:,}")
    if worst > 1e-4:
        raise SystemExit(f"Book II and Book III differ by {worst:.4%} of an oblast")
    print(f"check 4: Book II table 3.8 agrees with every oblast row both print, to {worst:.4%}; "
          f"the nine regions sum to {NATIONAL:,}")

    df = pd.DataFrame(out, columns=["book", "geo_level", "unit_name", "nationality",
                                    "source_category", "count"])
    df = df[df["count"] > 0]
    df["tier"] = "measured"
    df.insert(0, "geo_id", df["book"] + ":" + df["unit_name"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    u = df[df["geo_level"].isin(["unit", "district"]) | ((df["book"] == "osh_city")
                                                         & (df["geo_level"] == "city"))]
    print(f"\nwrote {OUT}: {len(df):,} rows; units and districts hold {int(u['count'].sum()):,} "
          "people")
    rem = u[u["source_category"] == "own language: other groups"]["count"].sum()
    print(f"  own language of groups a unit does not list: {int(rem):,} people "
          f"({rem / NATIONAL:.2%})")
    print("  national, by category:")
    tot = u.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for k, n in tot.items():
        print(f"    {k:<46} {int(n):>10,}")


def records(book, level, block, cols):
    lab, v, groups = block
    out = []
    left = {c: v[c] for c in cols}
    for g, gv in groups:
        for c in cols:
            left[c] -= gv[c]
            cat = f"own language: {g}" if c == "own" else c
            out.append((book, level, lab, g, cat, gv[c]))
    for c in cols:
        if left[c] < 0:
            raise SystemExit(f"{book} {lab} {c}: the listed groups exceed the whole population")
        cat = "own language: other groups" if c == "own" else c
        out.append((book, level, lab, "remainder", cat, left[c]))
    return out


if __name__ == "__main__":
    main()
