"""Turkmenistan, Complete Population and Housing Census 2022: mother tongue by velayat
-> data/normalized/tm.csv.

    python sources/tm_census.py [--fetch]

THE TABLE. State Committee of Turkmenistan on Statistics, *Results of the Complete Population
and Housing Census of Turkmenistan 2022*, section 4, "National composition of the population and
language proficiency" (English, 61 pages, `stat.gov.tm/population-census-pdfs/results/en/4.pdf`,
the index is `stat.gov.tm/en/population-census`). Census day 17 December 2022. Tables 4.9-4.29
print, for the whole country and for Ashgabat city and each of the five velayats, both sexes,
male and female: population by nationality (62 rows plus "All nationalities") and "considered as
their mother tongue", ten columns:

    Turkmen, Russian, Ukrainian, Uzbek, Kazakh, Tatar, Armenian, Azerbaijani, Baloch,
    other languages

One answer per person. Nothing below velayat is published (no etrap table in any section).
Arkadag city (velayat status 2023) is counted inside Ahal, as in religiondots' `sources/tm_geo.py`.

THE PARSE. The PDF has a text layer. Numbers are right-aligned with a space as the thousands
separator, so the words of one line are joined into numbers where the gap between two digit
groups is under 4 pt (inside a number it is 2.3-2.5 pt; between columns at least 6 pt), and every
data line must give exactly eleven numbers ("-" is 0). A label that wraps onto a line without
numbers is carried into the next line's label. The column headers are rotated words; their order
is asserted on every page that prints them.

CHECKS, all asserted:
  1. every row's ten language columns sum to its total;
  2. in every table the nationality rows sum to "All nationalities", column by column;
  3. male plus female equals both sexes, cell by cell, for all seven geographies;
  4. the six units sum to the national table, cell by cell;
  5. each unit's total equals section 1's table 1.3 (religiondots' `tm_lookup.csv`, read-only),
     and the fifteen nationalities religiondots transcribed from tables 4.1 and 4.3-4.8
     (`tm_nationality.csv`) equal the totals printed here.

OUTPUT: one row per (unit, nationality, language) with a non-zero count, both sexes, `tier`
measured. `nationality` is kept for the record; countries/tm.py sums over it.
"""
import argparse
import re
import shutil
import statistics
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD, RD_GEO  # noqa: E402

RAW = ROOT / "data" / "raw" / "tm"
PDF = RAW / "tm_census2022_results_en_4.pdf"
RD_PDF = RD / "data" / "raw" / "tm" / "tm_census2022_results_en_4.pdf"
URL = "https://www.stat.gov.tm/population-census-pdfs/results/en/4.pdf"
OUT = ROOT / "data" / "normalized" / "tm.csv"
N_PAGES = 61
LAST_PAGE = 32          # tables 4.30-4.50 (knowledge of other languages) start on page 33

LANGS = ["Turkmen", "Russian", "Ukrainian", "Uzbek", "Kazakh", "Tatar", "Armenian",
         "Azerbaijani", "Baloch", "other languages"]
HEADER = ["Turkmen", "Russian", "Ukrainian", "Uzbek", "Kazakh", "Tatar", "Armenian",
          "Azerbaijani", "Baloch", "other"]

# table number -> (geography, sex). 4.9-4.11 national; then each unit's both / male / female.
TABLES = {9: ("TM", "both"), 10: ("TM", "male"), 11: ("TM", "female")}
for i, g in enumerate(["TM-S", "TM-A", "TM-B", "TM-D", "TM-L", "TM-M"]):
    for j, sex in enumerate(["both", "male", "female"]):
        TABLES[12 + 3 * i + j] = (g, sex)
UNITS = ["TM-S", "TM-A", "TM-B", "TM-D", "TM-L", "TM-M"]
NAMES = {"TM-S": "Ashgabat", "TM-A": "Ahal", "TM-B": "Balkan", "TM-D": "Dashoguz",
         "TM-L": "Lebap", "TM-M": "Mary"}
TITLE_GEO = {"TM": "of Turkmenistan", "TM-S": "Ashgabat", "TM-A": "Ahal", "TM-B": "Balkan",
             "TM-D": "Dashoguz", "TM-L": "Lebap", "TM-M": "Mary"}
ALL = "All nationalities"
GAP = 4.0
DASHES = ("−", "–", "-")
NUM = r"\d+|[−–-]"
LINE_TOL = 11.0
# Pages whose printed header is wrong. Page 10 (table 4.12, Ashgabat) prints "Kazakh" over the
# sixth column, where every other page prints "Tatar". It is Tatar: the Tatars' row has 664 of
# its 2,585 there and the Kazakhs' 3 of 703 (392 sit in the fifth column), and check 4 (units
# sum to the national table, cell by cell) pins it.
HEADER_TYPOS = {10: ["Turkmen", "Russian", "Ukrainian", "Uzbek", "Kazakh", "Kazakh", "Armenian",
                     "Azerbaijani", "Baloch", "other"]}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if PDF.exists():
        return
    if RD_PDF.exists():
        shutil.copyfile(RD_PDF, PDF)
        print(f"copied {RD_PDF} -> {PDF}")
        return
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; "
                                               "Win64; x64) AppleWebKit/537.36 Chrome/131.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    if not data.startswith(b"%PDF") or b"%%EOF" not in data[-2048:]:
        raise SystemExit(f"{URL} is not a whole PDF")
    PDF.write_bytes(data)
    print(f"downloaded {PDF}")


def _num(tok):
    return 0 if tok in DASHES else int(tok)


def _lines(page):
    """Words grouped into lines, left to right: [(y, [(x0, x1, h, w)])].

    Rows are 13.5 pt apart or more. A few rows are typeset across two baselines about 10 pt
    apart (table 4.24's Turkmens: the label and the end of one figure sit one baseline below
    the rest), so words whose top edges are within LINE_TOL of the previous word's join it."""
    words = sorted((y0, x0, x1, y1 - y0, w) for x0, y0, x1, y1, w, *_ in page.get_text("words"))
    out = []
    for y0, x0, x1, h, w in words:
        if out and y0 - out[-1][2] < LINE_TOL:
            out[-1][1].append((x0, y0, x1, h, w))
            out[-1][2] = y0
        else:
            out.append([y0, [(x0, y0, x1, h, w)], y0])
    res = []
    for y, v, _ in out:
        v.sort()
        for i in range(len(v) - 1):        # same x on two baselines: the upper one first
            if abs(v[i][0] - v[i + 1][0]) < 1 and v[i][1] > v[i + 1][1]:
                v[i], v[i + 1] = v[i + 1], v[i]
        res.append((y, [(x0, x1, h, w) for x0, _, x1, h, w in v]))
    return res


def _rules(page):
    """The vertical rules of each table header on the page, as [(bottom y, [x, ...])]: the
    rules of one header end at the same y. The first rule (left of the Total column) is
    dropped, so a figure's column is the number of rules left of its right edge."""
    by_bottom = {}
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l" and abs(it[1].x - it[2].x) < 0.5 and abs(it[1].y - it[2].y) > 30:
                x, y1 = it[1].x, max(it[1].y, it[2].y)
            elif it[0] == "re" and it[1].width < 2 and it[1].height > 30:
                x, y1 = it[1].x0, it[1].y1
            else:
                continue
            key = next((k for k in by_bottom if abs(k - y1) < 3), y1)
            by_bottom.setdefault(key, set()).add(round(x, 1))
    out = []
    for y1, xs in sorted(by_bottom.items()):
        xs = sorted(xs)
        if len(xs) != 11:
            raise SystemExit(f"page {page.number + 1}: a header band at y {y1:.0f} has "
                             f"{len(xs)} rules, expected 11: {xs}")
        out.append((y1, xs[1:]))
    return out


def _join(nums):
    """Words of one line -> (numbers as strings, their right edges): digit groups closer than
    GAP join into one number; a dash is a cell of its own."""
    vals, ends, prev_x1, acc = [], [], None, None
    for x0, x1, w in nums:
        if acc is not None and w not in DASHES and acc not in DASHES and x0 - prev_x1 < GAP:
            acc += w
        else:
            if acc is not None:
                vals.append(acc)
                ends.append(prev_x1)
            acc = w
        prev_x1 = x1
    if acc is not None:
        vals.append(acc)
        ends.append(prev_x1)
    return vals, ends


def _unique(rows, name, n):
    if name in rows:
        raise SystemExit(f"table 4.{n}: {name!r} twice")
    return name


def parse():
    import fitz
    if not PDF.exists():
        raise SystemExit(f"{PDF} missing; run with --fetch")
    doc = fitz.open(PDF)
    if doc.page_count != N_PAGES:
        raise SystemExit(f"{PDF}: {doc.page_count} pages, expected {N_PAGES} (truncated?)")
    tables, cur, label, blanks = {}, None, "", []
    for pno, page in enumerate(doc, 1):
        rule_bands = None
        # the rotated column headers, in x order, wherever the page prints them
        # (a page holding the end of one table and the start of the next prints them twice,
        # one band each)
        heads = sorted((y0, x0, w) for x0, y0, x1, y1, w, *_ in page.get_text("words")
                       if (y1 - y0) > 1.5 * (x1 - x0) and w in HEADER)
        bands = []
        for y0, x0, w in heads:
            if not bands or y0 - bands[-1][-1][0] > 30:
                bands.append([])
            bands[-1].append((y0, x0, w))
        for band in bands:
            got = [w for _, _, w in sorted(band, key=lambda t: t[1])]
            if pno <= LAST_PAGE and got != HEADER and got != HEADER_TYPOS.get(pno):
                raise SystemExit(f"page {pno}: language columns {got}, expected {HEADER}")
        page_text = " ".join(w for _, ln in _lines(page) for *_, w in ln)
        for y, line in _lines(page):
            text = " ".join(w for *_, w in line)
            m = re.match(r"4\.(\d+)\.", text)
            if m:
                n = int(m.group(1))
                cur = n if n in TABLES else None
                if cur is not None:
                    geo, sex = TABLES[n]
                    if TITLE_GEO[geo] not in text or "mother" not in page_text or \
                            ((sex == "male") != ("(male)" in text)) or \
                            ((sex == "female") != ("(female)" in text)):
                        raise SystemExit(f"page {pno}: title {text!r} is not table 4.{n} "
                                         f"({geo}, {sex})")
                    if n in tables:
                        raise SystemExit(f"table 4.{n} printed twice")
                    tables[n] = {}
                label = ""
                continue
            if cur is None or text.startswith(("RESULTS OF THE", "AND HOUSING CENSUS")):
                continue
            words = [(x0, x1, w) for x0, x1, h, w in line if h < 20]   # skip rotated headers
            lab = [w for x0, x1, w in words if not re.fullmatch(NUM, w)]
            nums = [(x0, x1, w) for x0, x1, w in words if re.fullmatch(NUM, w)]
            if not nums:
                if lab and lab[0] not in ("RESULTS", "AND", "and", "Total", "including", "Ending", "Continuation",
                                          "including:", "of", "languages", "other"):
                    label = (label + " " + " ".join(lab)).strip()
                continue
            vals, ends = _join(nums)
            name = (label + " " + " ".join(lab)).strip()
            label = ""
            if len(vals) == 11:
                tables[cur][_unique(tables[cur], name, cur)] = [_num(v) for v in vals]
                continue
            # A line with fewer than eleven figures: each goes to the column whose header rules
            # enclose its right edge (the header band nearest above the line). The figures are
            # right-aligned just inside the rule on their right, though some tables' rules sit
            # 3 pt off their figures, so this is used only where it has to be, and check 3
            # (male + female = both) tests every cell it places.
            if rule_bands is None:
                rule_bands = _rules(page)
            above = [b for b in rule_bands if b[0] < y]
            if not above:
                raise SystemExit(f"page {pno}, table 4.{cur}, {name!r}: no header rules above")
            rules = above[-1][1]
            col = [sum(r < e - 1.5 for r in rules) for e in ends]   # a figure can touch its rule
            if len(set(col)) != len(col) or max(col) > 10:
                raise SystemExit(f"page {pno}, table 4.{cur}, {name!r}: cannot place {vals} "
                                 f"(right edges {ends}) on the rules {rules}")
            full = [None] * 11
            for k, v in zip(col, vals):
                full[k] = v
            if None in full:
                # A blank cell, no figure and no dash. Seen only in table 4.10 (male, national)
                # where the last column is not printed for 13 rows.
                if full.count(None) != 1 or full[0] is None:
                    raise SystemExit(f"page {pno}, table 4.{cur}, {name!r}: {full.count(None)} "
                                     "blank cells; only one can be read as the row's residual")
                k = full.index(None)
                full[k] = str(_num(full[0]) - sum(_num(v) for v in full[1:] if v is not None))
                blanks.append((cur, name, (["total"] + LANGS)[k], int(full[k])))
            tables[cur][_unique(tables[cur], name, cur)] = [_num(v) for v in full]
    missing = sorted(set(TABLES) - set(tables))
    if missing:
        raise SystemExit(f"tables not found: {missing}")
    print(f"  {len(blanks)} rows print one cell blank (no figure, no dash), filled as the row's "
          "residual and tested by check 3 against the both-sexes table: "
          + "; ".join(f"4.{n} {nm} {c}={v}" for n, nm, c, v in blanks))
    if any(v < 0 for *_, v in blanks):
        raise SystemExit("a blank cell's residual is negative")
    return tables


def check(tables):
    frames = {}
    for n, rows in tables.items():
        df = pd.DataFrame.from_dict(rows, orient="index", columns=["total"] + LANGS)
        if ALL not in df.index:
            raise SystemExit(f"table 4.{n}: no {ALL!r} row; rows {list(df.index)[:5]}")
        # 1. columns sum to the total
        bad = df[df[LANGS].sum(axis=1) != df["total"]]
        if len(bad):
            raise SystemExit(f"table 4.{n}: language columns do not sum to the total:\n{bad}")
        # 2. nationality rows sum to All nationalities
        diff = df.drop(index=ALL).sum() - df.loc[ALL]
        if diff.any():
            raise SystemExit(f"table 4.{n}: nationality rows minus {ALL!r}:\n{diff[diff != 0]}")
        frames[TABLES[n]] = df
    print(f"  {len(tables)} tables; every row's ten languages sum to its total and every "
          "table's nationality rows sum to its All nationalities row")
    # 3. male + female = both
    for geo in ["TM"] + UNITS:
        b, m, f = (frames[(geo, s)] for s in ("both", "male", "female"))
        idx = b.index.union(m.index).union(f.index)
        d = (m.reindex(idx, fill_value=0) + f.reindex(idx, fill_value=0)
             - b.reindex(idx, fill_value=0))
        if d.to_numpy().any():
            raise SystemExit(f"{geo}: male + female differs from both sexes:\n"
                             f"{d[(d != 0).any(axis=1)]}")
    print("  male + female equals both sexes in every cell, for the country and all six units")
    # 4. units sum to the national table
    nat = frames[("TM", "both")]
    idx = nat.index
    s = sum(frames[(g, "both")].reindex(idx, fill_value=0) for g in UNITS)
    extra = set().union(*(frames[(g, "both")].index for g in UNITS)) - set(idx)
    if extra or (s - nat).to_numpy().any():
        raise SystemExit(f"units against table 4.9: rows only in units {sorted(extra)}; "
                         f"differences:\n{(s - nat)[((s - nat) != 0).any(axis=1)]}")
    print(f"  the six units sum to table 4.9 in every cell ({len(idx) - 1} nationality rows)")
    # 5. religiondots' table 1.3 and tables 4.1/4.3-4.8 transcriptions
    lut = pd.read_csv(RD_GEO / "tm" / "tm_lookup.csv").set_index("geo_id")
    for g in UNITS:
        got = frames[(g, "both")].loc[ALL, "total"]
        if got != lut.loc[g, "pop"]:
            raise SystemExit(f"{g}: {got:,} here, table 1.3 says {lut.loc[g, 'pop']:,}")
    natn = pd.read_csv(RD_GEO / "tm" / "tm_nationality.csv")
    plural = {"Balochi": "Balochi"}
    n_chk = 0
    for r in natn.itertuples():
        if r.nationality == "other nationalities":
            continue
        name = plural.get(r.nationality, r.nationality)
        got = frames[(r.geo_id, "both")]["total"].get(name)
        if got != r.persons:
            raise SystemExit(f"{r.geo_id} {name}: {got} here, tables 4.3-4.8 say {r.persons:,}")
        n_chk += 1
    print(f"  unit totals equal table 1.3; {n_chk} nationality totals equal tables 4.3-4.8 "
          "(religiondots' transcription)")
    return frames


def note_figures(frames):
    """countries/tm.py's note_public quotes these; a change in the data fails here."""
    nat = frames[("TM", "both")]
    tot = nat.loc[ALL, "total"]
    leb = frames[("TM-L", "both")]

    def pct(a, b, nd=1):
        return round(100 * a / b, nd)
    got = {
        "Turkmen": pct(nat.loc[ALL, "Turkmen"], tot), "Uzbek": pct(nat.loc[ALL, "Uzbek"], tot),
        "Russian": pct(nat.loc[ALL, "Russian"], tot), "Baloch": pct(nat.loc[ALL, "Baloch"], tot),
        "Uzbeks naming Turkmen": round(100 * nat.loc["Uzbeks", "Turkmen"]
                                       / nat.loc["Uzbeks", "total"]),
        "Lebap Uzbeks naming Turkmen": round(100 * leb.loc["Uzbeks", "Turkmen"]
                                             / leb.loc["Uzbeks", "total"]),
        "Lebap Uzbek": pct(leb.loc[ALL, "Uzbek"], leb.loc[ALL, "total"]),
        "Lebap Uzbeks": pct(leb.loc["Uzbeks", "total"], leb.loc[ALL, "total"]),
        "Dashoguz Uzbek": pct(frames[("TM-D", "both")].loc[ALL, "Uzbek"],
                              frames[("TM-D", "both")].loc[ALL, "total"]),
        "Mary Baloch": pct(frames[("TM-M", "both")].loc[ALL, "Baloch"],
                           frames[("TM-M", "both")].loc[ALL, "total"]),
        "other": int(nat.loc[ALL, "other languages"]),
        "other %": pct(nat.loc[ALL, "other languages"], tot, 2),
    }
    want = {"Turkmen": 89.2, "Uzbek": 6.8, "Russian": 1.9, "Baloch": 1.2,
            "Uzbeks naming Turkmen": 26, "Lebap Uzbeks naming Turkmen": 71, "Lebap Uzbek": 2.8,
            "Lebap Uzbeks": 9.4, "Dashoguz Uzbek": 27.5, "Mary Baloch": 5.0, "other": 17_596,
            "other %": 0.25}
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if bad:
        raise SystemExit(f"note_public figures moved (got, written): {bad}")
    five = ["Persians", "Afghans", "Karakalpaks", "Lezgins", "Kurds"]
    o = nat.drop(index=ALL)["other languages"].sort_values(ascending=False)
    if [n for n in o.index if n != "Turkmens"][:5] != five or o[five].sum() * 2 <= o.sum():
        raise SystemExit(f"'other languages' is no longer mostly {five}: {o.head(8).to_dict()}")
    print("  note_public's figures match the tables")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    frames = check(parse())
    rows = []
    for g in UNITS:
        df = frames[(g, "both")].drop(index=ALL)
        for nat, r in df.iterrows():
            for lang in LANGS:
                if r[lang]:
                    rows.append((g, "velayat", NAMES[g], nat, lang, int(r[lang]), "measured"))
    out = pd.DataFrame(rows, columns=["geo_id", "geo_level", "geo_name", "nationality",
                                      "source_category", "count", "tier"])
    total = int(frames[("TM", "both")].loc[ALL, "total"])
    if out["count"].sum() != total:
        raise SystemExit(f"wrote {out['count'].sum():,}, census {total:,}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT} ({len(out):,} rows, {total:,} people)")
    note_figures(frames)
    nat = frames[("TM", "both")].loc[ALL, LANGS]
    print("\n  national, mother tongue:")
    for lang, v in nat.sort_values(ascending=False).items():
        print(f"    {lang:<16}{v:>10,}  {100 * v / total:6.2f}%")
    print("\n  by unit, % (Turkmen / Uzbek / Russian / Baloch / Azerbaijani / Kazakh / other):")
    for g in UNITS:
        r = frames[(g, "both")].loc[ALL]
        print(f"    {NAMES[g]:<10}{r['total']:>10,}  " + "  ".join(
            f"{100 * r[c] / r['total']:5.2f}" for c in
            ["Turkmen", "Uzbek", "Russian", "Baloch", "Azerbaijani", "Kazakh", "other languages"]))


if __name__ == "__main__":
    main()
