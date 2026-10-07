"""Armenia: Statistical Committee (Armstat), 2011 Population Census, table 5.2-1, population by
ethnicity, sex and mother tongue, one table per marz. Writes data/normalized/am.csv.

    python sources/am_census.py --fetch     12 one-page PDFs, about 1 MB, into data/raw/am/
    python sources/am_census.py             normalise from data/raw/am/

WHY 2011 AND NOT 2022 (sources/am.md has the numbers). Both censuses asked mother tongue and both
publish it for the eleven marzes. The 2022 table 5.2 does not name the language: its columns are
"the language of his nationality", "other tongue" and "refused", crossed with ethnicity, so the
46,846 people (1.6%) whose mother tongue differs from their ethnicity are an unnamed remainder,
and the rest are named only through the ethnicity row. 2011's table 5.2-1 prints each mother
tongue by name (Armenian, Yezidi, Russian, Assyrian, Kurdish, Ukrainian, English, Georgian,
Persian, Greek, Other) and lets the 2011 figure stand on its own answer.

THE TABLES. Each marz volume (`/am/?nid=533` Yerevan to `?nid=543` Tavush) carries its table 5.2-1
as a separate one-page-per-block PDF, Armenian only (the `/en/` marz pages are empty, as
religiondots found for religion). The national volume `?nid=532` has the same table; its English
twin (doc 99486258) is used only for Armstat's English wording, which is what am.csv carries.
Only the first block is read: the marz's total row (`Ընդամենը` or the marz's own name), before
the ethnicity rows and the man/woman and urban/rural repeats.

READ BY GEOMETRY, NOT BY TEXT ORDER. The column heads are set vertically and PyMuPDF's plain text
puts Yerevan's total row in the wrong place, so each file is read from word boxes: the rotated
words of the head, clustered by x into columns (`Հրաժարվել են պատասխանել` is three rotated words
on two lines), with `Ընդամենը` (Total) as the first column; then the first line below the head
holding exactly that many numbers. The numbers are right-aligned under their heads, so they are
taken in x order and must be exactly as many as the columns.

EACH MARZ PRINTS ONLY THE LANGUAGES IT HAS PEOPLE FOR, so the column set differs by file (Yerevan
has no Kurdish column, Aragatsotn has five languages). A language a marz gives no column is inside
that marz's Other. The national table is therefore checked as religiondots' am.py checks religion:
every marz's columns sum to its own total exactly; the eleven totals sum to the national total
exactly; a language's marz sum is a floor on its national figure, and Other's excess over the
national Other equals the sum of those shortfalls.

THE ETHNICITY ROWS ARE THE SECOND CUT. Below the total row each table splits the same people by
ethnicity. Those rows must add up to the total row column by column (all eleven marzes and the
country do, except Tavush's Other column, pinned in KNOWN_ROW_GAPS), and since a head read onto
the wrong column would still add up, the column ORDER is checked on them too: every nationality
of 300+ in a unit names its own language more than any language but Armenian.
"""
import csv
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "am")
OUT = os.path.join(ROOT, "data", "normalized", "am.csv")

SOURCE_ID = "am_census_2011_t5_2_1"
YEAR = 2011
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DOC = "https://www.armstat.am/file/doc/{}.pdf"

# nid -> (ISO 3166-2 code, as religiondots' marz layer keys them; English name; Armstat doc id of
# that volume's table 5.2-1). Doc ids read off each /am/?nid= page on 2026-10-05.
MARZES = {
    533: ("AM-ER", "Yerevan", 99483743),
    534: ("AM-AG", "Aragatsotn", 99480793),
    535: ("AM-AR", "Ararat", 99481133),
    536: ("AM-AV", "Armavir", 99481493),
    537: ("AM-GR", "Gegharkunik", 99481853),
    538: ("AM-LO", "Lori", 99482688),
    539: ("AM-KT", "Kotayk", 99482348),
    540: ("AM-SH", "Shirak", 99483063),
    541: ("AM-SU", "Syunik", 99483413),
    542: ("AM-VD", "Vayots Dzor", 99484073),
    543: ("AM-TV", "Tavush", 99485378),
}
NATIONAL_DOC = 99478358          # ?nid=532, Armenian
NATIONAL_EN_DOC = 99486258       # ?nid=532 under /en/, for Armstat's English wording only

# Armenian head -> Armstat's English head from the national English table (doc 99486258).
# `Yezidian` is Armstat's English spelling and is kept as printed.
HY_TO_EN = {
    "Հայերեն": "Armenian",
    "Եզդիերեն": "Yezidian",
    "Ռուսերեն": "Russian",
    "Ասորերեն": "Assyrian",
    "Քրդերեն": "Kurdish",
    "Ուկրաիներեն": "Ukrainian",
    "Անգլերեն": "English",
    "Վրացերեն": "Georgian",
    "Պարսկերեն": "Persian",
    "Հունարեն": "Greek",
    "Այլ": "Other",
    "Այլլեզու": "Other",                 # `Այլ լեզու`, "other language", in some marzes
    "Հրաժարվելենպատասխանել": "Refused to answer",
    # Yerevan's `են` is a two-letter word whose rotated box comes out about square, so the
    # rotation test drops it and the head reads without it.
    "Հրաժարվելպատասխանել": "Refused to answer",
}
TOTAL_HY = "Ընդամենը"
# Stubs that open the repeats after the ethnicity block: men, women, urban, rural.
BLOCK_ENDS = ("Տղամարդ", "Կին", "Կանայք", "Քաղաք", "Գյուղ")
TOTAL_CAT = "Total"

# Published in the English national table (doc 99486258), typed here as the target the read of the
# Armenian national table must reproduce.
NATIONAL_EN = {
    "Total": 3_018_854, "Armenian": 2_956_615, "Yezidian": 30_973, "Russian": 23_484,
    "Assyrian": 2_402, "Kurdish": 2_030, "Ukrainian": 733, "English": 491, "Georgian": 455,
    "Persian": 397, "Greek": 332, "Other": 913, "Refused to answer": 29,
}


def _fold(s):
    s = unicodedata.normalize("NFKC", str(s or "")).replace("­", "")
    return re.sub(r"[\s\-‐-―−]+", "", s)


def _num(s):
    s = s.replace(",", "").replace(" ", "")
    return int(s) if re.fullmatch(r"\d+", s) else None


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    docs = [(f"{code}.pdf", doc) for code, _, doc in MARZES.values()]
    docs += [("national_hy.pdf", NATIONAL_DOC), ("national_en.pdf", NATIONAL_EN_DOC)]
    for name, doc in docs:
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 5_000:
            print(f"  have {name}")
            continue
        r = requests.get(DOC.format(doc), headers=UA, timeout=300)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF") or b"%%EOF" not in r.content[-1024:]:
            raise SystemExit(f"{doc}: not a whole PDF ({len(r.content)} bytes)")
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  {name}: {len(r.content):,} bytes  <- {DOC.format(doc)}")


def read_total_row(path):
    """First page of one table 5.2-1 -> {English head: count}, the marz (or country) total row."""
    import fitz
    page = fitz.open(path)[0]
    words = page.get_text("words")       # x0, y0, x1, y1, text, block, line, word

    # The head's Total is the horizontal `Ընդամենը` away from the left margin; the stub's
    # `Ընդամենը` (the total row's label, where there is one) sits at the margin. It is also the
    # topmost: the stub's is below the head.
    margin = min(w[0] for w in words)
    tot = sorted((w for w in words if _fold(w[4]) == TOTAL_HY and w[0] > margin + 40),
                 key=lambda w: w[1])
    if not tot:
        raise SystemExit(f"{path}: no `Ընդամենը` column head")
    total_head = tot[0]
    # Language heads sit right of the Total column, below the spanning `Մայրենի լեզուն` (mother
    # tongue) and above the first number. Ten files set them vertically; Tavush sets them
    # horizontally (and its Total vertically), so orientation is not tested.
    span = [w for w in words if _fold(w[4]) == "լեզուն"]
    top = span[0][3] if span else total_head[1]
    first_num = min(w[1] for w in words if _num(w[4]) is not None and w[1] > top
                    and w[0] > total_head[0] - 30)
    rotated = [w for w in words if len(w[4]) > 1 and _num(w[4]) is None
               and w[0] > total_head[2] and w[3] > top + 2 and w[3] < first_num]
    if not rotated:
        raise SystemExit(f"{path}: no language column heads")
    head_bottom = max(w[3] for w in rotated)

    # Cluster the rotated head words by x: words of one head overlap or touch in x.
    rotated.sort(key=lambda w: w[0])
    clusters = []
    for w in rotated:
        if clusters and w[0] - clusters[-1]["x1"] < 6:
            c = clusters[-1]
            c["words"].append(w)
            c["x1"] = max(c["x1"], w[2])
        else:
            clusters.append({"x0": w[0], "x1": w[2], "words": [w]})
    heads = [TOTAL_CAT]
    for c in clusters:
        # Rotated text reads bottom to top; within one line of text, the lower word comes first.
        # Lines of a two-line head are left to right.
        ws = sorted(c["words"], key=lambda w: (round(w[0] / 4), -w[3]))
        label = _fold("".join(w[4] for w in ws))
        # The refusal head is typeset differently in every file (Kotayk's reads
        # `Հրահրաժարվել պատաս- խանել են`, with a stray syllable and a line-break hyphen), so it is
        # recognised by its verb, which no language head contains.
        if "րաժարվել" in label:
            label = "Հրաժարվելենպատասխանել"
        if label not in HY_TO_EN:
            raise SystemExit(f"{path}: unrecognised column head {label!r}; add it to HY_TO_EN")
        heads.append(HY_TO_EN[label])
    if len(set(heads)) != len(heads):
        raise SystemExit(f"{path}: a head appears twice: {heads}")

    # Data lines below the head. The refusal column is set a few points off its row in several
    # files (Vayots Dzor, Kotayk, Aragatsotn), so a row is anchored on its Total-column number
    # (left of the first language head) and every number goes to the nearest anchor.
    nums = [w for w in words if w[1] > head_bottom and _num(w[4]) is not None]
    anchors = []
    for w in sorted((w for w in nums if w[2] < clusters[0]["x0"]), key=lambda w: w[1]):
        if not anchors or w[1] - anchors[-1] >= 3:
            anchors.append(w[1])
    lines = [[] for _ in anchors]
    for w in nums:
        i = min(range(len(anchors)), key=lambda i: abs(anchors[i] - w[1]))
        # Further off is the page number; a cell lost this way fails the row's count in values().
        if abs(anchors[i] - w[1]) <= 8:
            lines[i].append(w)

    def values(line, where):
        line = sorted(line, key=lambda w: w[0])
        # Tavush separates thousands with a space, so `128 609` is two words a couple of points
        # apart; columns are at least 8 pt apart in every file.
        merged = []
        for w in line:
            if merged and re.fullmatch(r"\d{3}", w[4]) and w[0] - merged[-1][2] < 4:
                p = merged[-1]
                merged[-1] = (p[0], p[1], w[2], w[3], p[4] + w[4])
            else:
                merged.append(tuple(w[:5]))
        if len(merged) != len(heads):
            raise SystemExit(f"{path}: {where} has {len(merged)} numbers for {len(heads)} "
                             f"columns {heads}: {[w[4] for w in merged]}")
        return {h: _num(w[4]) for h, w in zip(heads, merged)}

    total = values(lines[0], "the total row")
    # The ethnicity rows of the first block, up to the repeat that follows it: men (`Տղամարդ`)
    # in most files, urban (`Քաղաք`) in Aragatsotn. Each is labelled by its stub. main() checks
    # they add up to the total row column by column.
    eth = []
    for line, y in zip(lines[1:], anchors[1:]):
        stub = " ".join(w[4] for w in sorted(words, key=lambda w: w[0])
                        if w[0] < total_head[0] - 5 and abs(w[1] - y) < 4
                        and _num(w[4]) is None)
        if _fold(stub).startswith(BLOCK_ENDS):
            break
        eth.append((stub, values(line, f"ethnicity row {stub!r}")))
    else:
        raise SystemExit(f"{path}: no men/women/urban/rural row after the ethnicity block")
    return total, eth


# Where the ethnicity rows do not add up to the total row, by column, as printed. Tavush only:
# its total row is consistent (127,857 + 634 + 118 = 128,609) and so is every row's Total, but
# the Other column's rows give 25 against the total row's 118 (the Armenian row is 15 short of
# its own Total and the Other row 78 short). The total row is what am.csv takes.
KNOWN_ROW_GAPS = {"AM-TV": {"Other": 93}}


def _rows_add_up(total, eth, where, known):
    gaps = {}
    for h in total:
        d = total[h] - sum(r[h] for _, r in eth)
        if d:
            gaps[h] = d
    good = gaps == known
    print(f"        {where}: {len(eth)} ethnicity rows add up to the total row "
          + ("in every column" if not gaps else f"except {gaps}")
          + ("" if good else " -- UNEXPECTED"))
    return good


# The ethnicity stub -> the language of that nationality, for the column-order check.
ETH_LANG = {"Հայ": "Armenian", "Եզդի": "Yezidian", "Ռուս": "Russian", "Ասորի": "Assyrian",
            "Քուրդ": "Kurdish", "Ուկրաինացի": "Ukrainian", "Հույն": "Greek", "Վրացի": "Georgian",
            "Պարսիկ": "Persian"}


def _own_language_check(eth, where):
    """Column order, checked against the ethnicity rows (the second cut of the same people).

    A head read onto the wrong column would still sum correctly, so the sums cannot catch it. The
    ethnicity rows can: for every nationality of 300 or more in the unit whose own language has
    a column, that column must be the row's largest apart from Armenian (Greeks and Kurds name
    Armenian more often than their own language in places; nobody names a third language more).
    """
    ok, n = True, 0
    for stub, row in eth:
        lang = ETH_LANG.get(_fold(stub))
        if lang is None or lang not in row or row[TOTAL_CAT] < 300:
            continue
        others = {h: v for h, v in row.items()
                  if h not in (TOTAL_CAT, lang) and (lang == "Armenian" or h != "Armenian")}
        # A tie passes: Yerevan's 603 Ukrainians name Ukrainian and Russian 251 each.
        good = all(row[lang] >= v for v in others.values())
        n += 1
        if not good:
            ok = False
            print(f"    BAD {where}: {stub} ({row[TOTAL_CAT]:,}) name {lang} {row[lang]:,} "
                  f"but {max(others, key=others.get)} {max(others.values()):,}")
    print(f"        {where}: own-language column is the largest on {n} nationality rows"
          + ("" if ok else ", NOT ALL"))
    return ok


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True
    rows, marz = [], {}
    for nid, (code, name, doc) in MARZES.items():
        path = os.path.join(RAW, f"{code}.pdf")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run with --fetch")
        cells, eth = read_total_row(path)
        marz[code] = cells
        parts = sum(v for k, v in cells.items() if k != TOTAL_CAT)
        good = parts == cells[TOTAL_CAT]
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {name:<12} total {cells[TOTAL_CAT]:>9,}  "
              f"columns sum {parts:>9,}  {len(cells) - 1} heads")
        ok &= _rows_add_up(cells, eth, name, KNOWN_ROW_GAPS.get(code, {}))
        ok &= _own_language_check(eth, name)
        for cat, n in cells.items():
            rows.append({"geo_id": code, "geo_level": "marz", "geo_name": name,
                         "source_category": cat, "count": n, "tier": "measured", "year": YEAR,
                         "source_id": SOURCE_ID,
                         "note": f"nid={nid}; doc {doc}; table 5.2-1 total row"
                                 + ("; universe total, not a language" if cat == TOTAL_CAT
                                    else "")})

    nat, nat_eth = read_total_row(os.path.join(RAW, "national_hy.pdf"))
    ok &= _rows_add_up(nat, nat_eth, "Armenia", {})
    ok &= _own_language_check(nat_eth, "Armenia")
    good = nat == NATIONAL_EN
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} Armenian national table reads as the English one "
          f"({nat[TOTAL_CAT]:,})" + ("" if good else f": {nat}"))
    for cat, n in nat.items():
        rows.append({"geo_id": "AM", "geo_level": "country", "geo_name": "Armenia",
                     "source_category": cat, "count": n, "tier": "measured", "year": YEAR,
                     "source_id": SOURCE_ID, "note": f"nid=532; doc {NATIONAL_DOC}; table 5.2-1"})

    s = sum(c[TOTAL_CAT] for c in marz.values())
    good = s == nat[TOTAL_CAT]
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} eleven marz totals sum to {s:,} (national "
          f"{nat[TOTAL_CAT]:,})")

    print("\n  each language: marz sum vs national (a floor: a marz with no column for it files"
          "\n  it under Other)")
    short = 0
    for cat in nat:
        if cat in (TOTAL_CAT, "Other"):
            continue
        got = sum(c.get(cat, 0) for c in marz.values())
        d = nat[cat] - got
        good = d >= 0
        ok &= good
        short += d
        print(f"    {'OK ' if good else 'BAD'} {cat:<18} {got:>10,} vs {nat[cat]:>10,}  {d:+,}")
    got = sum(c.get("Other", 0) for c in marz.values())
    good = got - nat["Other"] == short
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} {'Other':<18} {got:>10,} vs {nat['Other']:>10,}  "
          f"{got - nat['Other']:+,} == the shortfalls above ({short:,})")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
