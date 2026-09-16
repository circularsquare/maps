"""Mongolia: Ulaanbaatar's nine düüregs, from the capital's own 2020 census volume.

Called by sources/mn.py, which draws the capital from these rows instead of its city-wide ones.
sources/mn.md §11 has the record.

The volume (`Ulaanbaatar_XAOCT_Negdsen_dun.pdf`, 300 pages) prints religion city-wide in its
tables 3.5 to 3.7, and by düüreg only as two CHARTS:

  * ЗУРАГ 3.5 (PDF p.54): the share of people aged 15 and over who practise a religion, one bar
    per düüreg;
  * ЗУРАГ 3.6 (PDF p.55): the religious split Будда / Христ / Ислам / Бөө / Бусад, one stacked
    bar per düüreg.

**THE CHARTS ARE VECTOR DRAWINGS, SO THEY ARE MEASURED RATHER THAN READ.** Every bar and every
segment is a filled rectangle in `page.get_drawings()`, one fill per religion, and its width is
the value. The value labels and the düüreg names are drawn as glyph OUTLINES, not text, so
nothing on either chart can be extracted as a string. So:

  * the numbers come from the rectangle widths, at one scale per chart fitted on the printed
    labels (least squares through the origin). Every label was read by eye off a 300 dpi
    render and is asserted to agree with its own segment's width to within 0.05 points, which
    is the transcription check. The unlabelled segments (most Ислам and Бусад cells) are the
    widths alone.
  * The scale is FIXED, NOT PER ROW. Baganuur's bar is 0.09% wider than the other eight, because
    its five printed values sum to 100.1; normalising each row to 100 read its Buddhist share as
    85.4 against a printed 85.5.
  * The row names are transcribed, and checked by their glyphs: a letter drawn twice is the
    same outline twice, so every glyph's path signature (the sequence of line and curve
    operators) must map to one letter across all nine names on both charts and the legend. Two
    düüregs swapped in either transcription puts one outline under two letters.
  * Each printed label must sit inside the segment of the religion it is credited to. That is
    what settles Nalaikh.

**NALAIKH'S 13.6% IS ISLAM, AND THE VOLUME'S OWN PROSE SAYS CHRISTIAN.** PDF p.20: *"Налайх
дүүргийн шашинтнуудын 13.6 хувь нь христийн шашин шүтдэг ... Энэ нь тус дүүрэгт Казахууд
олноороо амьдарч байгаатай холбоотой"*, 13.6% of Nalaikh's religious are Christian, because
many Kazakhs live there. The chart draws that segment in the Islam fill, puts the `13.6` label
inside it, and draws Christian at 4.3; the sentence's own reason fits Islam; and
sources/mn.py's city-wide check fails if the 13.6 is moved to Christian (0.35 points on the
city's Islam share against a 0.15 tolerance). Read the figure, not the sentence.

**THE DENOMINATOR IS APPENDIX TABLE 1.1 OF THE SAME VOLUME** (PDF p.211): resident population by
düüreg and five-year age group, so the 15-and-over population is the total less the three child
bands. It is printed in two blocks on one page (Бүгд and 0-4 to 30-34, then 35-39 to 70+), so
every düüreg appears TWICE, the national report's appendix trap (sources/mn.md §8). The blocks
are told apart by the identity, not by position: the total must equal the sum of all fifteen
bands. Tables 1.2 (men) and 1.3 (women) are read the same way and must add to table 1.1 in every
cell, and the nine düüregs' 15-and-over must equal the national report's own Ulaanbaatar figure.

**THE JOIN IS BY NAME INSIDE ONE CITY, WITH THE CODE ORDER AS WITNESS.** COD-AB's `adm2_name1`
for the nine children of MN11 are exactly the Cyrillic names both charts and table 1.1 print,
and nine names inside one parent leave no twin to pick. The witness: the volume prints its
per-düüreg appendix tables 1.4 to 1.12 in Cyrillic alphabetical order, which is COD's pcode
order too (MN1101 Багануур to MN1125 Чингэлтэй); `check_join` asserts both.

Usage: imported. `python sources/mn_ub.py` prints what it reads.
"""

import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PDF = os.path.join(ROOT, "data", "raw", "mn", "aimags", "Ulaanbaatar_XAOCT_Negdsen_dun.pdf")
SOUMS = os.path.join(ROOT, "data", "geo", "mn", "mn_soums.gpkg")

CITY = "MN11"

# COD-AB adm2_pcode -> (the name the volume prints, a Latin name for geo_name).
DUUREG = {
    "MN1101": ("Багануур", "Baganuur"),
    "MN1104": ("Багахангай", "Bagakhangai"),
    "MN1107": ("Баянгол", "Bayangol"),
    "MN1110": ("Баянзүрх", "Bayanzurkh"),
    "MN1113": ("Налайх", "Nalaikh"),
    "MN1116": ("Сонгинохайрхан", "Songinokhairkhan"),
    "MN1119": ("Сүхбаатар", "Sukhbaatar"),
    "MN1122": ("Хан-Уул", "Khan-Uul"),
    "MN1125": ("Чингэлтэй", "Chingeltei"),
}
NAMES = {mn: code for code, (mn, _) in DUUREG.items()}

CATEGORIES = ["Будда", "Христ", "Ислам", "Бөө", "Бусад"]

# ---- ЗУРАГ 3.5, rows top to bottom as printed, labels read by eye off a 300 dpi render.
FIG35_ROWS = [
    ("Чингэлтэй", 58.2), ("Баянгол", 55.4), ("Налайх", 55.3), ("Сонгинохайрхан", 55.1),
    ("Хан-Уул", 53.2), ("Сүхбаатар", 52.7), ("Багануур", 51.3), ("Баянзүрх", 50.5),
    ("Багахангай", 47.1),
]
# Songinokhairkhan's label runs under the city-average callout box, so its last digit is
# partly hidden: it is checked against the measure but not used to fit the scale.
FIG35_PARTLY_HIDDEN = {"Сонгинохайрхан"}
FIG35_BAR = "#be96aa"
FIG35_NAME_FILL = "#77787b"

# ---- ЗУРАГ 3.6, rows top to bottom as printed; only the segments that carry a printed label.
FIG36_ROWS = [
    ("Баянгол", {"Будда": 91.6, "Христ": 2.9, "Бөө": 3.8}),
    ("Баянзүрх", {"Будда": 88.0, "Христ": 4.0, "Бөө": 5.5}),
    ("Багануур", {"Будда": 85.5, "Христ": 6.0, "Бөө": 7.8}),
    ("Багахангай", {"Будда": 95.5, "Бөө": 2.6}),
    ("Налайх", {"Будда": 77.7, "Христ": 4.3, "Ислам": 13.6, "Бөө": 4.0}),
    ("Чингэлтэй", {"Будда": 89.3, "Христ": 3.2, "Бөө": 6.9}),
    ("Сонгинохайрхан", {"Будда": 90.3, "Христ": 2.3, "Бөө": 5.3}),
    ("Сүхбаатар", {"Будда": 89.0, "Христ": 3.6, "Бөө": 6.4}),
    ("Хан-Уул", {"Будда": 89.1, "Христ": 3.3, "Бөө": 5.4}),
]
# The legend, left to right as printed. Its swatch fills are read off the page and its label
# glyphs go through the same outline check as the row names, so a colour credited to the wrong
# religion here fails there.
FIG36_LEGEND = ["Будда", "Христ", "Ислам", "Бөө", "Бусад"]
FIG36_TEXT_FILLS = {"#000000", "#ffffff"}

LABEL_TOL = 0.05      # a printed one-decimal label against its own segment's width
ROW_SUM_TOL = 0.2     # a düüreg's five measured shares against 100


def _hex(fill):
    return None if fill is None else "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in fill)


def _caption_y(page, number, must):
    """Top of the caption `ЗУРАГ <number>` whose next dozen words include `must`, or None."""
    ws = page.get_text("words")
    for i, w in enumerate(ws[:-1]):
        if w[4] == "ЗУРАГ" and ws[i + 1][4] == number + ".":
            if any(must in x[4] for x in ws[i:i + 14]):
                return w[1]
    return None


def _find_chart(doc, number, must):
    # From sources/mn.py's FRONT_MATTER on: the list of figures repeats every caption.
    sys.path.insert(0, HERE)
    import mn as M
    hits = [(i, y) for i in range(M.FRONT_MATTER, doc.page_count)
            if (y := _caption_y(doc[i], number, must)) is not None]
    if len(hits) != 1:
        raise SystemExit(f"{os.path.basename(PDF)}: ЗУРАГ {number} (with {must!r}) found on "
                         f"{len(hits)} pages, expected one")
    return hits[0]


class Glyph:
    __slots__ = ("x0", "x1", "yc", "sig", "dot")

    def __init__(self, d):
        r = d["rect"]
        self.x0, self.x1, self.yc = r.x0, r.x1, (r.y0 + r.y1) / 2
        self.sig = "".join(it[0] for it in d["items"])
        self.dot = self.sig == "re" and r.width < 1.3 and r.height < 1.3


def _glyphs(drawings, fills, y0, y1):
    out = []
    for d in drawings:
        r = d["rect"]
        if (_hex(d.get("fill")) in fills and r.width < 12 and r.height < 12
                and y0 <= r.y0 and r.y1 <= y1):
            out.append(Glyph(d))
    return out


def _by_row(glyphs, centres):
    rows = [[] for _ in centres]
    for g in glyphs:
        k = min(range(len(centres)), key=lambda j: abs(centres[j] - g.yc))
        rows[k].append(g)
    return [sorted(r, key=lambda g: g.x0) for r in rows]


def _decode(rows, names, sig2ch, where):
    """Every glyph outline must stand for one letter, across every name it appears in."""
    for glyphs, name in zip(rows, names):
        if len(glyphs) != len(name):
            raise SystemExit(f"{where}: the row transcribed as {name!r} draws {len(glyphs)} "
                             f"glyphs for {len(name)} letters")
        for g, ch in zip(glyphs, name):
            # Case-folded, because an upper- and a lower-case letter can share an outline
            # shape (И and и are both ten straight lines); a swap needs a different letter.
            prev = sig2ch.setdefault(g.sig, (ch.lower(), name))
            if prev[0] != ch.lower():
                raise SystemExit(f"{where}: one glyph outline stands for {prev[0]!r} in "
                                 f"{prev[1]!r} and for {ch!r} in {name!r}; two rows are "
                                 "transcribed in the wrong order")


def _fit(pairs):
    """Least-squares scale through the origin: value = width * k."""
    return sum(w * v for w, v in pairs) / sum(w * w for w, _ in pairs)


def read_fig35(doc, sig2ch):
    """{düüreg: % of the 15+ population who practise a religion}, measured."""
    i, cap = _find_chart(doc, "3.5", "ДҮҮРГЭЭР")
    page = doc[i]
    foot = page.rect.height - 50
    drawings = page.get_drawings()
    bars = sorted((it[1] for d in drawings if _hex(d.get("fill")) == FIG35_BAR
                   for it in d["items"] if it[0] == "re" and it[1].y0 > cap
                   and it[1].height > 10), key=lambda r: r.y0)
    if len(bars) != len(FIG35_ROWS):
        raise SystemExit(f"ЗУРАГ 3.5 (PDF p.{i + 1}): {len(bars)} bars, expected "
                         f"{len(FIG35_ROWS)}")
    origin = bars[0].x0
    if any(abs(b.x0 - origin) > 0.01 for b in bars):
        raise SystemExit("ЗУРАГ 3.5: the bars do not share one origin")

    names = [n for n, _ in FIG35_ROWS]
    glyphs = [g for g in _glyphs(drawings, {FIG35_NAME_FILL}, cap, foot) if g.x1 < origin]
    _decode(_by_row(glyphs, [(b.y0 + b.y1) / 2 for b in bars]), names, sig2ch, "ЗУРАГ 3.5")

    k = _fit([(b.width, v) for b, (n, v) in zip(bars, FIG35_ROWS)
              if n not in FIG35_PARTLY_HIDDEN])
    out, worst = {}, 0.0
    for b, (name, label) in zip(bars, FIG35_ROWS):
        v = b.width * k
        worst = max(worst, abs(v - label))
        if abs(v - label) > LABEL_TOL:
            raise SystemExit(f"ЗУРАГ 3.5: {name} is labelled {label} and its bar measures "
                             f"{v:.3f}")
        out[name] = v
    return out, {"page": i + 1, "worst": worst, "full": 100.0 / k}


def read_fig36(doc, sig2ch):
    """{düüreg: {religion: % of its religious}}, measured."""
    i, cap = _find_chart(doc, "3.6", "ШАШНЫ")
    page = doc[i]
    foot = page.rect.height - 50
    drawings = page.get_drawings()

    # The legend: square swatches, read left to right, their labels between them.
    swatches = sorted((d for d in drawings if d["rect"].y0 > cap and d["items"]
                       and all(it[0] == "re" for it in d["items"]) and len(d["items"]) == 1
                       and 3 < d["rect"].width < 8 and abs(d["rect"].width
                                                           - d["rect"].height) < 0.2),
                      key=lambda d: d["rect"].x0)
    if len(swatches) != len(FIG36_LEGEND):
        raise SystemExit(f"ЗУРАГ 3.6 (PDF p.{i + 1}): {len(swatches)} legend swatches, "
                         f"expected {len(FIG36_LEGEND)}")
    fill_of = {_hex(s["fill"]): cat for s, cat in zip(swatches, FIG36_LEGEND)}
    ly0 = min(s["rect"].y0 for s in swatches) - 4
    ly1 = max(s["rect"].y1 for s in swatches) + 4
    lglyphs = _glyphs(drawings, {"#000000"}, ly0, ly1)
    edges = [s["rect"].x1 for s in swatches] + [page.rect.width]
    legend_rows = [sorted((g for g in lglyphs if edges[j] < g.x0 < edges[j + 1] - 5),
                          key=lambda g: g.x0) for j in range(len(swatches))]
    _decode(legend_rows, FIG36_LEGEND, sig2ch, "ЗУРАГ 3.6 legend")

    # The segments.
    segs = []
    for d in drawings:
        cat = fill_of.get(_hex(d.get("fill")))
        if cat is None:
            continue
        for it in d["items"]:
            if it[0] == "re" and it[1].y0 > cap and it[1].y1 < ly0 and it[1].height > 10:
                segs.append((cat, it[1]))
    centres = sorted({round((r.y0 + r.y1) / 2, 1) for _, r in segs})
    if len(centres) != len(FIG36_ROWS):
        raise SystemExit(f"ЗУРАГ 3.6: {len(centres)} bars, expected {len(FIG36_ROWS)}")
    rows = [sorted(((r.x0, r.x1, c) for c, r in segs
                    if abs((r.y0 + r.y1) / 2 - yc) < 1), key=lambda s: s[0]) for yc in centres]
    origin = rows[0][0][0]
    if any(abs(r[0][0] - origin) > 0.01 or r[0][2] != "Будда" for r in rows):
        raise SystemExit("ЗУРАГ 3.6: the bars do not all start at one origin with Будда")
    for r in rows:
        if len({c for _, _, c in r}) != len(r):
            raise SystemExit("ЗУРАГ 3.6: a religion drawn twice in one bar")

    names = [n for n, _ in FIG36_ROWS]
    band0, band1 = cap, ly0
    nglyphs = [g for g in _glyphs(drawings, {"#000000"}, band0, band1) if g.x1 < origin]
    _decode(_by_row(nglyphs, centres), names, sig2ch, "ЗУРАГ 3.6")

    # The printed labels: split each bar's glyphs after the digit that follows a decimal point.
    vglyphs = [g for g in _glyphs(drawings, FIG36_TEXT_FILLS, band0, band1)
               if g.x0 >= origin - 0.5]
    pairs = []
    for (name, labels), segrow, glyphs in zip(FIG36_ROWS, rows, _by_row(vglyphs, centres)):
        found, cur, after_dot = [], [], False
        for g in glyphs:
            cur.append(g)
            if after_dot:
                found.append(cur)
                cur, after_dot = [], False
            elif g.dot:
                after_dot = True
        if cur:
            raise SystemExit(f"ЗУРАГ 3.6: {name}'s labels do not split on decimal points")
        if len(found) != len(labels):
            raise SystemExit(f"ЗУРАГ 3.6: {name} prints {len(found)} labels and "
                             f"{len(labels)} were transcribed")
        where = {c: (a, b) for a, b, c in segrow}
        ordered = sorted(labels, key=lambda c: where[c][0])
        for cat, lab in zip(ordered, found):
            centre = (lab[0].x0 + lab[-1].x1) / 2
            a, b = where[cat]
            if not a - 0.5 <= centre <= b + 0.5:
                raise SystemExit(f"ЗУРАГ 3.6: {name}'s label {labels[cat]} is transcribed as "
                                 f"{cat} but is printed at x={centre:.1f}, outside that "
                                 f"segment ({a:.1f}-{b:.1f})")
            pairs.append((b - a, labels[cat]))
    k = _fit(pairs)

    out, worst = {}, 0.0
    for (name, labels), segrow in zip(FIG36_ROWS, rows):
        shares = {c: 0.0 for c in CATEGORIES}
        for a, b, c in segrow:
            shares[c] = (b - a) * k
        for c, lab in labels.items():
            worst = max(worst, abs(shares[c] - lab))
            if abs(shares[c] - lab) > LABEL_TOL:
                raise SystemExit(f"ЗУРАГ 3.6: {name} {c} is labelled {lab} and its segment "
                                 f"measures {shares[c]:.3f}")
        if abs(sum(shares.values()) - 100.0) > ROW_SUM_TOL:
            raise SystemExit(f"ЗУРАГ 3.6: {name}'s five shares measure "
                             f"{sum(shares.values()):.2f}")
        out[name] = shares
    return out, {"page": i + 1, "worst": worst, "full": 100.0 / k}


# ---- the denominator
T11 = re.compile(r"ХҮСНЭГТ\s+(1\.[123])\.")
T14 = "ХҮСНЭГТ 1.4."
BLOCK = 8            # values per printed block: Бүгд + 7 bands, then 8 bands
LEAD = re.compile(r"^\d{1,4}$")
GROUP = re.compile(r"^\d{3}$")


def _table_rows(page):
    """[(label, [value, ...], words)] per printed line of appendix tables 1.1 to 1.3.

    **NOT sources/mn.py's `_rows`**, which reads a number only as groups of one to three digits
    (that is what keeps the `2010`/`2020` year labels out of its values). Table 1.2 prints one
    cell with no thousands space, Songinokhairkhan's men aged 45-49 on PDF p.212 as `9145`, and
    that reader files it in the label and shifts the rest of the row a column left. Here a
    four-digit token is a number; the identities in `read_population` are what check it (the
    cell is 18,789 in table 1.1 less 9,644 in table 1.3).
    """
    lines = {}
    for w in page.get_text("words"):
        lines.setdefault(round(w[1] / 3.0), []).append(w)
    out = []
    for y in sorted(lines):
        ws = sorted(lines[y], key=lambda w: w[0])
        label, vals, i = [], [], 0
        while i < len(ws):
            t = ws[i][4]
            if LEAD.match(t):
                text, x1 = t, ws[i][2]
                while (i + 1 < len(ws) and GROUP.match(ws[i + 1][4])
                       and ws[i + 1][0] - x1 < 6.0):
                    text += ws[i + 1][4]
                    x1 = ws[i + 1][2]
                    i += 1
                vals.append((float(text), ws[i][0]))
            else:
                label.append(t)
            i += 1
        out.append((" ".join(label).strip(), vals, ws))
    return out


def read_population(doc):
    """{düüreg: (resident population, aged 15+)} from appendix table 1.1, checked."""
    sys.path.insert(0, HERE)
    import mn as M

    got = {t: {} for t in ("1.1", "1.2", "1.3")}
    cur = None
    for i in range(M.FRONT_MATTER, doc.page_count):
        flat = " ".join(doc[i].get_text().split())
        start = flat.find("ХҮСНЭГТ 1.1. УЛААНБААТАР ХОТЫН СУУРИН")
        if cur is None and start < 0:
            continue
        for label, vals, _ in _table_rows(doc[i]):
            m = T11.search(label)
            if m:
                cur = m.group(1)
                continue
            if T14 in label:
                cur = "stop"
            if cur in got and (label in NAMES or label == "БҮГД") and len(vals) == BLOCK:
                got[cur].setdefault(label, []).append([int(v) for v, _ in vals])
        if T14 in flat:
            break

    tables = {}
    for t, rows in got.items():
        want = set(NAMES) | {"БҮГД"}
        if set(rows) != want:
            raise SystemExit(f"appendix table {t}: rows {sorted(set(rows) ^ want)} missing "
                             "or unexpected")
        tab = {}
        for name, blocks in rows.items():
            if len(blocks) != 2:
                raise SystemExit(f"appendix table {t}: {name} printed {len(blocks)} times, "
                                 "expected once per block")
            # Which block is which is decided by the identity, not by position.
            for first, second in (blocks, blocks[::-1]):
                total, bands = first[0], first[1:] + second
                if len(bands) == 15 and sum(bands) == total:
                    tab[name] = (total, bands)
                    break
            else:
                raise SystemExit(f"appendix table {t}: {name}'s fifteen age bands do not sum "
                                 "to its total in either block order")
        for j in range(16):
            col = [tab[n][0] if j == 0 else tab[n][1][j - 1] for n in NAMES]
            city = tab["БҮГД"][0] if j == 0 else tab["БҮГД"][1][j - 1]
            if sum(col) != city:
                raise SystemExit(f"appendix table {t}: column {j} of the nine düüregs sums "
                                 f"to {sum(col):,} against the city's {city:,}")
        tables[t] = tab
    for n in tables["1.1"]:
        both, men, women = tables["1.1"][n], tables["1.2"][n], tables["1.3"][n]
        if men[0] + women[0] != both[0] or any(
                a + b != c for a, b, c in zip(men[1], women[1], both[1])):
            raise SystemExit(f"appendix tables 1.2 + 1.3 do not add to 1.1 for {n}")
    return {n: (tot, tot - sum(bands[:3])) for n, (tot, bands) in tables["1.1"].items()
            if n in NAMES}


def check_join(doc):
    """COD's nine düüreg names and pcode order against the volume's own table order."""
    import geopandas as gpd

    g = gpd.read_file(SOUMS)
    g = g[g["aimag"] == CITY]
    cod = dict(zip(g["unit"], g["name_mn"]))
    want = {c: mn for c, (mn, _) in DUUREG.items()}
    if cod != want:
        raise SystemExit(f"COD-AB's children of {CITY} are {cod}, expected {want}")
    order = []
    for i in range(doc.page_count):
        flat = " ".join(doc[i].get_text().split())
        for m in re.finditer(r"ХҮСНЭГТ 1\.(\d+)\. УЛААНБААТАР ХОТЫН (\S+) ДҮҮРГИЙН СУУРИН",
                             flat):
            order.append((int(m.group(1)), m.group(2)))
    printed = [n for _, n in sorted(order)]
    by_code = [DUUREG[c][0].upper() for c in sorted(DUUREG)]
    if printed != by_code:
        raise SystemExit(f"the volume's per-düüreg tables run {printed}, COD's pcodes run "
                         f"{by_code}")
    return len(printed)


def read():
    """-> (populations, fig 3.5 shares, fig 3.6 shares, a report dict)."""
    import fitz

    if not os.path.exists(PDF):
        raise SystemExit(f"missing {PDF} -- run sources/mn.py --fetch")
    doc = fitz.open(PDF)
    sig2ch = {}
    rel, r35 = read_fig35(doc, sig2ch)
    types, r36 = read_fig36(doc, sig2ch)
    pop = read_population(doc)
    n_join = check_join(doc)
    return pop, rel, types, {"fig35": r35, "fig36": r36, "outlines": len(sig2ch),
                             "join_tables": n_join}


if __name__ == "__main__":
    pop, rel, types, rep = read()
    print(rep)
    for n in NAMES:
        t = types[n]
        print(f"{n:<16} {pop[n][0]:>9,} {pop[n][1]:>9,}  rel {rel[n]:6.2f}  "
              + "  ".join(f"{c} {t[c]:6.2f}" for c in CATEGORIES))
