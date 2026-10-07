"""Parse the 2022 census tables below upazila level.

  Union Statistics (BBS, May 2025), Table U 01: every union, grouped under its
      division, district and upazila. PDF pages 94-209 (printed 71-186).
  National Report Vol I (BBS, Nov 2023), Table P32: city corporation wards
      (PDF pages 424-432, printed 377-385), and Table P34: paurashavas by
      district (PDF pages 435-441, printed 388-394).

All three are text PDFs. Every cell is its own line (sometimes two numbers share
a line, split on whitespace); a row is a label (which may wrap over several
lines) followed by its numbers. The population taken is the table's "Total",
which in all three is male + female: the third-gender (hijra) count is only
published at zila level, exactly as for the upazila table P35 (see report.py).

Writes data/bangladesh/phc2022_level4.csv:
  table, division, district, upazila, cc, name, pop, male, female
"""
import csv
import re
from pathlib import Path

import fitz

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data/bangladesh/raw"
UNION_PDF = RAW / "phc2022_union_statistics.pdf"
REPORT_PDF = RAW / "phc2022_national_report_vol1.pdf"
OUT = HELPER / "data/bangladesh/phc2022_level4.csv"

U01_PAGES = range(94, 210)
P32_PAGES = range(424, 433)
P34_PAGES = range(435, 442)

NUMTOK = re.compile(r"^-?[\d,]+(\.\d+)?$|^-$")


def is_num_line(line):
    toks = line.split()
    return bool(toks) and all(NUMTOK.match(t) for t in toks)


def to_num(t):
    return 0 if t == "-" else float(t.replace(",", "")) if "." in t else int(t.replace(",", ""))


def valid_u01(n):
    """Table U 01's columns check each other: households = general +
    institutional + others, every age group (and all ages) total = male +
    female, and the age groups add up to all ages."""
    if n[0] != n[1] + n[2] + n[3]:
        return False
    if any(n[i] != n[i + 1] + n[i + 2] for i in range(4, 22, 3)):
        return False
    return n[4] == sum(n[i] for i in range(7, 22, 3))


def valid_wp(n):
    """P32/P34: household, total, male, female, household size, literacy."""
    return (all(isinstance(x, int) for x in n[:4]) and n[1] >= n[0]
            and abs(n[1] - n[2] - n[3]) <= max(10, n[1] // 1000) and n[5] <= 100)


def repair(tokens, ncols, valid):
    """A long number is sometimes broken over two lines ("176603" / "3" for
    1766033). Rejoin adjacent tokens until the row has ncols numbers and passes
    the table's own arithmetic; refuse if that is not exactly one way."""
    if len(tokens) == ncols:
        n = [to_num(t) for t in tokens]
        if valid(n):
            return n
    k = len(tokens) - ncols
    if k < 1 or k > 2:
        return None
    hits = []
    import itertools
    for joins in itertools.combinations(range(len(tokens) - 1), k):
        if any(b - a == 1 for a, b in zip(joins, joins[1:])):
            continue
        t, i = [], 0
        while i < len(tokens):
            if i in joins:
                t.append(tokens[i] + tokens[i + 1]); i += 2
            else:
                t.append(tokens[i]); i += 1
        try:
            n = [to_num(x) for x in t]
        except ValueError:
            continue
        if valid(n):
            hits.append(n)
    return hits[0] if len(hits) == 1 else None


KIND = re.compile(r"\b(Division|District|Upazila|Union)$")


def rows(doc, pages, ncols, header_end, valid, kinds=None):
    """Yield (label, [numbers]) for every table row. `header_end` is the last
    line of the column-number header ("23" or "7"); everything above it on a
    page is page furniture and column headings.

    A row label can be cut by a page break after its numbers have been printed
    ("Adamdighi" + numbers at the foot of one page, "Upazila" at the head of
    the next; "Sonarang" ... "Tongibari Union"). With `kinds`, a page that
    opens with label lines ending in a kind word *and then more label lines*
    before any number hands those first lines back to the previous row, when
    that row's label lacks its kind word."""
    pending = None  # (label lines, tokens, page) of the last row read

    def finish(label, toks, p):
        n = repair(toks, ncols, valid)
        if n is None:
            raise RuntimeError(f"page {p}: cannot read row {label} {toks}")
        return " ".join(label), n

    for p in pages:
        lines = [l.strip() for l in doc[p].get_text().splitlines()]
        # skip the header: up to the run of column numbers 1..n
        try:
            start = next(i for i in range(len(lines))
                         if lines[i] == header_end and lines[i - 1] == str(int(header_end) - 1))
        except StopIteration:
            raise RuntimeError(f"page {p}: no column-number header found")
        body = []
        for l in lines[start + 1:]:
            if not l:
                continue
            if l.startswith("Table"):
                break  # the next table starts on the same page
            body.append(l)

        lead = []
        for l in body:
            if is_num_line(l):
                break
            lead.append(l)
        if kinds and pending and not kinds.search(" ".join(pending[0])):
            j = next((i for i, l in enumerate(lead) if kinds.search(l)), None)
            if j is not None and j < len(lead) - 1:
                pending[0].extend(lead[:j + 1])
                body = body[j + 1:]
        if pending:
            yield finish(*pending)
            pending = None

        label, toks = [], []
        for l in body:
            if is_num_line(l):
                if label:
                    toks += l.split()
                continue  # else: a stray page number
            if toks:
                yield finish(label, toks, p)
                label, toks = [], []
            label.append(l)
        if toks:
            pending = (label, toks, p)
        # a label left at the page end with no numbers is footer text
    if pending:
        yield finish(*pending)


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def parse_u01(doc):
    out = []
    division = district = upazila = None
    for label, n in rows(doc, U01_PAGES, 22, "23", valid_u01, KIND):
        label = clean(label)
        pop, male, female = n[4], n[5], n[6]
        assert pop == male + female, (label, n)
        if label == "Union Total":
            out.append(dict(table="U01", division="", district="", upazila="", cc="",
                            name="TOTAL", pop=pop, male=male, female=female))
            continue
        m = re.match(r"^(.+?) (Division|District|Upazila|Union)$", label)
        # a few union rows drop the word "Union" (Chiknikandi)
        name, kind = m.groups() if m else (label, "Union")
        if kind == "Division":
            division, district, upazila = name, None, None
            continue
        if kind == "District":
            district, upazila = name, None
            continue
        if kind == "Upazila":
            upazila = name
            out.append(dict(table="U01", division=division, district=district, upazila=name,
                            cc="", name="", pop=pop, male=male, female=female))
            continue
        out.append(dict(table="U01", division=division, district=district, upazila=upazila,
                        cc="", name=name, pop=pop, male=male, female=female))
    return out


def parse_p32(doc):
    out, cc = [], None
    for label, n in rows(doc, P32_PAGES, 6, "9", valid_wp):
        label = clean(label)
        hh, pop, male, female = n[:4]
        if label == "Total":
            continue
        if label.endswith("City Corporation"):
            cc = label
            out.append(dict(table="P32", division="", district="", upazila="", cc=cc,
                            name="", pop=pop, male=male, female=female))
            continue
        # Dhaka North prints its cantonment as "Ward No. 98 38 (Restricted Area)"
        m = re.match(r"^Ward No\.? ?(\d+)\b.*?(\(Restricted Area\))?$", label)
        # other rows (a cantonment) keep their own label
        name = (f"Ward {int(m.group(1)):02d}" + (" (Restricted Area)" if m.group(2) else "")
                if m else label)
        out.append(dict(table="P32", division="", district="", upazila="", cc=cc,
                        name=name, pop=pop, male=male, female=female))
    return out


def parse_p34(doc):
    out, division, district = [], None, None
    for label, n in rows(doc, P34_PAGES, 6, "7", valid_wp):
        label = clean(label)
        hh, pop, male, female = n[:4]
        if label == "Total":
            out.append(dict(table="P34", division="", district="", upazila="", cc="",
                            name="TOTAL", pop=pop, male=male, female=female))
            continue
        m = re.match(r"^(.+?) (Division|District|Paurashava)$", label)
        assert m, label
        name, kind = m.groups()
        if kind == "Division":
            division = name
        elif kind == "District":
            district = name
        else:
            out.append(dict(table="P34", division=division, district=district, upazila="",
                            cc="", name=name, pop=pop, male=male, female=female))
    return out


def parse():
    u = parse_u01(fitz.open(UNION_PDF))
    rep = fitz.open(REPORT_PDF)
    w = parse_p32(rep)
    p = parse_p34(rep)

    # Each table must add up to its own printed totals.
    un = [r for r in u if r["name"] not in ("", "TOTAL")]
    tot = [r for r in u if r["name"] == "TOTAL"][0]["pop"]
    assert sum(r["pop"] for r in un) == tot, (sum(r["pop"] for r in un), tot)
    for up in {(r["district"], r["upazila"]) for r in un}:
        head = [r for r in u if r["name"] == "" and (r["district"], r["upazila"]) == up]
        assert len(head) == 1, up
        s = sum(r["pop"] for r in un if (r["district"], r["upazila"]) == up)
        assert s == head[0]["pop"], (up, s, head[0]["pop"])
    for cc in {r["cc"] for r in w}:
        head = [r for r in w if r["cc"] == cc and r["name"] == ""][0]["pop"]
        assert sum(r["pop"] for r in w if r["cc"] == cc and r["name"]) == head, cc
    ptot = [r for r in p if r["name"] == "TOTAL"][0]["pop"]
    ps = [r for r in p if r["name"] != "TOTAL"]
    assert sum(r["pop"] for r in ps) == ptot, (sum(r["pop"] for r in ps), ptot)
    return u + w + p


def main():
    recs = parse()
    with OUT.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0]))
        wr.writeheader()
        wr.writerows(recs)
    from collections import Counter
    c = Counter((r["table"], r["name"] == "") for r in recs)
    print(f"{len(recs)} rows -> {OUT}: {dict(c)}")


if __name__ == "__main__":
    main()
