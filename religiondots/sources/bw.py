"""Botswana — religion by named locality, 2011 census, out of eighteen district booklets.

Writes data/normalized/bw.csv.

**THE SOURCE IS A BOOKLET SERIES AND NOT A REPORT, WHICH IS WHY THIS COUNTRY LOOKED SHUT.**
Statistics Botswana's 2022 census publishes religion crossed with sex, marital status,
locality type and employment, and with no geography at all; `sources.md` §11p read that in
2026-09-06 and recorded Botswana as *"religion x LANGUAGE only"*. The 2011 census is the
opposite: its national volumes are also geography-free, but the office issued a separate
*Population and Housing Census 2011 Selected Indicators* booklet for each census district,
and **every one of them prints a religion table by named village**. Eighteen booklets, one
table each, on `statsbots.org.bw/sites/default/files/publications/`.

So the country is drawn from an eleven-year-old census at ~500 localities rather than from
the current one at nothing. See `sources/bw.md` §1 for why that trade was taken.

**TWO OF THE TWENTY-EIGHT DISTRICTS HAVE NO BOOKLET.** Central Boteti and Central Bobonong
are the gaps at 6.1 and 6.3 in the series numbering and were never put online; a filename
sweep and the Wayback CDX listing for the whole host both come back with the same eighteen
files and no more. Those two districts are 129,312 people in 2011 and are NOT DRAWN. They
are named in `gap=`.

**THE UNIVERSE IS AGE 12 AND OVER**, which is Peru's shape. The under-twelves were never
asked; they are in `gap=` rather than drawn as a §3.5 undercount.

**WHAT THE PARSE ASSERTS, BECAUSE THE BOOKLETS ARE NOT UNIFORM.** They were typeset
separately and disagree in three ways that all fail silently:

  * **Some tables carry a tenth column, a row TOTAL, and it comes FIRST.** The test is
    arithmetic and not typographic: in a ten-wide row `v[0] == sum(v[1:])`. A parser that
    assumed nine columns reads the total as `Christian` and every category shifts by one.
  * **The captions lie.** Central Serowe/Palapye captions BOTH halves of its pair `(%)`
    while 12a holds the counts; the Cities and Towns booklet captions its religion counts
    *"Number of people by marital status"*. So counts are told from percentages by looking
    at the VALUES, never at the caption.
  * **`Other` is a row label as well as a column name.** Every booklet ends its list with an
    `Other` residual for the localities it does not name, immediately above `Total`. A
    parser that skips header words anywhere in the table silently drops it, and with it the
    people who live in every village too small to print.

Every table is required to reconcile twice: the nine categories sum to the printed row total
where one exists, and the body rows sum to the printed `Total` row column by column. Nothing
is written if either fails.

Usage:
    python sources/bw.py --fetch     the eighteen booklets, ~20 MB
    python sources/bw.py             rebuild from data/raw/bw/mono/
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
RAW = os.path.join(ROOT, "data", "raw", "bw", "mono")
OUT = os.path.join(ROOT, "data", "normalized", "bw.csv")

BASE = "https://www.statsbots.org.bw/sites/default/files/publications/"

# booklet -> (filename, ADM2 p-code it covers, printed district name)
# Chobe's booklet also carries the Ngamiland Delta and Ghanzi's also carries the CKGR; both
# arrive as ordinary locality ROWS and are re-homed by name in ADM2_OVERRIDE below.
BOOKLETS = {
    "towns": ("Cities%20%20and%20%20Towns%20Population%20and%20Housing%20Census%202011%20%20"
              "Selected%20Indicators.pdf", None, "Cities and Towns"),
    "ngwaketse": ("Ngwaketse%20District.pdf", "BW0801", "Southern"),
    "barolong": ("Barolong%20District.pdf", "BW0802", "Barolong"),
    "ngwaketsewest": ("Ngwaketse%20West.pdf", "BW0803", "Ngwaketse West"),
    "southeast": ("South%20East%20District-Population%20and%20Housing%20Census%202011%20"
                  "Selected%20Indicators.pdf", "BW0901", "South East"),
    "kwenengeast": ("Kweneng%20East%20Sub%20District.pdf", "BW1001", "Kweneng East"),
    "kwenengwest": ("Kweneng%20West.pdf", "BW1002", "Kweneng West"),
    "kgatleng": ("Kgatleng%20District.pdf", "BW1101", "Kgatleng"),
    "serowepalapye": ("Serowe_Palapye.pdf", "BW1201", "Central Serowe/Palapye"),
    "mahalapye": ("Central%20Mahalapye%20District%20Selected%20indicators_0.pdf",
                  "BW1202", "Central Mahalapye"),
    "tutume": ("Central%20Tutume.pdf", "BW1205", "Central Tutume"),
    "northeast": ("North%20East%20District.pdf", "BW1301", "North East"),
    "ngamieast": ("Ngami%20East%20District.pdf", "BW1401", "Ngamiland East"),
    "ngamiwest": ("Ngami%20West.pdf", "BW1402", "Ngamiland West"),
    "chobe": ("Chobe%20District.pdf", "BW1501", "Chobe"),
    "ghanzi": ("Ghanzi%20District.pdf", "BW1601", "Ghanzi"),
    "kgalagadisouth": ("Kgalagadi%20South%20District.pdf", "BW1701", "Kgalagadi South"),
    "kgalagadinorth": ("Kgalagadi%20North%20District.pdf", "BW1702", "Kgalagadi North"),
}

# The seven cities and towns are their own census districts and their own ADM2 polygons, so
# the Cities and Towns booklet's seven rows are re-homed one by one.
TOWN_ADM2 = {
    "gaborone": "BW0101", "francistown": "BW0201", "lobatse": "BW0301",
    "selebiphikwe": "BW0401", "selibephikwe": "BW0401", "orapa": "BW0501",
    "jwaneng": "BW0601", "sowatown": "BW0701", "sowa": "BW0701",
}
# Rows that sit in a booklet for a district other than the booklet's own. The Ghanzi booklet
# prints the Central Kgalagadi Game Reserve as one of its localities, and the Chobe booklet
# carries the whole of the Ngamiland Delta as seven more; both are census districts in their
# own right and have their own ADM2 polygon, so they are re-homed here. The Delta list is
# the complement of the nine Chobe villages the 2022 locality report names for district 72,
# so it is a partition of the booklet and not a guess.
ADM2_OVERRIDE = {("ghanzi", "ckgr"): "BW1602"}
ADM2_OVERRIDE.update({("chobe", n): "BW1403" for n in (
    "daonara", "ditshiping", "jao", "katamaga", "morutsha", "xaxaba",
    "deltanoaffiliation")})

# A row that is a district's unnamed remainder rather than a place. Every booklet has one and
# three of them spell it differently; `Localities with no Affiliation` is the census's own
# phrase for people enumerated at cattle posts, lands and freehold farms.
RESIDUAL_ROW = re.compile(r"^(other|other\s*localit(y|ies)|"
                          r"(.*\s)?no\s*affiliation)$", re.I)

CATS = ["Christian", "Muslim", "Bahai", "Hindu", "Badimo", "No religion",
        "Rastafarian", "Other", "Not stated"]

# UNSD table 28, forwarded by Statistics Botswana: the 2011 national figures this file is
# checked against.  `Unknown` there is the under-twelves plus non-response together.
NATIONAL_2011 = {"Christian": 1_171_537, "No religion": 225_416, "Badimo": 60_613,
                 "Muslim": 10_941, "Hindu": 3_729, "Bahai": 2_074,
                 "Rastafarian": 2_030, "Other": 1_461}
CENSUS_POPULATION_2011 = 2_024_904
# the two districts with no booklet, 2011 census population
NOT_COVERED = {"BW1203": ("Central Bobonong", 71_936), "BW1204": ("Central Boteti", 57_376)}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for key, (fn, _, _) in BOOKLETS.items():
        dest = os.path.join(RAW, key + ".pdf")
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            print(f"  have {key}")
            continue
        r = requests.get(BASE + fn, timeout=600, headers={"User-Agent": "Mozilla/5.0"})
        print(f"  {key:16s} {r.status_code} {len(r.content):>9,}")
        if r.status_code != 200 or r.content[:4] != b"%PDF":
            raise SystemExit(f"{key}: not a PDF -- {BASE + fn}")
        with open(dest, "wb") as fh:
            fh.write(r.content)


def clean(s):
    s = unicodedata.normalize("NFKC", str(s))
    return " ".join("".join(c for c in s
                            if not unicodedata.category(c).startswith("C")).split())


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# ---------------------------------------------------------------------------------------
# THE PARSE
# ---------------------------------------------------------------------------------------
# Captions are never used to find or label a table. They are wrong too often: Central
# Serowe/Palapye captions both halves of its pair `(%)`, the Cities and Towns booklet calls
# its religion counts "Number of people by marital status", Ngwaketse West prints every
# caption on the page in a block at the end of the reading order, and Chobe writes
# "Table: 7A". The anchor is the literal column header `Christian`.

DASH = "‐‑‒–—―−-"
THOUSANDS = "’‘'` , "

# One entry per column the header can name. Seeing a slot TWICE means the header is over,
# which is what rescues the residual row literally called `Other` in the two booklets where
# it is the first body row and sits directly against the `Other` column heading.
SLOTS = [
    (re.compile(r"^christian$", re.I), "Christian"),
    (re.compile(r"^muslim$", re.I), "Muslim"),
    (re.compile(r"^baha.?i$", re.I), "Bahai"),
    (re.compile(r"^hindu$", re.I), "Hindu"),
    (re.compile(r"^badimo$", re.I), "Badimo"),
    (re.compile(r"^(no\s*religion|none)$", re.I), "No religion"),
    (re.compile(r"^rastafarian(ism)?$", re.I), "Rastafarian"),
    (re.compile(r"^other(\s*religion)?\s*(\(nec\))?$", re.I), "Other"),
    (re.compile(r"^(\(nec\)\s*)?not\s*stated$", re.I), "Not stated"),
]
# Header debris that never identifies a column on its own: spanner labels, and the pieces of
# a header word the typesetter hyphenated across two lines.
CONT = re.compile(
    r"^(religion|religions|religious\s*affiliation|villages?|locality|localities|"
    r"name|type|total|household|households|cities\s*and\s*towns|city\s*/?\s*town|"
    r"no|not|stated|nec|\(nec\)|religion\s*\(nec\)|religion\(nec\)|"
    r"mus-?|lim|rasta-?|farian|baha-?|’?i|and|by|number|percent(age)?|%)$", re.I)

# The two booklets whose printed Total row does not equal the sum of its own body rows.
# Both are the office's arithmetic and not the parse; see sources/bw.md §3.
# `exclude` is a row the printed Total demonstrably does not include; `max_off` is what is
# left over after that, in people, and is deliberately tight. Widening either of these is
# the wrong repair: re-read the booklet.
KNOWN_SLIPS = {
    "ghanzi": dict(
        exclude="ckgr", max_off=3,
        why="the printed Total covers 18 of the 19 rows. The Central Kgalagadi Game "
            "Reserve is printed as a locality of the Ghanzi booklet but is its own census "
            "district and is left out of the district total; with it removed, seven of the "
            "nine columns agree to the person and Other and Not stated are one and two out."),
    "tutume": dict(
        exclude=None, max_off=32,
        why="every one of the 42 rows is internally consistent and the row totals sum "
            "exactly, but the Total row's own category cells are 16 out on Christian and 8 "
            "each the other way on Badimo and No religion, netting to zero. The office "
            "evidently built the total row from a separate pass over the microdata."),
}


def parse_value(t):
    """(kind, value) for a table cell, else None. A bare dash is a printed zero."""
    if t and all(c in DASH for c in t):
        return ("int", 0)
    u = t
    for c in THOUSANDS:
        u = u.replace(c, "")
    u = "".join(c if c not in DASH else "-" for c in u)
    if re.fullmatch(r"\d+", u):
        return ("int", int(u))
    if re.fullmatch(r"\d+\.\d+", u):
        return ("dec", float(u))
    return None


def slot_of(line):
    for pat, name in SLOTS:
        if pat.match(line):
            return name
    if re.search(r"not\s*stated", line, re.I) and len(line) < 24:
        return "Not stated"
    return None


def header_span(lines, anchor):
    """Consume the column header from the `Christian` line -> (body start, slots filled)."""
    i, filled = anchor, set()
    while i < len(lines):
        s = slot_of(lines[i])
        if s and s not in filled:
            filled.add(s)
            i += 1
        elif s and s in filled:
            break                                   # the `Other` residual row starts here
        elif CONT.match(lines[i]):
            i += 1
        else:
            break
    return i, filled


def read_rows(lines, start):
    """[(name, [(kind, value)])] from `start` up to and including the row named Total."""
    rows, name, vals = [], [], []
    for l in lines[start:]:
        v = parse_value(l)
        if v is not None:
            vals.append(v)
            continue
        if vals:
            rows.append((" ".join(name).strip(), vals))
            if re.fullmatch(r"total", rows[-1][0], re.I):
                return rows
            name, vals = [], []
        # Only a fresh `Christian` or a caption ends the table. NEVER slot_of(): one body
        # row is legitimately called `Other`, which is also a column name.
        if re.fullmatch(r"christian", l, re.I) or re.match(r"^table[\s:]*\d", l, re.I):
            break
        name.append(l)
    if vals:
        rows.append((" ".join(name).strip(), vals))
    return rows


def orient(nums):
    """Where the row-total column is: ('first' | 'last' | 'none', category count).

    Detected arithmetically on every row of the table and never from the header, because
    all three layouts occur in this series and none of them announces itself.
    """
    w = len(nums[0])
    if all(r[0] == sum(r[1:]) for r in nums):
        return "first", w - 1
    if all(r[-1] == sum(r[:-1]) for r in nums):
        return "last", w - 1
    return "none", w


def to_cells(vals, kind, has_ns):
    """Drop the row-total column and name what is left."""
    cats = vals[1:] if kind == "first" else (vals[:-1] if kind == "last" else vals[:])
    if len(cats) == 9:
        return dict(zip(CATS, cats))
    if len(cats) == 8 and not has_ns:
        return dict(zip([c for c in CATS if c != "Not stated"], cats))
    return None


def extract(path):
    """Every religion COUNT table in one booklet."""
    import fitz

    doc = fitz.open(path)
    out = []
    for pg in range(doc.page_count):
        lines = [l for l in (clean(x) for x in doc[pg].get_text().splitlines()) if l]
        for a, l in enumerate(lines):
            if not re.fullmatch(r"christian", l, re.I):
                continue
            start, filled = header_span(lines, a)
            rows = [(n, v) for n, v in read_rows(lines, start) if n]
            if len(rows) < 3:
                continue
            if any(k == "dec" for _, vs in rows for k, _ in vs):
                continue                                  # the percentage twin
            widths = [len(v) for _, v in rows]
            w = max(set(widths), key=widths.count)
            if w not in (9, 10):
                continue
            keep = [(n, [x for _, x in v]) for n, v in rows if len(v) == w]
            if len(keep) < 3 or not re.fullmatch(r"total", keep[-1][0], re.I):
                continue
            kind, _ = orient([v for _, v in keep])
            if kind == "none" and w != 9:
                continue
            window = " ".join(lines[max(0, a - 16):start])
            has_ns = bool(re.search(r"not\s*stated", window, re.I)) or "Not stated" in filled
            out.append(dict(page=pg, rows=keep, width=w, kind=kind, has_ns=has_ns))
    return out


def booklet_table(key):
    """The one real count table in a booklet, with duplicate prints removed."""
    path = os.path.join(RAW, key + ".pdf")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    cands = extract(path)
    if not cands:
        raise SystemExit(f"{key}: no religion count table found")
    uniq, seen = [], set()
    for c in cands:
        sig = tuple(tuple(v) for _, v in c["rows"])
        if sig not in seen:
            seen.add(sig)
            uniq.append(c)
    # Ngamiland West prints the SAME table twice, as 10A and again as 11A: identical values
    # and an identical village list, so it is one district table and not two halves.
    c = max(uniq, key=lambda c: sum(c["rows"][-1][1]))
    c["duplicates"] = len(cands) - len(uniq)
    return c


def main():
    if "--fetch" in sys.argv:
        fetch()

    rows, report = [], []
    for key, (_, adm2, dname) in BOOKLETS.items():
        c = booklet_table(key)
        body, total = c["rows"][:-1], c["rows"][-1][1]
        cells_t = to_cells(total, c["kind"], c["has_ns"])
        if cells_t is None:
            raise SystemExit(f"{key}: {c['width']} columns and no mapping for them")

        # ---- the acceptance test: body rows sum to the printed Total, column by column ---
        n = len(total)
        slip = KNOWN_SLIPS.get(key)
        drop = slip["exclude"] if slip else None
        cmp_rows = [(nm, v) for nm, v in body if drop is None or fold(nm) != drop]
        if drop is not None and len(cmp_rows) == len(body):
            raise SystemExit(f"{key}: no row called {drop!r} to exclude any more")
        colsum = [sum(v[i] for _, v in cmp_rows) for i in range(n)]
        off = sum(abs(a - b) for a, b in zip(colsum, total))
        if off and slip is None:
            raise SystemExit(f"{key}: body rows do not sum to the printed Total\n"
                             f"  printed {total}\n  summed  {colsum}")
        if off > (slip["max_off"] if slip else 0):
            raise SystemExit(f"{key}: the known discrepancy has grown to {off:,}, above the "
                             f"{slip['max_off']} this file expects -- re-read the booklet "
                             "rather than widening it")

        for i, (name, vals) in enumerate(body, 1):
            cells = to_cells(vals, c["kind"], c["has_ns"])
            resid = bool(RESIDUAL_ROW.fullmatch(name.strip()))
            a2 = adm2
            if key == "towns":
                a2 = TOWN_ADM2.get(fold(name))
                if a2 is None:
                    raise SystemExit(f"towns: {name!r} is not one of the seven")
            a2 = ADM2_OVERRIDE.get((key, fold(name)), a2)
            gid = f"BW-{key}-{i:02d}"
            for cat in CATS:
                if cat not in cells:
                    continue          # Chobe and Ngamiland East print no `Not stated` column
                # EVERY DRAWN ROW IS `locality`, INCLUDING THE RESIDUALS, and that is for the
                # tooling rather than for tidiness: check_mapping.py, coverage.py and the
                # rest pick the level with the most units and read only that one, so a
                # second level would hide 20 rows and 148,000 people from every check in the
                # project. Residual-ness rides in the note, and bw_geo.py reads it there.
                rows.append(dict(
                    geo_id=gid, geo_level="locality",
                    geo_name=name, source_category=cat, count=cells[cat],
                    basis="self_id", year=2011, source_id=f"bw_phc_2011_{key}",
                    note=f"adm2={a2}; booklet={key}; page={c['page']}; district={dname}"
                         + ("; residual=yes" if resid else "")))
        for cat in CATS:
            if cat in cells_t:
                rows.append(dict(
                    geo_id=f"BW-{key}", geo_level="district", geo_name=dname,
                    source_category=cat, count=cells_t[cat], basis="self_id", year=2011,
                    source_id=f"bw_phc_2011_{key}",
                    note=f"adm2={adm2 or 'BW0101'}; booklet={key}; page={c['page']}; "
                         f"district={dname}; the booklet's own printed total"))
        report.append((key, c["page"], len(body), c["width"], c["kind"],
                       sum(cells_t.values()), off, c["duplicates"], c["has_ns"]))

    # ---- what got drawn, against UNSD's national 2011 table ----
    drawn = {}
    for r in rows:
        if r["geo_level"] in ("locality", "residual"):
            drawn[r["source_category"]] = drawn.get(r["source_category"], 0) + r["count"]

    hdr = (f"{'booklet':16s} {'pg':>4} {'rows':>5} {'w':>2} {'layout':<6} {'12+ total':>10} "
           f"{'slip':>5}  notes")
    print(hdr)
    print("-" * len(hdr))
    for key, pg, nrow, w, kind, tot, off, dup, ns in report:
        extra = []
        if dup:
            extra.append(f"{dup} duplicate table ignored")
        if not ns:
            extra.append("no `Not stated` column in the source")
        print(f"{key:16s} p{pg:<3} {nrow:>5} {w:>2} {kind:<6} {tot:>10,} {off:>5}  "
              + "; ".join(extra))

    nloc = len({r["geo_id"] for r in rows if r["geo_level"] == "locality"})
    nres = len({r["geo_id"] for r in rows if r["geo_level"] == "residual"})
    print(f"\n  named localities   {nloc:>6}")
    print(f"  district residuals {nres:>6}")
    print("\n  drawn against UNSD's national 2011 table, which covers all 28 districts:")
    print(f"    {'category':<14}{'drawn':>10}{'national':>10}{'share':>8}")
    for cat, nat in sorted(NATIONAL_2011.items(), key=lambda kv: -kv[1]):
        d = drawn.get(cat, 0)
        print(f"    {cat:<14}{d:>10,}{nat:>10,}{d / nat:>8.1%}")
    tot_drawn = sum(drawn.get(c, 0) for c in NATIONAL_2011)
    tot_nat = sum(NATIONAL_2011.values())
    print(f"    {'(not stated)':<14}{drawn.get('Not stated', 0):>10,}"
          f"{'':>10}   excluded, §3.5")
    print(f"    {'TOTAL':<14}{tot_drawn:>10,}{tot_nat:>10,}{tot_drawn / tot_nat:>8.1%}")
    missing = sum(p for _, p in NOT_COVERED.values())
    print(f"\n  the shortfall is the two districts with no booklet: "
          f"{', '.join(n for n, _ in NOT_COVERED.values())}, {missing:,} people in 2011, "
          f"{missing / CENSUS_POPULATION_2011:.1%} of the census.")
    if not 0.90 <= tot_drawn / tot_nat <= 0.98:
        raise SystemExit(f"drawn/national is {tot_drawn / tot_nat:.3f}, outside the band the "
                         "two missing districts explain -- something else is wrong")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis",
            "year", "source_id", "note"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
