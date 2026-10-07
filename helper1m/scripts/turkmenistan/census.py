"""Read the 2022 census settlement tables (section 1, tables 1.3-1.14) into a tree.

Reads  helper1m/data/turkmenistan/raw/tm_census2022_results_en_1.pdf
Writes helper1m/data/turkmenistan/census_units.csv        one row per census unit
       helper1m/data/turkmenistan/census_settlements.csv  one row per leaf settlement

Section 1 of "Results of the Complete Population and Housing Census of
Turkmenistan 2022" (17 December 2022) prints, for each velayat, an urban table
(1.5-1.9: etrap or city of velayat subordination, then its cities and towns) and
a rural table (1.10-1.14: etrap, then the rural areas run by a town and the
gengeshliks, then their villages). Ashgabat (1.4) is four etraps, all urban.
Every row is both sexes, male, female.

The text layer gives one cell per line. A row is a run of name lines followed by
three figures. The nesting is not marked, so it is rebuilt from the sums: a row
is a parent when the rows after it add up to it exactly. Each table is checked
to add up to its velayat row in table 1.3, and male + female = both sexes on
every row.
"""
import csv
import re
import sys
from pathlib import Path

import fitz

HELPER = Path(__file__).resolve().parents[2]
DATA = HELPER / "data" / "turkmenistan"
PDF = DATA / "raw" / "tm_census2022_results_en_1.pdf"

# A few figures are printed without the thousands space ("1393").
NUM = re.compile(r"^\s*(\d{1,3}(?:[  ]+\d{3})*|\d{4,7}|[–-])\s*$")
DROP = {"RESULTS OF THE COMPLETE POPULATION AND", "HOUSING CENSUS OF TURKMENISTAN – 2022",
        "Ending", "Continuation", "Both sexes", "including:", "male", "female"}

# Table 1.3 (both sexes, urban, rural) — asserted against the PDF in main().
VELAYAT_ROWS = {
    "Ashgabat": (1030063, 1030063, 0),
    "Ahal": (886845, 313785, 573060),
    "Balkan": (529895, 435090, 94805),
    "Dashoguz": (1550354, 473861, 1076493),
    "Lebap": (1447298, 656021, 791277),
    "Mary": (1613386, 412677, 1200709),
}
NATIONAL = 7057841


def num(s):
    s = s.strip()
    return 0 if s in ("–", "-") else int(re.sub(r"[  ]", "", s))


def rows_of_table(lines):
    """Lines of one table (title already removed) -> [(name, both, male, female)]."""
    out, name, figs = [], [], []
    for ln in lines:
        s = ln.strip()
        if not s or s in DROP:
            continue
        m = NUM.match(ln)
        if m:
            figs.append(num(m.group(1)))
            if len(figs) == 3:
                nm = re.sub(r"\s+", " ", " ".join(name)).strip()
                if not nm:
                    sys.exit(f"figures without a name after {out[-1] if out else None}")
                out.append((nm, *figs))
                name, figs = [], []
        else:
            if figs:
                sys.exit(f"name line {s!r} inside a row of figures (after {name})")
            name.append(s)
    if name or figs:
        sys.exit(f"dangling {name} {figs}")
    for r in out:
        if r[2] + r[3] != r[1]:
            sys.exit(f"male + female != both: {r}")
    return out


def tables(pdf):
    """{table number: [lines]}, for tables 1.3-1.14."""
    text = []
    for page in fitz.open(pdf):
        text.extend(page.get_text().split("\n"))
    out, cur, skipping = {}, None, False
    for ln in text:
        m = re.match(r"^(1\.\d+)\.\s", ln.strip())
        if m:
            cur = m.group(1)
            out[cur] = []
            skipping = True          # title may wrap; resume at "Both sexes"
            continue
        if skipping:
            if ln.strip() == "Both sexes":
                skipping = False
            continue
        if cur:
            out[cur].append(ln)
    return {k: v for k, v in out.items() if k in {f"1.{i}" for i in range(3, 15)}}


# "geneshlik" is a misprint the table has once (Goyunjy, Yoloten).
PARENT_WORDS = re.compile(r"\b(gengeshlik|geneshlik|town|city|etrap)\b", re.I)


def build(rows, i, target, depth, maxdepth):
    """Consume rows from i as children until they sum to target. Returns
    (children, next i) or None. A child at depth < maxdepth whose name could be
    a parent is tried as one first."""
    kids, s = [], 0
    while s < target:
        if i >= len(rows):
            return None
        nm, both, m, f = rows[i]
        i += 1
        node = {"name": nm, "pop": both, "male": m, "female": f, "children": []}
        if depth < maxdepth and PARENT_WORDS.search(nm) and both > 0:
            sub = build(rows, i, both, depth + 1, maxdepth)
            if sub is not None:
                node["children"], i = sub
        kids.append(node)
        s += both
    if s != target:
        return None
    return kids, i


VELAYAT_OF = {"1.5": "Ahal", "1.6": "Balkan", "1.7": "Dashoguz", "1.8": "Lebap", "1.9": "Mary",
              "1.10": "Ahal", "1.11": "Balkan", "1.12": "Dashoguz", "1.13": "Lebap", "1.14": "Mary"}


# Unit labels that differ between the urban and rural tables.
UNIT_FIX = {
    "including Balkanabat city": "Balkanabat city",   # rural Balkan, first row
    "Tukmengala etrap": "Turkmengala etrap",          # rural Mary, misprint
}


def leaves(node, path):
    if not node["children"]:
        yield path + [node["name"]], node["pop"]
    for c in node["children"]:
        yield from leaves(c, path + [node["name"]])


def main():
    tb = tables(PDF)
    t13 = rows_of_table(tb["1.3"])
    got = {r[0]: r[1] for r in t13}
    assert got["Turkmenistan"] == NATIONAL, got["Turkmenistan"]
    for v, (tot, urb, rur) in VELAYAT_ROWS.items():
        key = "Ashgabat city" if v == "Ashgabat" else f"{v} velayat"
        assert got[key] == tot, (v, got[key])
    assert sum(t[0] for t in VELAYAT_ROWS.values()) == NATIONAL

    units, sett = [], []
    # Ashgabat: table 1.4, the city row then four etraps.
    r14 = rows_of_table(tb["1.4"])
    assert r14[0][0] == "Ashgabat city" and r14[0][1] == VELAYAT_ROWS["Ashgabat"][0]
    assert sum(r[1] for r in r14[1:]) == r14[0][1]
    for nm, both, *_ in r14[1:]:
        units.append({"velayat": "Ashgabat", "unit": nm, "urban": both, "rural": 0})
        sett.append({"velayat": "Ashgabat", "unit": nm, "parent": "", "name": nm,
                     "kind": "urban", "pop": both})

    acc = {}
    for t in [f"1.{i}" for i in range(5, 15)]:
        v = VELAYAT_OF[t]
        kind = "urban" if int(t.split(".")[1]) <= 9 else "rural"
        rows = rows_of_table(tb[t])
        head = rows[0]
        assert head[0].lower() == f"{kind} population of {v} velayat".lower(), head
        want = VELAYAT_ROWS[v][1 if kind == "urban" else 2]
        assert head[1] == want, (t, head, want)
        # urban: velayat > etrap/city > settlement (a city can hold towns);
        # rural: velayat > etrap > town/gengeshlik > village
        res = build(rows, 1, head[1], 1, 2 if kind == "urban" else 3)
        if res is None:
            sys.exit(f"table {t}: could not rebuild the nesting")
        kids, nxt = res
        if nxt != len(rows):
            sys.exit(f"table {t}: {len(rows) - nxt} rows left over: {rows[nxt:nxt + 3]}")
        for k in kids:
            k["name"] = UNIT_FIX.get(k["name"], k["name"])
            key = (v, k["name"])
            a = acc.setdefault(key, {"velayat": v, "unit": k["name"], "urban": 0, "rural": 0})
            a[kind] += k["pop"]
            for path, pop in leaves(k, []):
                sett.append({"velayat": v, "unit": k["name"],
                             "parent": path[-2] if len(path) > 2 else "",
                             "name": path[-1], "kind": kind, "pop": pop})
        print(f"table {t} {v} {kind}: {len(kids)} units, {len(rows) - 1} rows, "
              f"{sum(1 for _ in kids)} top-level, total {head[1]:,}")
    units.extend(acc.values())
    tot = sum(u["urban"] + u["rural"] for u in units)
    assert tot == NATIONAL, tot
    assert sum(s["pop"] for s in sett) == NATIONAL

    with open(DATA / "census_units.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["velayat", "unit", "urban", "rural", "total"])
        w.writeheader()
        for u in units:
            w.writerow({**u, "total": u["urban"] + u["rural"]})
    with open(DATA / "census_settlements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["velayat", "unit", "parent", "name", "kind", "pop"])
        w.writeheader()
        w.writerows(sett)
    print(f"{len(units)} census units, {len(sett)} leaf settlements, total {tot:,}")
    for u in units:
        print(f"  {u['velayat']:9s} {u['unit']:40s} {u['urban'] + u['rural']:>9,}")


if __name__ == "__main__":
    main()
