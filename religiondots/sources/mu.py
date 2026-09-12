"""Mauritius — Statistics Mauritius, 2022 Housing and Population Census, Table D6.

Reads (or fetches) data/raw/mu/ and writes data/normalized/mu.csv.

**THE ONLY SOURCE ON THIS MAP THAT SPLITS HINDUISM BY COMMUNITY.** Table D6 gives thirteen
religious groups of which **five are Hindu** — Marathi, Tamil, Telugu, Vedic/Arya Samaj and
"Hindu & Other Hindu" — on the **finest civil geography Mauritius has**: Municipal Council
Wards and Village Council Areas, ~8,200 people each. India's census does not do this and
neither does Guyana's; before Mauritius the tree had exactly two Hindu nodes, one of them
Vietnam's Cham Balamon.

**IT IS §3.9'S TRADE MADE INSIDE ONE REPORT, AND BOTH HALVES ARE PUBLISHED.** Table D5, eight
pages earlier, is the same census at **island level** with **sixty-odd individually named
bodies** — `L'Assemblée de Dieu`, `Mission Salut et Guérison`, `La Voix de la Delivrance`,
`Peniel Tabernacle`, `Church of England`, `Arya Samajist`, `Christian Tamil`. D6 is the same
people at ~150 units with those bodies pooled into thirteen. The report's own contents page
labels the levels: `I` for island, `R` for ward/VCA. **D6 is what is drawn**; D5 is read only
to check the national column, because a category list with three units is not a geography.

**THE PARSE RULE IS "THE LAST FOURTEEN TOKENS ARE THE FIGURES".** Nothing else survives the
row shapes:

  * `PAMPLEMOUSSES DISTRICT-Wholly Rural 140,856 579 …` — name and figures on one line;
  * `Town of Port Louis-Ward 3 12,612 235 …` — **the ward number is part of the name** and
    reads as a fifteenth figure to any numeric filter;
  * `Region 1 - La Ferme 7,578 1 …` — Rodrigues, same trap with the index in the middle;
  * `Town of Port Louis-Ward 2-North (South in Moka & West in B/R) 8,055 …` — a
    parenthetical that names *other districts* inside the unit's own name.

**AND FOUR PARENT ROWS PUT THEIR NAME AND THEIR FIGURES ON DIFFERENT LINES.** `PORT LOUIS
DISTRICT-Wholly Urban` is at y=179 and its fourteen numbers at y=180; same for `GRAND PORT
DISTRICT`, `RODRIGUES` and `ISLAND OF RODRIGUES - Wholly Rural`. Grouping words by exact y
splits them into a nameless number row and a numberless name row, and **both then fail the
row test silently** — the district vanishes from the parent-sum check that is the only thing
verifying the units underneath it. Rows are clustered with a y tolerance instead.

**THE DRAWN TIER IS IDENTIFIED BY INDENTATION AND NOTHING ELSE.** There is no code column and
no type column anywhere in the table. The stub is indented by level — 24.6 republic, 31.8
island, 39.0 district, 46.2 district urban/rural split, **53.4 the drawn unit** — and that x
position is the only thing distinguishing `MOKA DISTRICT - Urban` from a village inside it.
Asserted: every drawn unit must sit at 53.4, and the units must sum to their district.

Usage:
    python sources/mu.py --fetch    one 4.0 MB PDF, seconds
    python sources/mu.py            normalise from data/raw/mu/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mu")
OUT = os.path.join(ROOT, "data", "normalized", "mu.csv")

SOURCE_ID = "mu_hpc_2022"
YEAR = 2022
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# The site is SharePoint and its .aspx page paths are not guessable — the Census2022 and HPC
# landing pages both 404 and SB_Population.aspx redirect-loops a scripted client at 30 hops.
# The /Documents/ tree is open and is what to fetch. sources/mu.md 1.
PDF_URL = ("https://statsmauritius.govmu.org/Documents/Census_and_Surveys/Census2022/"
           "HPC_TR_Vol2_Demography_Yr22.pdf")
PDF_NAME = "mu_hpc2022_vol2.pdf"

TITLE_RE = re.compile(r"Table D6\s*-\s*Resident population by geographical location and "
                      r"religious group", re.I)

# Left to right across the page, exactly as the header prints them. `Total` first because it
# is the unit's own universe and not a religion.
CATEGORIES = [
    "Total",
    "Buddhist/Chinese",
    "L'Assemblee de Dieu / M.S et Guerison",
    "Church of England/Protestant",
    "Roman Catholic",
    "Other Christian",
    "Marathi/Marathi Hindu",
    "Tamil/Tamil Hindu",
    "Telugu/Telugu Hindu",
    "Vedic/Hindu Vedic & Aryan",
    "Hindu & Other Hindu",
    "Islam/Muslim & Other Muslim",
    "No religion",
    "Other & Not stated",
]
TOTAL_CAT = "Total"
NCOL = len(CATEGORIES)

# Stub indentation -> level. The ONLY level marker in the table.
UNIT_X = 53.4
DISTRICT_X = 39.0
X_TOL = 1.5

REPUBLIC = "REPUBLIC OF MAURITIUS"
NATIONAL = 1_233_097              # D6's own REPUBLIC OF MAURITIUS row

# Rows that are sub-totals of a row already counted, and must never be summed with it.
SPLIT_SUFFIX = re.compile(r"\s*-\s*(Urban|Rural|Wholly Urban|Wholly Rural)\s*$", re.I)
# `PAMPLEMOUSSES DISTRICT-Wholly Rural` -> `PAMPLEMOUSSES`, `ISLAND OF RODRIGUES - Wholly
# Rural` -> `RODRIGUES`. The bare district name is what COD-AB's ADM1_EN carries.
DISTRICT_NAME = re.compile(r"\s*(ISLAND OF\s*|DISTRICT.*$|-\s*Wholly.*$)", re.I)

NUM = re.compile(r"^[\d,]+$")
Y_TOL = 3.0                       # merges the four name/figures-on-two-lines parents


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 2_000_000:
        print("already have", dest)
        return
    print("GET", PDF_URL)
    r = requests.get(PDF_URL, timeout=900, verify=False, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    # §5a: HTTP 200 is not a download. Assert size AND type.
    with open(dest, "rb") as fh:
        magic = fh.read(5)
    if magic != b"%PDF-":
        raise SystemExit(f"{dest} is not a PDF -- starts {magic!r}, "
                         f"{os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def _rows(page):
    """(x0, name, [14 ints]) per printed row, in page order.

    Words are clustered into rows with a y tolerance, because four parent rows put their
    name and their figures on lines one point apart. Then the LAST fourteen tokens are the
    figures and everything before them is the name — the only rule that survives ward
    numbers, Rodrigues region indices and the parentheticals.
    """
    bands = []
    for w in sorted(page.get_text("words"), key=lambda w: (w[1], w[0])):
        if bands and abs(w[1] - bands[-1][0]) <= Y_TOL:
            bands[-1][1].append(w)
        else:
            bands.append((w[1], [w]))
    out = []
    for _, ws in bands:
        ws = sorted(ws, key=lambda w: w[0])
        toks = [w[4] for w in ws]
        if len(toks) <= NCOL:
            continue
        tail = toks[-NCOL:]
        if not all(NUM.match(t) for t in tail):
            continue
        name = " ".join(toks[:-NCOL]).strip()
        if not name:
            continue
        out.append((round(ws[0][0], 1), name,
                    [int(t.replace(",", "")) for t in tail]))
    return out


def _nested_towns(raw):
    """The town rows that are PARENTS of the ward rows printed beside them.

    **A SIXTH TIER HIDES INSIDE THE FINEST ONE AND NOTHING MARKS IT.** `Town of Curepipe`
    (70,008) is printed at x0=53.4, the drawn-unit indentation, and so are `Town of
    Curepipe-Ward 1` … `Ward 5`, which sum to it exactly. Four towns do this — Beau
    Bassin/Rose Hill, Curepipe, Quatre Bornes and Vacoas/Phoenix, 334,496 people, **27.1% of
    the country** — and summing the indentation alone counts every one of them twice while
    leaving all 209 per-row totals perfect. Serbia's §9p `Grad` problem in a second country,
    and the same fix: a parent's children are the consecutive following rows that sum to it
    EXACTLY, which doubles as a parse check.

    Two conditions are required, not one. The exact sum is the structural test; the name
    prefix is an independent corroboration, and demanding both makes a false positive need a
    coincidence in two unrelated spaces at once. Port Louis is NOT here — its town row is the
    district row `PORT LOUIS DISTRICT-Wholly Urban` one level up, so its wards have no
    same-level parent to remove.
    """
    units = [(i, n, v[0]) for i, (x0, n, v) in enumerate(raw)
             if abs(x0 - UNIT_X) <= X_TOL]
    parents = {}
    for k, (i, name, total) in enumerate(units):
        run = 0
        for j in range(k + 1, len(units)):
            # children must be consecutive rows in the table, not merely later ones
            if units[j][0] != units[j - 1][0] + 1:
                break
            run += units[j][2]
            if run == total and j - k >= 2:
                kids = [units[m][1] for m in range(k + 1, j + 1)]
                if all(c.startswith(name) for c in kids):
                    parents[name] = kids
                break
            if run > total:
                break
    return parents


def read():
    import fitz

    p = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    doc = fitz.open(p)

    pages = [i for i in range(doc.page_count)
             if TITLE_RE.search(" ".join(doc[i].get_text().split()))]
    if not pages:
        raise SystemExit(f"{p}: no page carries Table D6's title -- Statistics Mauritius "
                         "has re-issued the volume")
    print(f"  Table D6 on pages {[i + 1 for i in pages]}")

    raw = []
    for i in pages:
        raw.extend(_rows(doc[i]))
    if not raw:
        raise SystemExit("Table D6 pages found but no row parsed -- the layout has moved")

    for town, kids in sorted(_nested_towns(raw).items()):
        print(f"  nested tier: {town!r} is the parent of {len(kids)} wards printed at the "
              "same indentation")

    nested = _nested_towns(raw)

    out, seen, district = [], {}, None
    for x0, name, vals in raw:
        is_unit = abs(x0 - UNIT_X) <= X_TOL and name not in nested
        if is_unit:
            level = "unit"
        elif name in nested:
            level = "town"
        elif name.upper().startswith(REPUBLIC):
            level = "country"
        elif "ISLAND OF" in name.upper() and "RODRIGUES" not in name.upper():
            level = "island"
        else:
            level = "district"

        # The district a unit belongs to is its POSITION in the table and nothing else —
        # there is no code column anywhere in D6. sources/mu_geo.py checks every unit's
        # polygon against this, which is the only thing that can catch a wrong name pairing
        # (§12 shape 2). Keyed on the INDENTATION, not the name: `ISLAND OF RODRIGUES -
        # Wholly Rural` is a district row and does not say `DISTRICT`, and a name-based rule
        # silently filed all six Rodrigues regions under Black River.
        if abs(x0 - DISTRICT_X) <= X_TOL and not name.upper().startswith("ISLAND OF MAURITIUS"):
            district = DISTRICT_NAME.sub("", name).strip()
        # `MOKA DISTRICT - Urban` is a slice of `MOKA DISTRICT`, not a peer of it.
        if level != "unit" and SPLIT_SUFFIX.search(name) and "DISTRICT-Wholly" not in name:
            level += "_split"
        key = (level, name)
        if key in seen:
            raise SystemExit(f"duplicate row {key} -- the same unit is printed twice")
        seen[key] = True
        code = f"{level[:1].upper()}{len(seen):03d}"
        for cat, n in zip(CATEGORIES, vals):
            note = f"level={level}; x0={x0}"
            if cat == TOTAL_CAT:
                note += "; universe total, not a religion category"
            if level == "unit":
                note += f"; district={district}"
            out.append({"geo_id": code, "geo_level": level, "geo_name": name,
                        "source_category": cat, "count": n, "basis": BASIS,
                        "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return out


def check(rows):
    ok = True

    levels = {}
    for r in rows:
        levels.setdefault(r["geo_level"], set()).add(r["geo_name"])
    for lv in sorted(levels):
        print(f"      {lv:<16} {len(levels[lv]):>4} rows")

    units = sorted(levels.get("unit", ()))
    good = len(units) > 100
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} {len(units)} drawn units at x0={UNIT_X}")

    nat = {r["source_category"]: r["count"] for r in rows
           if r["geo_level"] == "country" and not SPLIT_SUFFIX.search(r["geo_name"])}
    good = nat.get(TOTAL_CAT) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national universe {nat.get(TOTAL_CAT):,} "
          f"(expected {NATIONAL:,})")

    by_row = {}
    for r in rows:
        by_row.setdefault((r["geo_level"], r["geo_name"]), {})[r["source_category"]] = \
            r["count"]
    bad = [k for k, d in by_row.items()
           if sum(v for c, v in d.items() if c != TOTAL_CAT) != d[TOTAL_CAT]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 13 groups sum to Total on all "
          f"{len(by_row)} rows ({len(bad)} failures)")
    for k in bad[:5]:
        print(f"        {k}")

    # THE CHECK THAT MATTERS: the drawn units must sum to the republic on every column.
    # Nothing else verifies that the indentation rule picked the right rows -- a district
    # read as a unit would double-count it, and a unit read as a district would lose it,
    # and every per-row total would still be perfect either way.
    bad = []
    for cat in CATEGORIES:
        s = sum(r["count"] for r in rows
                if r["geo_level"] == "unit" and r["source_category"] == cat)
        if s != nat[cat]:
            bad.append((cat, s, nat[cat], s - nat[cat]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the {len(units)} units sum to the republic on "
          f"all {NCOL} columns")
    for c, s, n, d in bad:
        print(f"        {c}: {s:,} vs {n:,}  ({d:+,})")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in CATEGORIES:
        n = nat[cat]
        mark = "  <- universe" if cat == TOTAL_CAT else ""
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:6.2f}%  {cat}{mark}")

    hindu = sum(nat[c] for c in CATEGORIES if "Hindu" in c or "Aryan" in c)
    chris = sum(nat[c] for c in ("L'Assemblee de Dieu / M.S et Guerison",
                                 "Church of England/Protestant", "Roman Catholic",
                                 "Other Christian"))
    print(f"\n  the three headline shares, as a read of the column order:")
    print(f"    Hindu (5 cells)   {hindu:>9,}  {100.0 * hindu / NATIONAL:5.1f}%  "
          "(Statistics Mauritius publishes 47.9%)")
    print(f"    Christian (4)     {chris:>9,}  {100.0 * chris / NATIONAL:5.1f}%  "
          "(32.3%)")
    print(f"    Muslim (1)        {nat['Islam/Muslim & Other Muslim']:>9,}  "
          f"{100.0 * nat['Islam/Muslim & Other Muslim'] / NATIONAL:5.1f}%  (18.2%)")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows = read()
    check(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
