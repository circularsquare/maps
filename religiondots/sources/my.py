"""Malaysia — Banci Penduduk dan Perumahan Malaysia 2020 (DOSM).

Religion x administrative district, from the sixteen state volumes plus the
national volume.  Writes ``data/normalized/my.csv``.

    python sources/my.py

Source
------
DOSM, *Penemuan Utama Banci Penduduk dan Perumahan Malaysia 2020*, released
2022-05-29, downloaded from the eStatistik portal (free registration).

Each state's publication ships ~21 files.  Twenty carry the subject
``MYLOCAL STATS`` (tables 17-102, socioeconomic, **no religion**); the one that
matters is ``<STATE> JADUAL 1 HINGGA 16``.  In Perak's download list it was
record 21 of 21, alone on page 3.

**Table 7** is *Bilangan penduduk mengikut agama, jantina dan daerah
pentadbiran/ jajahan* — population by religion, sex and administrative
district.  The national volume's equivalent is **Table 6**, by state.

Sabah and Sarawak title their volumes *"State Sabah"* / *"State Sarawak"* where
every peninsular one is *"Negeri X"*, and they sit among 27 and 40 per-district
publications.  The state file reproduces those district volumes exactly — this
was checked against the standalone Kampar volume, which agrees on all eight
cells.

Religion stops at administrative district.  ``JADUAL BANCI MALAYSIA 2020 MUKIM
BANDAR PEKAN`` covers all 1,756 mukim but carries only population, **ethnicity**
and age; the state volumes' Table 11 is mukim-level population/households only.

The five traps this parser exists to survive
--------------------------------------------
1. **Sarawak ships an empty decoy of its own religion table.**  Sheet ``'7'``
   has the right title, headers, footnote and forty district names in ALL CAPS,
   alphabetical — and every value cell blank, under ``data_only`` both ways.
   The real table is ``'7 (T)'``.  Taking the first title match yields a Sarawak
   with zero people in all forty districts while every other check still passes.
   Hence :func:`pick_total_panels` scores candidates by how much data they hold.
2. **A ``(cont'd)`` sheet means two different things.**  In fourteen states
   ``'7'/'7 (2)'/'7 (3)'`` are Total/Male/Female.  In Sabah and Sarawak the
   continuations are *more districts* and the sexes get their own runs.  The
   sheet name cannot tell you which; the header's ``Sex : Total`` marker can.
3. **``-`` is an in-band zero**, in six Sabah districts.  A "every cell numeric"
   guard drops those rows and Sabah silently becomes 18 districts summing to
   2,935,745 against a printed 3,418,785.  It is a true zero, not a suppression:
   Kota Belud's six numeric cells sum to its printed total exactly.
4. **A sheet can repeat its header block mid-sheet.**  Sabah's Table 7
   paginates at row 28 and resumes at row 38.  Read to ``max_row``.
5. **Match a header string exactly, never by prefix** — dropping rows that
   start with ``negeri`` to skip the ``Negeri/ State`` header also drops
   Negeri Sembilan.

Checks
------
Every state: religion columns sum to the state total, and districts sum to the
state total.  The whole extract then reconciles to the person against the
national volume's Table 6 — a different publication, so a real outside check.
All are assertions; the module refuses to write a file that fails one.
"""

from __future__ import annotations

import csv
import os
import sys

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "my")
OUT = os.path.join(ROOT, "data", "normalized", "my.csv")

SOURCE_ID = "my_phc_2020_dosm"
YEAR = 2020
BASIS = "self_id"

# DOSM's own state codes, as served by the Population Quick Info dropdown.
STATES = [
    ("01", "Johor", "JOHOR JADUAL 1 HINGGA 16.xlsx"),
    ("02", "Kedah", "KEDAH JADUAL 1 HINGGA 16.xlsx"),
    ("03", "Kelantan", "KELANTAN JADUAL 1 HINGGA 16.xlsx"),
    ("04", "Melaka", "MELAKA JADUAL 1 HINGGA 16.xlsx"),
    ("05", "Negeri Sembilan", "NEGERI SEMBILAN JADUAL 1 HINGGA 16.xlsx"),
    ("06", "Pahang", "PAHANG JADUAL 1 HINGGA 16.xlsx"),
    ("07", "Pulau Pinang", "PULAU PINANG JADUAL 1 HINGGA 16.xlsx"),
    ("08", "Perak", "PERAK JADUAL 1 HINGGA 16.xlsx"),
    ("09", "Perlis", "PERLIS JADUAL 1 HINGGA 16.xlsx"),
    ("10", "Selangor", "SELANGOR JADUAL 1 HINGGA 16.xlsx"),
    ("11", "Terengganu", "TERENGGANU JADUAL 1 HINGGA 16.xlsx"),
    ("12", "Sabah", "SABAH JADUAL 1 HINGGA 16.xlsx"),
    ("13", "Sarawak", "SARAWAK JADUAL 1 HINGGA 16.xlsx"),
    ("14", "W.P. Kuala Lumpur", "W.P. KUALA LUMPUR JADUAL 1 HINGGA 16.xlsx"),
    ("15", "W.P. Labuan", "W.P. LABUAN JADUAL 1 HINGGA 16.xlsx"),
    ("16", "W.P. Putrajaya", "W.P. PUTRAJAYA JADUAL 1 HINGGA 16.xlsx"),
]
NATIONAL = "JADUAL 1 HINGGA 29.xlsx"

# Entities the census does not break into districts: the state row IS the unit.
SINGLE_UNIT = {"Perlis", "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"}

# Malay header stem -> the source's own English label, kept verbatim per §3.
CATEGORIES = [
    ("islam", "Islam"),
    ("kristian", "Christianity"),
    ("buddha", "Buddhism"),
    ("hindu", "Hinduism"),
    ("lain-lain", "Others"),
    ("tiada agama", "No Religion"),
    ("tidak diketahui", "Unknown"),
]
LABELS = [lab for _, lab in CATEGORIES]

DASHES = {"-", "‐", "‑", "‒", "–", "—", "−"}

# Header strings that must be matched EXACTLY -- trap 5.
HEADER_EXACT = {"negeri", "negeri state", "state", "daerah pentadbiran",
                "daerah pentadbiran/ jajahan", "jumlah", "jumlah total"}
HEADER_PREFIX = ("nota", "note", "lain-lain terdiri", "other include",
                 "others include", "jadual", "table")


def norm(x) -> str:
    return " ".join(str(x).split()) if x is not None else ""


def to_num(v):
    """`-` is a true zero (trap 3). Anything else non-numeric is not data."""
    if isinstance(v, (int, float)):
        return int(round(v))
    if isinstance(v, str):
        s = v.strip()
        if s in DASHES:
            return 0
        try:
            return int(round(float(s.replace(",", "").replace(" ", ""))))
        except ValueError:
            return None
    return None


def _head(ws, maxrow=12, maxcol=24):
    return [[norm(c) for c in row] for row in
            ws.iter_rows(min_row=1, max_row=maxrow, max_col=maxcol, values_only=True)]


def is_total_panel(ws, table_word="religion") -> bool:
    """Is this the religion table, and the Total panel rather than Male/Female?

    Trap 2: the sheet NAME cannot answer this. The header marker can.

    Trap 2a: the marker's spacing is not stable across DOSM's own volumes --
    the sixteen state books write ``Sex : Male`` and the national book writes
    ``Sex: Male``.  Matching the spaced form alone silently admits the Male and
    Female panels of the national volume, and then the last one read wins, so
    every state reconciles against its own FEMALE count.  Strip whitespace
    before comparing.
    """
    flat = " || ".join(" | ".join(r) for r in _head(ws)).lower()
    if f"by {table_word}" not in flat:
        return False
    squashed = flat.replace(" ", "")
    for marker in ("sex:male", "jantina:lelaki", "sex:female", "jantina:perempuan"):
        if marker in squashed:
            return False
    return True


def header_map(ws):
    """Row index and column of each religion, from the header block."""
    for i, cells in enumerate(_head(ws)):
        low = [c.lower() for c in cells]
        if any(c.startswith("islam") for c in low):
            m = {}
            for j, c in enumerate(low):
                for stem, label in CATEGORIES:
                    if c.startswith(stem):
                        m.setdefault(label, j)
            if len(m) == len(CATEGORIES):
                return i + 1, m
    return None, None


def data_rows(ws):
    """Every data row in the sheet.

    Reads to ``max_row`` because the header block can repeat mid-sheet (trap 4).
    """
    hrow, cmap = header_map(ws)
    if not cmap:
        return []
    out = []
    for row in ws.iter_rows(min_row=hrow + 1, max_row=ws.max_row, max_col=26, values_only=True):
        name = norm(row[0])
        low = name.lower()
        if not name or low in HEADER_EXACT or low.startswith(HEADER_PREFIX):
            continue
        vals = {}
        for label, j in cmap.items():
            n = to_num(row[j] if j < len(row) else None)
            if n is None:
                vals = None
                break
            vals[label] = n
        if vals is None:
            continue
        total = to_num(row[1])
        out.append((name, total if total is not None else sum(vals.values()), vals))
    return out


def pick_total_panels(path, table_word="religion"):
    """Every populated Total panel, concatenated.

    Trap 1: Sarawak's decoy sheet is well-formed and empty, so panels are kept
    only if they actually carry rows, and all surviving panels are unioned.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    used, rows = [], []
    try:
        for s in wb.sheetnames:
            ws = wb[s]
            if not is_total_panel(ws, table_word):
                continue
            r = data_rows(ws)
            if r:
                used.append(s)
                rows.extend(r)
    finally:
        wb.close()
    return used, rows


def split_state(rows, state):
    """Separate the state's own row from its districts, de-duplicating."""
    key = state.lower().replace("w.p. ", "")
    head, districts, seen = None, [], set()
    for name, total, vals in rows:
        if name.lower().replace("w.p. ", "") == key:
            if head is None:
                head = (name, total, vals)
            continue
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        districts.append((name, total, vals))
    return head, districts


def main() -> int:
    missing = [f for _, _, f in STATES if not os.path.exists(os.path.join(RAW, f))]
    if missing:
        print("missing raw workbooks in %s:" % RAW, file=sys.stderr)
        for m in missing:
            print("   ", m, file=sys.stderr)
        return 1

    records = []
    per_state = {}

    for code, state, fname in STATES:
        used, rows = pick_total_panels(os.path.join(RAW, fname))
        head, districts = split_state(rows, state)
        assert head is not None, f"{state}: no state total row found"

        cat_sum = sum(head[2].values())
        assert cat_sum == head[1], (
            f"{state}: religion columns sum to {cat_sum:,}, printed total {head[1]:,}")

        if state in SINGLE_UNIT:
            assert not districts, f"{state}: expected no district breakdown, got {len(districts)}"
        else:
            dsum = sum(d[1] for d in districts)
            assert dsum == head[1], (
                f"{state}: districts sum to {dsum:,}, state total {head[1]:,} "
                f"({len(districts)} districts from sheets {used})")

        per_state[state] = head
        sid = f"MYS_{code}"
        for label in LABELS:
            records.append([sid, "state", state, label, head[2][label],
                            BASIS, YEAR, SOURCE_ID, f"level=state; dosm={code}"])

        units = districts or [(state, head[1], head[2])]
        level = "district" if districts else "district"
        for k, (name, total, vals) in enumerate(units, start=1):
            did = f"MYS_{code}_{k:02d}"
            note = f"level=district; state={state}"
            if not districts:
                note += "; single-unit state"
            for label in LABELS:
                records.append([did, level, name, label, vals[label],
                                BASIS, YEAR, SOURCE_ID, note])

        print(f"  {state:<20} {len(districts) or 1:>3} units  {head[1]:>11,}  sheets={','.join(used)}")

    # ---- outside check: the national volume is a different publication ----
    nat_used, nat_rows = pick_total_panels(os.path.join(RAW, NATIONAL))
    nat = {}
    for name, total, vals in nat_rows:
        nat[name.lower().replace("w.p.", "").replace(".", "").strip()] = (total, vals)
    assert nat, f"national volume: no religion panel found (sheets tried: {nat_used})"

    for state, head in per_state.items():
        k = state.lower().replace("w.p.", "").replace(".", "").strip()
        assert k in nat, f"{state} absent from the national table"
        assert nat[k][0] == head[1], (
            f"{state}: state volume {head[1]:,} vs national volume {nat[k][0]:,}")
        for label in LABELS:
            assert nat[k][1][label] == head[2][label], (
                f"{state}/{label}: {head[2][label]:,} vs national {nat[k][1][label]:,}")

    my_total = sum(h[1] for h in per_state.values())
    nat_total = nat["malaysia"][0]
    assert my_total == nat_total, f"country total {my_total:,} vs national {nat_total:,}"

    for label in LABELS:
        mine = sum(h[2][label] for h in per_state.values())
        assert mine == nat["malaysia"][1][label], (
            f"{label}: states sum to {mine:,}, national says {nat['malaysia'][1][label]:,}")
        records.append(["MYS_00", "country", "MALAYSIA", label, mine,
                        BASIS, YEAR, SOURCE_ID, "level=country"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count",
                    "basis", "year", "source_id", "note"])
        w.writerows(records)

    ndist = sum(1 for r in records if r[1] == "district") // len(LABELS)
    print(f"\n  country total {my_total:,} reconciles against the national volume")
    print(f"  {ndist} districts, 16 states, {len(records)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
