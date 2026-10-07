"""Mauritius: Statistics Mauritius, 2022 Housing and Population Census, Volume II Table D9.

Reads (or fetches) data/raw/mu/ and writes data/normalized/mu.csv.

Table D9 is "Resident population by geographical location and language usually spoken at
home": twelve language columns on Municipal Council Wards and Village Council Areas, the same
183 rows (182 drawn units) as Table D6, which religiondots draws for religion. The parse is
religiondots' sources/mu.py, re-pointed at D9 (thirteen figures a row instead of fourteen);
its docstring has the row-shape traps, all of which recur here.

THE COLUMNS ARE THE OFFICE'S OWN ONE-ANSWER-PER-PERSON ALLOCATION. The questionnaire takes up
to two home languages; Table D8 (island level only) prints every combination, and D9 files each
combination under one language. Footnote 1: "Includes the language mentioned together with its
combination with other languages." Read off D8, the rule is the first-named language of the
pair: every "Bhojpuri & X" (Bhojpuri & Creole 63,101 among them) is in Bhojpuri, every
"Creole & X" in Creole, "English & X" in English, "French & X" in French, "Hindi & X" in Hindi.
check_d8() rebuilds D9's republic and Rodrigues rows from D8 with exactly this rule, to the
person, which is what proves it.

GEO_ID IS RELIGIONDOTS' OWN. Unit rows are joined by name to religiondots'
data/normalized/mu.csv (read-only) and take its geo_id, so its mu_lookup.csv (geo_id -> unit)
applies unchanged. The join is asserted both ways, and each unit's D9 total must equal its D6
total: two tables of the same census agreeing per unit.

Usage:
    python sources/mu_hpc.py --fetch    one 4.0 MB PDF
    python sources/mu_hpc.py            normalise from data/raw/mu/
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
RD_NORM = os.path.join(os.path.dirname(ROOT), "religiondots", "data", "normalized", "mu.csv")

SOURCE_ID = "mu_hpc_2022_d9"
YEAR = 2022

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]

PDF_URL = ("https://statsmauritius.govmu.org/Documents/Census_and_Surveys/Census2022/"
           "HPC_TR_Vol2_Demography_Yr22.pdf")
PDF_NAME = "mu_hpc2022_vol2.pdf"

D9_TITLE = re.compile(r"Table D9\s*-\s*Resident population by geographical location and "
                      r"language usually spoken at home", re.I)
D8_TITLE = re.compile(r"Table D8\s*-\s*Resident population by language usually spoken at "
                      r"home and sex", re.I)

# Left to right as the header prints them (footnote marks dropped).
CATEGORIES = ["Total", "Bangla", "Bhojpuri", "Chinese languages", "Creole", "English",
              "French", "Hindi", "Marathi", "Tamil", "Telugu", "Urdu", "Other & Not stated"]
TOTAL_CAT = "Total"
NCOL = len(CATEGORIES)

# Stub indentation. NOT D6's fixed ladder: in D9 the drawn tier sits at x0 66.6 under a
# district with no urban/rural split (pages 175-177) and at 59.0 under one with a split
# (178-180), while districts are at 44.6 on every page. So a unit is "deeper than 57"; the
# unit sums and the D6 join below are what verify it.
UNIT_MIN_X = 57.0
DISTRICT_X = 44.6
X_TOL = 1.5


def _is_unit_x(x0):
    return x0 >= UNIT_MIN_X
Y_TOL = 3.0

REPUBLIC = "REPUBLIC OF MAURITIUS"
NATIONAL = 1_233_097
SPLIT_SUFFIX = re.compile(r"\s*-\s*(Urban|Rural|Wholly Urban|Wholly Rural)\s*$", re.I)
NUM = re.compile(r"^[\d,]+$")


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
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    with open(tmp, "rb") as fh:
        magic = fh.read(5)
    if magic != b"%PDF-":
        raise SystemExit(f"{tmp} is not a PDF -- starts {magic!r}")
    os.replace(tmp, dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def _rows(page):
    """(x0, name, [13 ints]) per printed row: the last thirteen tokens are the figures."""
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
        out.append((round(ws[0][0], 1), name, [int(t.replace(",", "")) for t in tail]))
    return out


def _nested_towns(raw):
    """Town rows printed at the unit indentation whose following consecutive unit rows sum to
    them exactly and share their name prefix (religiondots sources/mu.py, `_nested_towns`)."""
    units = [(i, n, v[0]) for i, (x0, n, v) in enumerate(raw) if _is_unit_x(x0)]
    parents = {}
    for k, (i, name, total) in enumerate(units):
        run = 0
        for j in range(k + 1, len(units)):
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


def _norm(name):
    """Join key: D6 and D9 print the same names with different spacing ("Grand Baie VCA- East"
    against "VCA-East"), so all whitespace goes; uniqueness is asserted on both sides."""
    return "".join(name.split()).replace("’", "'").lower()


def _rd_units():
    """religiondots' D6 unit rows: normalised name -> (geo_id, total)."""
    out = {}
    with open(RD_NORM, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["geo_level"] == "unit" and r["source_category"] == "Total":
                k = _norm(r["geo_name"])
                if k in out:
                    raise SystemExit(f"religiondots mu.csv has two units named {k!r}")
                out[k] = (r["geo_id"], int(r["count"]))
    return out


def read(doc):
    pages = [i for i in range(doc.page_count)
             if D9_TITLE.search(" ".join(doc[i].get_text().split()))]
    if not pages:
        raise SystemExit("no page carries Table D9's title -- the volume has been re-issued")
    print(f"  Table D9 on pages {[i + 1 for i in pages]}")
    raw = []
    for i in pages:
        raw.extend(_rows(doc[i]))
    nested = _nested_towns(raw)
    for town, kids in sorted(nested.items()):
        print(f"  nested tier: {town!r} is the parent of {len(kids)} wards")

    rd = _rd_units()
    out, seen, district, matched = [], set(), None, set()
    for x0, name, vals in raw:
        is_unit = _is_unit_x(x0) and name not in nested
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
        if abs(x0 - DISTRICT_X) <= X_TOL and not name.upper().startswith("ISLAND OF MAURITIUS"):
            district = name
        if level != "unit" and SPLIT_SUFFIX.search(name) and "DISTRICT-Wholly" not in name:
            level += "_split"
        key = (level, name)
        if key in seen:
            raise SystemExit(f"duplicate row {key}")
        seen.add(key)
        if level == "unit":
            k = _norm(name)
            if k not in rd:
                raise SystemExit(f"D9 unit {name!r} has no D6 row in religiondots' mu.csv")
            if k in matched:
                raise SystemExit(f"two D9 units join to one D6 unit: {name!r}")
            code, d6_total = rd[k]
            if d6_total != vals[0]:
                raise SystemExit(f"{name}: D9 total {vals[0]:,} != D6 total {d6_total:,}")
            matched.add(k)
        else:
            code = f"{level[:1].upper()}9{len(seen):03d}"
        for cat, n in zip(CATEGORIES, vals):
            note = f"level={level}; x0={x0}"
            if cat == TOTAL_CAT:
                note += "; universe total, not a language"
            if level == "unit":
                note += f"; district={district}"
            out.append({"geo_id": code, "geo_level": level, "geo_name": name,
                        "source_category": cat, "count": n, "tier": "measured",
                        "year": YEAR, "source_id": SOURCE_ID, "note": note})
    missing = sorted(set(rd) - matched)
    if missing:
        raise SystemExit(f"D6 units with no D9 row: {missing}")
    print(f"  name join to religiondots' D6 units: {len(matched)} of {len(rd)} both ways, "
          "and every unit's D9 total equals its D6 total")
    return out


# D8 label -> the D9 column it lands in, by the first-named rule (module docstring).
def d8_to_d9(label):
    first = label.split("&")[0].strip()
    if first in ("Cantonese", "Chinese", "Hakka", "Mandarin", "Other Chinese"):
        return "Chinese languages"
    if first in ("Creole", "Bhojpuri", "English", "French", "Hindi"):
        return first
    if "&" not in label and first in ("Bangla", "Marathi", "Tamil", "Telugu", "Urdu"):
        return first
    return "Other & Not stated"


def read_d8(doc):
    """{block: {label: both sexes}} for the three blocks of Table D8."""
    pages = [i for i in range(doc.page_count)
             if D8_TITLE.search(" ".join(doc[i].get_text().split()))]
    blocks, cur = {}, None
    for i in pages:
        lines = [l.strip() for l in doc[i].get_text().split("\n") if l.strip()]
        k = 0
        while k < len(lines):
            l = lines[k]
            if l in (REPUBLIC, "ISLAND OF MAURITIUS", "ISLAND OF RODRIGUES"):
                cur = l
                blocks[cur] = {}
                k += 1
                continue
            nxt = lines[k + 1:k + 4]
            if (cur and not NUM.match(l) and len(nxt) == 3 and all(NUM.match(t) for t in nxt)):
                both, m, f = (int(t.replace(",", "")) for t in nxt)
                if both != m + f:
                    raise SystemExit(f"D8 {cur} {l}: {both} != {m} + {f}")
                blocks[cur][" ".join(l.split())] = both
                k += 4
                continue
            k += 1
    return blocks


def check_d8(d8, rows):
    ok = True
    for block, level_name in ((REPUBLIC, REPUBLIC), ("ISLAND OF RODRIGUES",
                                                     "ISLAND OF RODRIGUES - Wholly Rural")):
        lab = d8[block]
        total = lab.pop("All languages")
        if sum(lab.values()) != total:
            print(f"  BAD D8 {block}: labels sum {sum(lab.values()):,} != {total:,}")
            ok = False
        built = {}
        for l, n in lab.items():
            built[d8_to_d9(l)] = built.get(d8_to_d9(l), 0) + n
        d9 = {r["source_category"]: r["count"] for r in rows
              if r["geo_name"].strip() == level_name}
        if not d9:
            print(f"  BAD no D9 row named {level_name!r}")
            ok = False
            continue
        bad = [(c, built.get(c, 0), d9[c]) for c in CATEGORIES[1:] if built.get(c, 0) != d9[c]]
        print(f"  {'OK ' if not bad else 'BAD'} D8 {block} ({len(lab)} labels) rebuilds D9 "
              f"by the first-named rule on all {NCOL - 1} columns")
        for b in bad:
            print("       ", b)
        ok &= not bad
        lab["All languages"] = total
    return ok


def check(rows):
    ok = True
    levels = {}
    for r in rows:
        levels.setdefault(r["geo_level"], set()).add(r["geo_name"])
    for lv in sorted(levels):
        print(f"      {lv:<16} {len(levels[lv]):>4} rows")
    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"
           and not SPLIT_SUFFIX.search(r["geo_name"])}
    good = nat.get(TOTAL_CAT) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat.get(TOTAL_CAT):,}")
    by_row = {}
    for r in rows:
        by_row.setdefault((r["geo_level"], r["geo_name"]), {})[r["source_category"]] = r["count"]
    bad = [k for k, d in by_row.items() if sum(v for c, v in d.items() if c != TOTAL_CAT)
           != d[TOTAL_CAT]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 12 columns sum to Total on all {len(by_row)} "
          f"rows")
    for k in bad[:5]:
        print("       ", k)
    bad = []
    for cat in CATEGORIES:
        s = sum(r["count"] for r in rows if r["geo_level"] == "unit"
                and r["source_category"] == cat)
        if s != nat[cat]:
            bad.append((cat, s, nat[cat]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the {len(levels.get('unit', ()))} units sum to the "
          f"republic on all {NCOL} columns")
    for b in bad:
        print("       ", b)
    print("\n  national:")
    for cat in CATEGORIES:
        print(f"    {nat[cat]:>10,}  {100.0 * nat[cat] / NATIONAL:6.2f}%  {cat}")
    return ok


def main():
    import fitz
    if "--fetch" in sys.argv:
        fetch()
    p = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    doc = fitz.open(p)
    rows = read(doc)
    ok = check(rows)
    d8 = read_d8(doc)
    print("\n  Table D8 blocks:", {k: len(v) for k, v in d8.items()})
    ok &= check_d8(d8, rows)
    if not ok:
        raise SystemExit("reconciliation FAILED")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
