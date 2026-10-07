"""Serbia: RZS, Popis stanovnistva 2022, mother tongue by municipality.

    python sources/rs_census.py --fetch    download the two workbooks (0.6 MB) if missing
    python sources/rs_census.py            normalise from data/raw/rs/

-> data/normalized/rs.csv (levels `country`, `half`, `region`, `oblast`, `city`, `municipality`;
   alternatives, never summed; only `municipality` is drawn)

Two workbooks from the census results portal's Excel page
(https://popis2022.stat.gov.rs/sr-latn/popisni-podaci-eksel-tabele/):

  7_stanovnistvo-prema-maternjem-jeziku.xls   mother tongue by municipality and city, sheet
      `opstine`: 18 named languages, `Other languages`, `Did not declare`, `Unknown`, each by
      sex and by settlement type (urban / other). The drawn table.
  1_stanovnistvo-prema-nacionalnoj-pripadnosti-i-maternjem-jeziku.xlsx   ethnicity x mother
      tongue by region, the same 21 categories. The second-table check.

THE LAYOUT IS RELIGIONDOTS' RELIGION WORKBOOK (religiondots/sources/rs.py), with one more row
type, so the parsing follows it exactly and produces the SAME geo_ids, which are the `unit` keys
of religiondots' rs_grid_400m.gpkg:
  1. Six levels share column 0 with no codes and no level column: republic, two halves, four
     regions, 25 oblasti, municipalities. Each unit is a block of rows: its total, then male and
     female, then `Градска` (urban) and `Остала` (other settlements) each with their own sexes.
     Only the unit's own total row (sex `с` / `t`) is kept.
  2. `Grad Niš`, `Grad Požarevac`, `Grad Užice`, `Grad Vranje` sit in the municipality tier as
     parents of their own city municipalities. Their children are the following rows whose
     totals sum to them exactly; the four parents are re-levelled `city` and not drawn.
  3. No codes and Palilula twice (Belgrade's and Niš's), so city municipalities are keyed
     `<city> - <name>` and every other municipality by its bare name, as religiondots keys them.
  4. Kosovo is a region row of `...` (not enumerated). Dropped, and the check says so.

CHECKS: national total 6,647,003; each level sums to the national row per category; the 21
categories partition every municipality; every municipality's total equals religiondots' total
for the same unit from the religion table of the same census (a join check: the units are the
same 168, keyed the same way); and the municipalities summed by region against the second
workbook's four region rows, per category.
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "rs")
OUT = os.path.join(ROOT, "data", "normalized", "rs.csv")
RD_NORM = os.path.join(os.path.dirname(ROOT), "religiondots", "data", "normalized", "rs.csv")

SOURCE_ID = "rs_popis_2022_mt"
YEAR = 2022

BASE = "https://popis2022.stat.gov.rs/media/"
FILES = {
    "mother_tongue_municipality.xls": ("31330/7_stanovnistvo-prema-maternjem-jeziku.xls", 400_000),
    "ethnicity_mother_tongue_region.xlsx": (
        "31348/1_stanovnistvo-prema-nacionalnoj-pripadnosti-i-maternjem-jeziku.xlsx", 20_000),
}
XLS = os.path.join(RAW, "mother_tongue_municipality.xls")
REGION_XLSX = os.path.join(RAW, "ethnicity_mother_tongue_region.xlsx")

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]

NATIONAL = 6_647_003           # resident population enumerated, Popis 2022

TOTAL_LABEL = "Total"
# Column -> (first word of the Cyrillic header, the label written to rs.csv). The label is the
# English half of RZS's bilingual header with whitespace and a line-break hyphen tidied
# ("Other langu-ages"). Asserted against both workbooks, so a re-cut sheet fails loudly.
CATEGORY_COLS = {
    2: ("Укупно", TOTAL_LABEL),
    3: ("Српски", "Serbian"),
    4: ("Албански", "Albanian"),
    5: ("Босански", "Bosnian"),
    6: ("Бугарски", "Bulgarian"),
    7: ("Буњевачки", "Bunjevački"),
    8: ("Влашки", "Vlach language"),
    9: ("Мађарски", "Hungarian"),
    10: ("Македонски", "Macedonian"),
    11: ("Немачки", "German"),
    12: ("Ромски", "Roma language"),
    13: ("Румунски", "Romanian"),
    14: ("Руски", "Russian"),
    15: ("Русински", "Ruthenian"),
    16: ("Словачки", "Slovak"),
    17: ("Словеначки", "Slovenian"),
    18: ("Украјински", "Ukrainian"),
    19: ("Хрватски", "Croatian"),
    20: ("Црногорски", "Montenegrin"),
    21: ("Остали", "Other languages"),
    22: ("Нису", "Did not declare"),
    23: ("Непознато", "Unknown"),
}
NAME_COL_SR, SEX_COL_SR, SEX_COL_EN, NAME_COL_EN = 0, 1, 24, 25
SETTLEMENT_TYPE_ROWS = {"Градска", "Остала"}     # urban / other subtotals inside each unit
NO_PHENOMENON = "-"
NOT_AVAILABLE = "..."

EXPECTED = {"country": 1, "half": 2, "region": 4, "oblast": 25, "city": 4, "municipality": 168}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, (path, min_bytes) in FILES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > min_bytes:
            print("already have", dest)
            continue
        print("GET", BASE + path)
        r = requests.get(BASE + path, timeout=180,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        r.raise_for_status()
        magic = b"\xd0\xcf\x11\xe0" if name.endswith(".xls") else b"PK"
        if not r.content.startswith(magic) or len(r.content) < min_bytes:
            raise SystemExit(f"{name}: not a workbook ({len(r.content):,} bytes, "
                             f"starts {r.content[:16]!r})")
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(r.content)
        os.replace(tmp, dest)
        print(f"  {os.path.getsize(dest):,} bytes")


def _txt(v):
    return " ".join(str(v).split())


def _level(en, sr):
    s = en or sr
    u = s.upper()
    if "REPUBLIC" in u or "РЕПУБЛИКА" in u:
        return "country"
    if u.startswith("SRBIJA") or u.startswith("СРБИЈА"):
        return "half"
    if re.search(r"\bregion\b", s, re.I) or s.startswith("Регион"):
        return "region"
    if "oblast" in s.lower() or "област" in s.lower():
        return "oblast"
    return "municipality"


def _check_header(cells, where):
    for col, (cyr, label) in CATEGORY_COLS.items():
        got = _txt(cells[col]).replace("-", "")       # "Маке-донски": a line-break hyphen
        if not got.startswith(cyr):
            raise SystemExit(f"{where}, column {col}: header reads {got!r}, expected it to "
                             f"start {cyr!r} ({label}); the column map is stale")


def _value(v, where):
    s = _txt(v)
    if s == NO_PHENOMENON or s == "":
        return 0
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise SystemExit(f"{where}: cannot read {v!r} as a count")
    if f != int(f):
        raise SystemExit(f"{where}: {v!r} is not a whole number")
    return int(f)


def _nest(records):
    """religiondots/sources/rs.py's _nest(): city of each municipality, and re-level the four
    `Grad X` parents to `city`. A parent's children are the consecutive rows summing to it."""
    i = 0
    while i < len(records):
        rec = records[i]
        if rec["level"] == "oblast":
            m = re.search(r"\(Grad ([^)]+)\)", rec["en"] or "")
            city = m.group(1).strip() if m else None
            j = i + 1
            while j < len(records) and records[j]["level"] == "municipality":
                if city:
                    records[j]["city"] = city
                j += 1
            i += 1
            continue
        if rec["level"] == "municipality" and (rec["en"] or "").startswith("Grad "):
            city = rec["en"][len("Grad "):].strip()
            want = rec["values"][TOTAL_LABEL]
            run, got, j = [], 0, i + 1
            while j < len(records) and records[j]["level"] == "municipality" \
                    and not (records[j]["en"] or "").startswith("Grad ") and got < want:
                got += records[j]["values"][TOTAL_LABEL]
                run.append(records[j])
                j += 1
            if got != want:
                raise SystemExit(f"{rec['en']}: its {len(run)} following rows sum to {got:,}, "
                                 f"not {want:,}; the city nesting is not what _nest() assumes")
            rec["level"] = "city"
            rec["parent_of"] = len(run)
            for child in run:
                child["city"] = city
            i = j
            continue
        i += 1
    return records


def read():
    import xlrd

    if not os.path.exists(XLS):
        raise SystemExit(f"missing {XLS}; run with --fetch first")
    sh = xlrd.open_workbook(XLS).sheet_by_name("opstine")
    _check_header(sh.row_values(1), "opstine header")

    records, empty, region = [], [], None
    for r in range(2, sh.nrows):
        sr = _txt(sh.cell_value(r, NAME_COL_SR))
        if not sr:
            continue
        if sr.startswith("Списак ознака"):
            break                                   # the symbols legend at the foot
        if sr in SETTLEMENT_TYPE_ROWS:
            continue                                # urban / other subtotal of the unit above
        all_na = all(_txt(sh.cell_value(r, c)) == NOT_AVAILABLE for c in CATEGORY_COLS)
        is_total = (_txt(sh.cell_value(r, SEX_COL_EN)) == "t"
                    or _txt(sh.cell_value(r, SEX_COL_SR)) == "с")
        if not is_total and not all_na:
            continue
        en = _txt(sh.cell_value(r, NAME_COL_EN)) or None
        level = _level(en, sr)
        if all_na:
            empty.append(dict(row=r, sr=sr, en=en, level=level))
            continue
        values = {label: _value(sh.cell_value(r, c), f"row {r} col {c}")
                  for c, (_, label) in CATEGORY_COLS.items()}
        if level == "region":
            region = en
        records.append(dict(row=r, sr=sr, en=en, level=level, values=values, city=None,
                            parent_of=0, region=region))

    _nest(records)

    rows = []
    for rec in records:
        if rec["level"] == "municipality" and rec["city"]:
            geo_id = f"{rec['city']} - {rec['en']}"
        else:
            geo_id = rec["en"] or rec["sr"]
        rec["geo_id"] = geo_id
        bits = [f"level={rec['level']}"]
        if rec["city"]:
            bits.append(f"city municipality of {rec['city']}")
        if rec["parent_of"]:
            bits.append(f"parent of {rec['parent_of']} city municipalities, not drawn")
        if rec["level"] not in ("country", "half", "region"):
            bits.append(f"region={rec['region']}")
        for label, n in rec["values"].items():
            note = "; ".join(bits)
            if label == TOTAL_LABEL:
                note += "; universe total, not a language"
            rows.append(dict(geo_id=geo_id, geo_level=rec["level"], geo_name=rec["sr"],
                             source_category=label, count=n, tier="measured", year=YEAR,
                             source_id=SOURCE_ID, note=note))
    return rows, records, empty


def read_regions():
    """The second workbook's four region rows: {region total: {label: count}}, in order."""
    import pandas as pd

    df = pd.read_excel(REGION_XLSX, header=None)
    _check_header([""] + df.iloc[1].tolist(), "region workbook header")   # one column fewer
    out = []
    for r in range(2, len(df)):
        raw = str(df.iat[r, 0])
        sr = _txt(raw)
        # Region rows sit flush left (`Београдски регион`, `Регион Војводине`); ethnicity rows
        # are indented, and one of them, `Регионална припадност`, also starts `Регион`.
        if raw[:1].isspace() or "регион" not in sr.lower():
            continue
        cells = [df.iat[r, c - 1] for c in CATEGORY_COLS]
        if all(_txt(v) == NOT_AVAILABLE for v in cells):
            continue                                 # Kosovo
        vals = {label: _value(df.iat[r, c - 1], f"region row {r}")
                for c, (_, label) in CATEGORY_COLS.items()}
        out.append((sr, vals))
    return out


def check(rows, records, empty):
    ok = True
    print(f"  {len(empty)} row(s) present but entirely '...':")
    for x in empty:
        print(f"      {x['level']:<12} {x['sr']}  /  {x['en']}")
    good = len(empty) == 1 and empty[0]["level"] == "region"
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} exactly one empty row, a region (Kosovo and Metohija, "
          "not enumerated)")

    levels = {}
    for rec in records:
        levels.setdefault(rec["level"], []).append(rec)
    for lv, want in EXPECTED.items():
        got = len(levels.get(lv, ()))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<13} {got:>4} rows (expected {want})")

    nat = levels["country"][0]["values"]
    good = nat[TOTAL_LABEL] == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} national total {nat[TOTAL_LABEL]:,} "
          f"(published {NATIONAL:,})")

    for lv in ("half", "region", "oblast", "municipality"):
        bad = [(k, sum(r["values"][k] for r in levels[lv]), v) for k, v in nat.items()
               if sum(r["values"][k] for r in levels[lv]) != v]
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} all {len(nat)} columns sum from "
              f"{len(levels[lv]):>3} {lv} rows to the national row")
        for k, s, v in bad[:6]:
            print(f"        {k}: {s:,} vs {v:,}")

    muni = levels["municipality"]
    bad = [r["en"] for r in muni
           if sum(v for k, v in r["values"].items() if k != TOTAL_LABEL) != r["values"][TOTAL_LABEL]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 21 categories partition every one of the "
          f"{len(muni)} municipalities")
    for name in bad[:6]:
        print(f"        {name}")

    ids = [r["geo_id"] for r in muni]
    good = len(set(ids)) == len(ids)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(set(ids))} distinct municipality ids "
          "(Palilula twice, keyed by city)")

    # Join check: religiondots' religion table is the same census on the same 168 units, keyed
    # by the same rules. Every unit must exist there with exactly the same total population.
    if os.path.exists(RD_NORM):
        rd = {}
        with open(RD_NORM, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["geo_level"] == "municipality" and row["source_category"] == "Total":
                    # religiondots keeps RZS's double space in `Stari  grad`; this file does
                    # not, so compare with whitespace collapsed (countries/rs.py does the same)
                    rd[_txt(row["geo_id"])] = int(row["count"])
        mine = {r["geo_id"]: r["values"][TOTAL_LABEL] for r in muni}
        miss = sorted(set(mine) ^ set(rd))
        diff = [(k, mine[k], rd[k]) for k in mine if k in rd and mine[k] != rd[k]]
        good = not miss and not diff
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} all {len(mine)} municipality ids match religiondots' "
              f"rs.csv both ways, with identical totals in every unit")
        for k in miss[:6]:
            print(f"        only on one side: {k}")
        for k, a, b in diff[:6]:
            print(f"        {k}: {a:,} here, {b:,} in the religion table")
    else:
        ok = False
        print(f"  BAD religiondots' {RD_NORM} is missing, so the join is unchecked")

    # Second table: ethnicity x mother tongue by region. Its region rows must equal this
    # workbook's municipalities summed by region, in every category.
    if os.path.exists(REGION_XLSX):
        theirs = read_regions()
        ours = {}
        for r in muni:
            acc = ours.setdefault(r["region"], {k: 0 for k in nat})
            for k, v in r["values"].items():
                acc[k] += v
        regions = [r["en"] for r in levels["region"]]
        good = len(theirs) == len(regions) == 4
        bad = []
        for (sr, vals), en in zip(theirs, regions):
            for k, v in vals.items():
                if ours[en][k] != v:
                    bad.append((en, k, ours[en][k], v))
        good &= not bad
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} municipalities summed by region equal the "
              f"ethnicity x mother tongue table's {len(theirs)} region rows in all "
              f"{len(nat)} columns")
        for en, k, a, b in bad[:6]:
            print(f"        {en} / {k}: {a:,} here, {b:,} there")
    else:
        ok = False
        print(f"  BAD {REGION_XLSX} missing; run --fetch")

    print("\n  categories, national:")
    for k, v in sorted(nat.items(), key=lambda kv: -kv[1]):
        print(f"    {v:>10,}  {100.0 * v / NATIONAL:6.2f}%  {k}")
    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, records, empty = read()
    check(rows, records, empty)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
