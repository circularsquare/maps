"""Serbia — Republički zavod za statistiku, Popis 2022, religion by municipality.

Reads (or fetches) data/raw/rs/ and writes data/normalized/rs.csv.

One workbook, `6_stanovnistvo-prema-veroispovesti.xlsx` from the census results portal —
13 religion categories x 168 municipalities, 6,647,003 people, no API needed and no
suppression anywhere in it. Croatia's shape almost exactly (§9e), one country over.

**IT IS FIVE NESTED LEVELS IN ONE SHEET, AND THE MUNICIPALITY ROWS ARE NOT ALL THE SAME
TIER.** Republic -> two halves (Srbija-sever / Srbija-jug) -> 4 regions -> 25 oblasti ->
municipalities, all in column 0 with no level column and no indentation to read. Summing
the sheet as delivered counts the country five times over. Worse, the municipality tier
contains FOUR PARENTS OF ITS OWN — `Grad Niš`, `Grad Požarevac`, `Grad Užice` and
`Grad Vranje` are each followed by their own city municipalities, so a sixth copy of
462,527 people is hiding inside the finest level. `_nest()` resolves both, and the test is
arithmetic: a `Grad X` row's children are the consecutive rows that sum to it exactly.

**PALILULA IS TWO DIFFERENT PLACES AND THE SHEET NAMES THEM IDENTICALLY.** Belgrade has a
Palilula (69,113) and so does Niš (69,811), 200 km apart, and the source publishes no codes
of any kind — only names, as Romania and Ghana do. Ghana's `TMA` again, in a shape that
looks friendlier because the string is a real place name rather than an acronym. Resolved
the same way: the name is unique only WITHIN its parent city, so city municipalities carry
their city in the geo_id and everything else carries a bare name.

**KOSOVO IS IN THE SHEET AND IS EMPTY.** `Регион Косовo и Метохија` is present as a region
row whose every cell is `...` — RZS's symbol for data not available — because the census
did not enumerate it. The row is dropped and counted, rather than being silently absent:
see `check()`. Nothing on this map draws Kosovo from a Serbian source.

The Christian column is Ghana's case, not Hungary's: `Хришћанска / свега` sits beside its
four children and equals their sum **in every one of the 204 rows**, so it is a duplicate
to drop rather than a parent needing a remainder (§12). Checked per row, not nationally.

Usage:
    python sources/rs.py --fetch    one GET, 83 KB
    python sources/rs.py            normalise from data/raw/rs/
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

SOURCE_ID = "rs_popis_2022"
YEAR = 2022
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

URL = ("https://popis2022.stat.gov.rs/media/31329/"
       "6_stanovnistvo-prema-veroispovesti.xlsx")
XLSX = os.path.join(RAW, "religion_municipality.xlsx")
SHEET = "OpstinePol"

NATIONAL = 6_647_003          # resident population enumerated, Popis 2022

EXPECTED_MUNICIPALITIES = 168
EXPECTED_OBLASTI = 25
EXPECTED_REGIONS = 4          # Kosovo's fifth is present and empty

# Column -> the English half of the two-row header. Asserted against the sheet in read(),
# so a re-cut workbook fails loudly instead of quietly shifting a column.
CATEGORY_COLS = {
    2:  ("Total", None),
    3:  ("All", "Christian"),
    4:  ("Orthodox", "Christian"),
    5:  ("Catholic", "Christian"),
    6:  ("Protestant", "Christian"),
    7:  ("Other Christian", "Christian"),
    8:  ("Islam", None),
    9:  ("Judaism", None),
    10: ("Eastern religions", None),
    # RZS's own header cell reads `Otherreligions`, unspaced. Kept as the source wrote it
    # (§2.4) so the taxonomy key matches the file rather than a tidied version.
    11: ("Otherreligions", None),
    12: ("Agnostics", None),
    13: ("Not believers (atheists)", None),
    14: ("Did not declare", None),
    15: ("Unknown", None),
}
TOTAL_LABEL = "Total"
CHRISTIAN_ALL = "Christian - All"

NAME_COL_SR, NAME_COL_EN, SEX_COL_SR, SEX_COL_EN = 0, 17, 1, 16
NO_PHENOMENON = "-"           # RZS symbol: zero
NOT_AVAILABLE = "..."         # RZS symbol: data not available (Kosovo)


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(XLSX) and os.path.getsize(XLSX) > 40_000:
        print("already have", XLSX)
        return
    print("GET", URL)
    r = requests.get(URL, timeout=180,
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    r.raise_for_status()
    # §5a: HTTP 200 is not a download. An xlsx is a zip and starts 'PK'.
    if r.content[:2] != b"PK":
        raise SystemExit(f"not a workbook: first bytes {r.content[:40]!r}")
    with open(XLSX, "wb") as fh:
        fh.write(r.content)
    print(f"  {os.path.getsize(XLSX):,} bytes")


def _txt(v):
    return " ".join(str(v).split()) if isinstance(str(v), str) else ""


def _last_line(v):
    """The English half of a bilingual header cell is its last non-empty line."""
    parts = [p.strip() for p in str(v).split("\n") if p.strip()]
    return " ".join(parts[-1].split()) if parts else ""


def _level(en, sr):
    s = en or sr
    u = s.upper()
    if "REPUBLIC" in u or "РЕПУБЛИКА" in u:
        return "country"
    if u.startswith("SRBIJA") or u.startswith("СРБИЈА"):
        return "half"
    if re.search(r"\bregion\b", s, re.I) or s.startswith("Регион") or s.startswith("Region"):
        return "region"
    if "oblast" in s.lower() or "област" in s.lower():
        return "oblast"
    return "municipality"


def _read_sheet():
    import pandas as pd

    if not os.path.exists(XLSX):
        raise SystemExit(f"missing {XLSX} -- run with --fetch first")
    df = pd.read_excel(XLSX, sheet_name=SHEET, header=None)

    # ---- the header is two rows deep and the columns must be what we think they are ----
    for col, (want, family) in CATEGORY_COLS.items():
        cell = df.iat[2, col] if isinstance(df.iat[2, col], str) else df.iat[1, col]
        got = _last_line(cell)
        if got != want:
            raise SystemExit(
                f"column {col}: header reads {got!r}, expected {want!r} -- RZS has "
                "re-cut the workbook and the column map in CATEGORY_COLS is stale")
    # The family header is one line with a slash — `Хришћанска / Christian` — where every
    # other bilingual cell in this sheet stacks its two languages on separate lines.
    family = _txt(df.iat[1, 3]).split("/")[-1].strip()
    if family != "Christian":
        raise SystemExit(f"column 3's family header is {df.iat[1, 3]!r}, not Christian")
    return df


def _cells(df, r):
    """One row's category values, or None if the whole row is `...` (Kosovo)."""
    out, seen_na = {}, 0
    for col, (label, family) in CATEGORY_COLS.items():
        v = df.iat[r, col]
        key = f"{family} - {label}" if family else label
        s = _txt(v)
        if s == NOT_AVAILABLE:
            seen_na += 1
            continue
        if s == NO_PHENOMENON or s == "":
            out[key] = 0
            continue
        try:
            out[key] = int(float(v))
        except (TypeError, ValueError):
            raise SystemExit(f"row {r}, column {col}: cannot read {v!r} as a count")
    if seen_na == len(CATEGORY_COLS):
        return None
    if seen_na:
        raise SystemExit(f"row {r}: {seen_na} cells are '...' but not all of them")
    return out


def _nest(records):
    """Assign every municipality row its parent city, where it has one.

    Two shapes, and both are in this sheet:

      * `Beogradska oblast (Grad Beograd)` is an OBLAST whose 17 children are city
        municipalities — Belgrade is a city that fills a whole oblast;
      * `Grad Niš`, `Grad Požarevac`, `Grad Užice`, `Grad Vranje` are MUNICIPALITY-level
        rows followed by their own city municipalities, inside an oblast that holds
        ordinary municipalities as well.

    The second is the one with no structural marker at all: after Niš's five, the next row
    is Aleksinac, a plain municipality of the same oblast, at the same indent and in the
    same column. **The test is arithmetic** — a `Grad X` row's children are the consecutive
    rows whose totals sum to it exactly — which is also a check that the sheet parsed.

    Returns records with `city` and `parent_of` filled, and the four `Grad X` rows
    re-levelled to `city` so they are dropped from the drawn tier.
    """
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
                raise SystemExit(
                    f"{rec['en']}: its {len(run)} following rows sum to {got:,}, "
                    f"not {want:,} -- the city/municipality nesting is not what "
                    "_nest() assumes")
            rec["level"] = "city"
            rec["parent_of"] = len(run)
            for child in run:
                child["city"] = city
            i = j
            continue
        i += 1
    return records


def read():
    df = _read_sheet()

    records = []
    for r in range(3, len(df)):
        sr = df.iat[r, NAME_COL_SR]
        if not isinstance(sr, str) or not sr.strip():
            continue
        if sr.strip().startswith("Списак ознака"):
            break                        # the symbols legend at the foot of the sheet
        is_total = (_txt(df.iat[r, SEX_COL_EN]) == "t"
                    or _txt(df.iat[r, SEX_COL_SR]) == "с")
        # Kosovo's region row carries no sex breakdown at all, because it carries no data:
        # every cell is `...`. It must not be filtered out with the male/female rows —
        # a source that publishes a unit as EMPTY is saying something, and `check()`
        # asserts on it.
        all_na = all(_txt(df.iat[r, c]) == NOT_AVAILABLE for c in CATEGORY_COLS)
        if not is_total and not all_na:
            continue
        en = df.iat[r, NAME_COL_EN]
        en = en.strip() if isinstance(en, str) else None
        sr = " ".join(sr.split())
        values = _cells(df, r)
        if values is None:
            records.append(dict(row=r, sr=sr, en=en, level=_level(en, sr),
                                values=None, city=None, parent_of=0))
            continue
        records.append(dict(row=r, sr=sr, en=en, level=_level(en, sr),
                            values=values, city=None, parent_of=0))

    empty = [x for x in records if x["values"] is None]
    live = [x for x in records if x["values"] is not None]
    _nest(live)

    rows = []
    for rec in live:
        if rec["level"] == "municipality" and rec["city"]:
            geo_id = f"{rec['city']} - {rec['en']}"
        else:
            geo_id = rec["en"] or rec["sr"]
        note_bits = [f"level={rec['level']}"]
        if rec["city"]:
            note_bits.append(f"city municipality of {rec['city']}")
        if rec["parent_of"]:
            note_bits.append(f"parent of {rec['parent_of']} city municipalities, "
                             "dropped from the drawn tier")
        for label, n in rec["values"].items():
            note = "; ".join(note_bits)
            if label == TOTAL_LABEL:
                note += "; universe total, not a religion category"
            elif label == CHRISTIAN_ALL:
                note += ("; parent of the four Christian categories, published beside "
                         "them and equal to their sum")
            rows.append({"geo_id": geo_id, "geo_level": rec["level"],
                         "geo_name": rec["sr"], "source_category": label,
                         "count": n, "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID, "note": note})
    return rows, live, empty


def check(rows, live, empty):
    ok = True

    print(f"  {len(empty)} row(s) present but entirely '...' (data not available):")
    for x in empty:
        print(f"      {x['level']:<12} {x['sr']}  /  {x['en']}")
    good = len(empty) == 1 and empty[0]["level"] == "region"
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} exactly one empty row and it is a region "
          "(Kosovo and Metohija, not enumerated)")

    levels = {}
    for rec in live:
        levels.setdefault(rec["level"], []).append(rec)
    for lv, want in (("country", 1), ("half", 2), ("region", EXPECTED_REGIONS),
                     ("oblast", EXPECTED_OBLASTI), ("city", 4),
                     ("municipality", EXPECTED_MUNICIPALITIES)):
        got = len(levels.get(lv, ()))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<13} {got:>4} rows (expected {want})")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat.get(TOTAL_LABEL) == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} national total {nat.get(TOTAL_LABEL):,} "
          f"(published {NATIONAL:,})")

    # Every level is a partition of the same country, so each must sum to the national row
    # exactly -- RZS neither rounds nor suppresses this table. `city` is the one level that
    # is NOT a partition: it is four rows, and it is excluded here and dropped downstream.
    for lv in ("half", "region", "oblast", "municipality"):
        bad = []
        for label in sorted(nat):
            s = sum(r["count"] for r in rows
                    if r["geo_level"] == lv and r["source_category"] == label)
            if s != nat[label]:
                bad.append((label, s, nat[label]))
        good = not bad
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} all {len(nat)} categories sum from "
              f"{len(levels[lv]):>3} {lv} rows to the national row")
        for label, s, want in bad[:6]:
            print(f"        {label}: {s:,} vs {want:,}")

    # §3.2: the categories must partition each unit, in EVERY unit and not just nationally.
    drawn = [rec for rec in live if rec["level"] == "municipality"]
    bad = [rec for rec in drawn
           if sum(v for k, v in rec["values"].items()
                  if k not in (TOTAL_LABEL, CHRISTIAN_ALL)) != rec["values"][TOTAL_LABEL]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 11 drawn categories partition every one "
          f"of the {len(drawn)} municipalities")
    for rec in bad[:6]:
        print(f"        {rec['en']}")

    # §12's Ghana/Hungary test, PER ROW: does the published Christian parent equal the sum
    # of its four published children? If it does everywhere it is a duplicate to drop; if
    # it exceeds them anywhere there is a remainder that has to be emitted.
    kids = ["Christian - Orthodox", "Christian - Catholic",
            "Christian - Protestant", "Christian - Other Christian"]
    off = [(rec["en"], rec["values"][CHRISTIAN_ALL] - sum(rec["values"][k] for k in kids))
           for rec in live if rec["values"][CHRISTIAN_ALL] != sum(rec["values"][k] for k in kids)]
    ok &= not off
    print(f"  {'OK ' if not off else 'BAD'} `Christian - All` equals its four children "
          f"in all {len(live)} rows, so it is a duplicate and not a parent with a "
          "remainder (spec §12)")
    for name, d in off[:6]:
        print(f"        {name}: parent exceeds children by {d:,}")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for label, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        mark = ""
        if label == TOTAL_LABEL:
            mark = "  <- universe"
        elif label == CHRISTIAN_ALL:
            mark = "  <- duplicate of its four children"
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:5.2f}%  {label}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, live, empty = read()
    check(rows, live, empty)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
