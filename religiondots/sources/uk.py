"""United Kingdom — four censuses, four statistical systems, one file.

Writes data/normalized/uk.csv.  The UK does not have a census; it has four, run
by three agencies on two different dates with four different religion questions.
They are kept apart by `source_id` and never summed into a UK total here.

  uk_ew_census_2021               ONS,   England & Wales, 21 Mar 2021
  uk_sc_census_2022               NRS,   Scotland,        20 Mar 2022
  uk_ni_census_2021               NISRA, Northern Ireland, 21 Mar 2021
                                  -- the RELIGION question (MS-B19)
  uk_ni_census_2021_brought_up_in NISRA, Northern Ireland, 21 Mar 2021
                                  -- "religion OR RELIGION BROUGHT UP IN" (MS-B23)

**The two Northern Ireland questions get different source_ids on purpose.**  They
are different variables over the same people and both publish a category called
"Catholic" at the same geography, so anything keyed on (source_id, geo_level)
alone would silently mix them.  MS-B19 asks what you belong to now; MS-B23 falls
back to the religion you were brought up in when you belong to none or did not
answer.  NI's sectarian headline figures are always MS-B23.  See sources/uk.md.

England & Wales is spec.md 3.4's shape: TS030 has the top-level categories at
OUTPUT AREA (188,880 units) but only 9 of them; TS031 adds 50 write-in
sub-categories (Alevi, Jain, Pagan, Zoroastrian, Rastafarian, Yazidi, ...) but
stops at MSOA (7,264 units).  Both are emitted, at their own geographies, and
neither is rescaled to the other here.

Northern Ireland has the same shape one level down: MS-B19 (8 categories) at Data
Zone, MS-B20 (32) at Local Government District.

TS031 is HIERARCHICAL: "Other religion: Pagan" is a child of "Other religion",
and "No religion: Humanist" a child of "No religion".  The parent path is kept in
`source_category` verbatim, so summing every TS031 row for a unit double counts.
Take either the top-level set or the leaf set.

Rows with count 0 are dropped, except the per-unit total, which is always kept so
that every unit appears.  Categories are left exactly as the source writes them
(spec.md 2.4); only the dimension-name prefix ONS bolts onto its column headers
("Religion: ", "Religion (detailed): ") is stripped.

Re-fetch: see sources/uk.md.
"""

import csv
import io
import os
import re
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "uk")
OUT = os.path.join(ROOT, "data", "normalized", "uk.csv")

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASIS = "self_id"  # all four are census self-declaration

# Nomis bulk files carry every geography in one zip; these are the ones taken,
# chosen so that no two ONS tables land on the same geo_level.
TS030_LEVELS = {"oa": "output_area"}
TS031_LEVELS = {"msoa": "msoa", "ltla": "ltla", "ctry": "country"}

# "England and Wales" is the sum of the other two rows in the same file.
EW_COMBINED = "K04000001"


# --------------------------------------------------------------- England & Wales

def _nomis(zip_name, member, level, prefix, source_id, note, skip_geog=()):
    rows = []
    with zipfile.ZipFile(os.path.join(RAW, zip_name)) as z:
        text = z.read(member).decode("utf-8-sig", "replace")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    cats = []
    for i, h in enumerate(header):
        if h.startswith(prefix):
            # ONS ships at least one header with a trailing space
            # ("Religion (detailed): Muslim "), which would silently break any
            # later join on the category string.  Trim, change nothing else.
            cats.append((i, h[len(prefix):].strip()))
    if not cats:
        raise SystemExit("no category columns matched %r in %s" % (prefix, member))
    for r in reader:
        if not r or r[2] in skip_geog:
            continue
        code, name = r[2], r[1]
        for i, cat in cats:
            n = int(r[i])
            if n == 0 and not cat.startswith("Total"):
                continue
            rows.append({
                "geo_id": code, "geo_level": level, "geo_name": name,
                "source_category": cat, "count": n, "basis": BASIS,
                "year": 2021, "source_id": source_id, "note": note,
            })
    return rows


def read_ew():
    rows = []
    for member, level in TS030_LEVELS.items():
        rows += _nomis(
            "census2021-ts030.zip", "census2021-ts030-%s.csv" % member, level,
            "Religion: ", "uk_ew_census_2021",
            "TS030",
            skip_geog=(EW_COMBINED,))
    for member, level in TS031_LEVELS.items():
        rows += _nomis(
            "census2021-ts031.zip", "census2021-ts031-%s.csv" % member, level,
            "Religion (detailed): ", "uk_ew_census_2021",
            "TS031 hierarchical",
            skip_geog=(EW_COMBINED,))
    return rows


# ---------------------------------------------------------------------- Scotland

def read_sc():
    """NRS UV205, Output Area.  Four preamble lines, then a footer of notes."""
    path = os.path.join(RAW, "sc_UV205_religion_OA.csv")
    with open(path, encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()
    # The header row is the first one whose first field is empty and which names
    # "All people" -- the three lines above it are the table title block.
    start = next(i for i, l in enumerate(lines) if l.startswith(',"All people"'))
    reader = csv.reader(io.StringIO("\n".join(lines[start:])))
    header = next(reader)
    cats = list(enumerate(header))[1:]
    rows = []
    for r in reader:
        if not r or not r[0].startswith("S00"):
            continue  # footer notes and the trailing blank line
        code = r[0]
        for i, cat in cats:
            v = r[i].strip()
            n = 0 if v in ("-", "") else int(v)
            if n == 0 and cat != "All people":
                continue
            rows.append({
                "geo_id": code, "geo_level": "output_area", "geo_name": code,
                "source_category": cat, "count": n, "basis": BASIS,
                "year": 2022, "source_id": "uk_sc_census_2022",
                "note": "UV205",
            })
    return rows


# -------------------------------------------------------------- Northern Ireland

def _nisra(filename, level, source_id, note, year=2021):
    path = os.path.join(RAW, filename)
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # columns: <geo> Code, <geo> Label, <var> Code, <var> Label, Count
        for r in reader:
            if len(r) < 5:
                continue
            n = int(r[4])
            if n == 0:
                continue
            rows.append({
                "geo_id": r[0], "geo_level": level, "geo_name": r[1],
                "source_category": r[3], "count": n, "basis": BASIS,
                "year": year, "source_id": source_id, "note": note,
            })
    return rows, header


NI_BULK = "ni_census-2021-main-statistics-phase-1-all-tables.zip"
NOTE_RE = re.compile(r"\[note\s*\d+\]")


def _nisra_bulk(member, sheet, source_id, note):
    """MS-B20 / MS-B24: an Excel sheet, geographies down, categories across.

    The sheet holds two stacked tables -- counts (…a) then percentages (…b).
    Only the counts block is read, identified by the header row that starts
    with "Geography" and is followed by the "Northern Ireland" row.
    """
    import pandas as pd  # only needed for the two Excel tables

    with zipfile.ZipFile(os.path.join(RAW, NI_BULK)) as z:
        book = pd.ExcelFile(io.BytesIO(z.read(member)))
    df = book.parse(sheet, header=None)
    hdr_rows = [i for i in range(len(df)) if str(df.iat[i, 0]).strip() == "Geography"]
    if not hdr_rows:
        raise SystemExit("no Geography header row in %s" % member)
    h = hdr_rows[0]                       # first block = counts
    end = hdr_rows[1] - 2 if len(hdr_rows) > 1 else len(df)
    cats = []
    for j in range(2, df.shape[1]):
        lab = NOTE_RE.sub("", str(df.iat[h, j]))
        lab = " ".join(lab.split())       # headers carry embedded newlines
        if lab and lab != "nan":
            cats.append((j, lab))
    rows = []
    for i in range(h + 1, end):
        name = str(df.iat[i, 0]).strip()
        code = str(df.iat[i, 1]).strip()
        if not code or code == "nan":
            continue
        level = "country" if code == "N92000002" else "lgd"
        for j, cat in cats:
            v = df.iat[i, j]
            if v != v:                    # NaN
                continue
            n = int(v)
            if n == 0 and not cat.startswith("All usual residents"):
                continue
            rows.append({
                "geo_id": code, "geo_level": level, "geo_name": name,
                "source_category": cat, "count": n, "basis": BASIS,
                "year": 2021, "source_id": source_id, "note": note,
            })
    return rows


def read_ni():
    """Same split as England & Wales: the coarse table supplies the fine
    geography, the expanded table supplies the fine categories, and the two
    never share a geo_level so nothing can be double counted by accident."""
    rows = []
    rows += _nisra("ni_MS-B19_religion_DZ21.csv", "data_zone",
                   "uk_ni_census_2021", "MS-B19 belongs-to")[0]
    rows += _nisra("ni_MS-B23_religion_brought_up_in_DZ21.csv", "data_zone",
                   "uk_ni_census_2021_brought_up_in", "MS-B23 brought-up-in")[0]
    rows += _nisra_bulk("census-2021-ms-b20.xlsx", "MS-B20",
                        "uk_ni_census_2021", "MS-B20 hierarchical")
    rows += _nisra_bulk("census-2021-ms-b24.xlsx", "MS-B24",
                        "uk_ni_census_2021_brought_up_in", "MS-B24 hierarchical")
    return rows


# ------------------------------------------------------------------- reconciling

def check(rows):
    def tot(sid, level, cat):
        return sum(r["count"] for r in rows if r["source_id"] == sid
                   and r["geo_level"] == level and r["source_category"] == cat)

    def units(sid, level):
        return len({r["geo_id"] for r in rows
                    if r["source_id"] == sid and r["geo_level"] == level})

    print("=== England & Wales, ONS Census 2021 ===")
    # ONS's own published TS030 "England and Wales" row (29 Nov 2022).  Summing
    # the output-area table does NOT reproduce it: Census 2021 applies cell key
    # perturbation independently at every geography, so aggregates drift.  The
    # check is therefore a tolerance, and the drift itself is the finding.
    pub = {"Total: All usual residents": 59597540, "Christian": 27522672,
           "No religion": 22162062, "Not answered": 3595589,
           "Muslim": 3868133, "Hindu": 1032775, "Sikh": 524140,
           "Jewish": 271327, "Buddhist": 272508, "Other religion": 348334}
    TOL = 0.001  # 0.1%
    ok = True
    for cat, want in pub.items():
        got = tot("uk_ew_census_2021", "output_area", cat)
        d = got - want
        good = abs(d) <= TOL * want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} OA sum {cat:<28} {got:>11,}  "
              f"(ONS {want:>11,}) {d:>+6,}  {d / want * 100:+.4f}%")
    print(f"      output areas {units('uk_ew_census_2021', 'output_area'):,} · "
          f"MSOAs {units('uk_ew_census_2021', 'msoa'):,} · "
          f"LTLAs {units('uk_ew_census_2021', 'ltla')} · "
          f"countries {units('uk_ew_census_2021', 'country')}")

    ts031_tot = tot("uk_ew_census_2021", "country", "Total: All Usual Residents")
    print(f"      TS031 country total {ts031_tot:,} vs TS030 "
          f"{pub['Total: All usual residents']:,} "
          f"-> {ts031_tot - pub['Total: All usual residents']:+,} "
          "(disclosure control, see uk.md)")

    print("=== Scotland, NRS Census 2022 ===")
    sc_tot = tot("uk_sc_census_2022", "output_area", "All people")
    print(f"      OA sum, All people {sc_tot:,}  (NRS published 5,436,600, "
          f"{(sc_tot - 5436600) / 5436600 * 100:+.3f}% — NRS perturbs too)")
    for cat, pct in [("No religion", 51.1), ("Church of Scotland", 20.4),
                     ("Roman Catholic", 13.3), ("Other Christian", 5.1),
                     ("Muslim", 2.2), ("Religion not stated", 6.2)]:
        got = tot("uk_sc_census_2022", "output_area", cat)
        print(f"      {cat:<22} {got:>10,}  {got / sc_tot * 100:5.2f}%  "
              f"(NRS {pct}%)")
    for cat in ("Pagan", "Buddhist", "Hindu", "Jewish", "Sikh", "Other religion"):
        print(f"      {cat:<22} {tot('uk_sc_census_2022', 'output_area', cat):>10,}")
    print(f"      output areas {units('uk_sc_census_2022', 'output_area'):,}")

    print("=== Northern Ireland, NISRA Census 2021 ===")
    print("      NI Census 2021 published population 1,903,175")
    for sid, label in [("uk_ni_census_2021", "MS-B19 religion"),
                       ("uk_ni_census_2021_brought_up_in", "MS-B23 brought up in")]:
        # data_zone tables are flat with no total column, so the whole table is
        # the population; the LGD/country tables carry their own total column.
        s = sum(r["count"] for r in rows
                if r["source_id"] == sid and r["geo_level"] == "data_zone")
        print(f"      {label:<24} data_zone  sum {s:>10,} "
              f"({s - 1903175:+,})  units {units(sid, 'data_zone'):,}")
    for sid, label in [("uk_ni_census_2021", "MS-B20 expanded"),
                       ("uk_ni_census_2021_brought_up_in", "MS-B24 expanded")]:
        for lvl in ("lgd", "country"):
            s = tot(sid, lvl, "All usual residents")
            print(f"      {label:<24} {lvl:<10} total {s:>10,} "
                  f"({s - 1903175:+,})  units {units(sid, lvl):,}")
    for sid, lvl, label in [("uk_ni_census_2021", "data_zone", "MS-B19"),
                            ("uk_ni_census_2021_brought_up_in", "data_zone",
                             "MS-B23"),
                            ("uk_ni_census_2021", "country", "MS-B20"),
                            ("uk_ni_census_2021_brought_up_in", "country",
                             "MS-B24")]:
        cats = sorted({r["source_category"] for r in rows
                       if r["source_id"] == sid and r["geo_level"] == lvl},
                      key=lambda c: -tot(sid, lvl, c))
        print(f"      {label} categories ({lvl}):")
        for c in cats:
            print(f"        {c:<62} {tot(sid, lvl, c):>10,}")

    print()
    print("rows:", len(rows))
    lv = {}
    for r in rows:
        lv.setdefault((r["source_id"], r["geo_level"]), set()).add(r["geo_id"])
    for k in sorted(lv):
        print("  ", k, len(lv[k]), "units")
    if not ok:
        raise SystemExit("England & Wales reconciliation FAILED")


def main():
    rows = read_ew() + read_sc() + read_ni()
    check(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, "%.1f MB" % (os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
