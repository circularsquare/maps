"""
New Zealand — Stats NZ census religious affiliation.  Two tables, because the depth and the
geography trade off against each other and neither table has both:

    sa2       2023 Census, 2,395 SA2s, **level 1** of the classification (11 categories)
    national  2018 Census, one row per area, **level 3** (165 categories)

    python sources/nz.py --fetch     # download data/raw/nz/  (~7MB, no key needed)
    python sources/nz.py             # data/raw/nz/ -> data/normalized/nz.csv

The SA2 table is the "2023 Census totals by topic for individuals by SA2" feature service
published by Stats NZ Geospatial (CC BY 4.0).  Religious affiliation is carried on *part 1*
of that service as level 1 of the classification: 9 religion groups plus "Object to
answering" plus "Residual Categories", for the 2013, 2018 and 2023 Censuses, all on 2023 SA2
boundaries.

Level 3 — Anglican, Roman Catholic, Rātana, Ringatū, Sikhism, Tenrikyo, Zoroastrian, and 158
more — exists for 2023 only inside Aotearoa Data Explorer, whose API needs a subscription
key, and only down to territorial authority.  The 2018 national highlights CSV is the same
classification at full depth, free and unauthenticated, so it is carried here as the
structure source in the sense of spec.md §3.4: **2018 shares, 2023 totals**, never mixed
into one figure by this script.  See sources/nz.md.

Two things about this table that the normaliser has to preserve rather than tidy away:

1.  It is a **total responses** variable.  A person could give up to four religions, so the
    category counts sum past the number of people (spec.md §3.1, §3.3).  Both aggregate rows
    the source publishes — "Total" (people) and "Total stated" (people who named at least one
    religion) — are carried through with a note so the inflation stays measurable, and they
    must never be added to the category rows.

2.  Counts are randomly rounded to base 3 by Stats NZ, independently per cell, so a row of
    categories does not add exactly to its own total.  See sources/nz.md.

3.  Suppressed cells carry **negative sentinels**, -999 "Confidential" and -997 "Not
    available", in the same integer column as the counts.  Summing the column as it stands
    turns Islam nationally into -36,753 and hides a 4% shortfall in every other category, so
    they are dropped here and counted out loud rather than being allowed through as numbers.

Field names in the service are opaque (VAR_1_403 …); the mapping to category names comes out
of the layer's own field aliases, read from the saved metadata rather than hardcoded, so a
re-release that renumbers the columns fails loudly instead of silently mislabelling them.
"""

import argparse
import csv
import io
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "nz"
OUT = ROOT / "data" / "normalized" / "nz.csv"

SERVICE = (
    "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/"
    "2023_Census_totals_by_topic_for_individuals_by_SA2/FeatureServer"
)
# Layer 1 is "part 1", unclipped (layer 0 is the same table clipped to the coastline; the
# attributes are identical and we take geometry from the boundary files, not from here).
LAYER = 1
ITEM = "https://www.arcgis.com/sharing/rest/content/items/29a82d5a0ea24a3880219bcb3df126dc"
PAGE = 2000  # the service's own maxRecordCount

SOURCE_ID = "nz_census_2023"
YEAR = 2023

# 2018 Census totals by topic, national highlights.  52 CSVs in one zip; we want one of them.
NAT2018_URL = (
    "https://www.stats.govt.nz/assets/Uploads/2018-Census-totals-by-topic/Download-data/"
    "2018-census-totals-by-topic-national-highlights-csv.zip"
)
NAT2018_MEMBER = "religious-affiliation-total-responses-2018-census-csv.csv"
NAT2018_SOURCE_ID = "nz_census_2018"

META = RAW / "sa2_part1_layer.json"
ITEM_META = RAW / "arcgis_item.json"
NAT2018 = RAW / "2018-census-totals-by-topic-national-highlights-csv.zip"

# alias shape: "Subject pop: ..., Year: 2023, Measure: Count, Var1: Religious affiliation (X)"
RELIGION_RE = re.compile(
    r"Year:\s*(?P<year>\d{4}),\s*Measure:\s*Count,\s*"
    r"Var1:\s*Religious affiliation \((?P<cat>.+)\)\s*$"
)
POP_RE = re.compile(
    r"Year:\s*(?P<year>\d{4}),\s*Measure:\s*Count,\s*"
    r"Var1:\s*Census usually resident population count \(Total\)\s*$"
)

ID_FIELD = "SA22023_V1_00"
NAME_FIELD = "SA22023_V1_00_NAME"

# The two rows the source publishes that are aggregates over the others.  Kept, because the
# gap between them and the sum of the categories IS the multiple-response inflation, which is
# the thing spec.md §3.3 wants visible; flagged, because summing them with the categories
# would double count every person in the country.
AGGREGATES = {"Total", "Total stated"}

# Stats NZ's in-band missing-value codes, documented in the layer's own metadata.
SENTINELS = {-999: "Confidential", -997: "Not available"}


def get(url: str, params: dict) -> bytes:
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(url + "?" + q, headers={"User-Agent": "religiondots/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    print("layer metadata ->", META)
    META.write_bytes(get(f"{SERVICE}/{LAYER}", {"f": "json"}))
    ITEM_META.write_bytes(get(ITEM, {"f": "json"}))

    fields = dict(religion_fields())
    fields.update({f: (y, "population") for f, y in population_fields().items()})
    wanted = [ID_FIELD, NAME_FIELD] + sorted(fields, key=lambda f: int(f.split("_")[-1]))
    print(f"{len(fields)} count fields")

    offset = 0
    page = 0
    while True:
        body = get(
            f"{SERVICE}/{LAYER}/query",
            {
                "where": "1=1",
                "outFields": ",".join(wanted),
                "returnGeometry": "false",
                "orderByFields": ID_FIELD,
                "resultOffset": offset,
                "resultRecordCount": PAGE,
                "f": "json",
            },
        )
        path = RAW / f"sa2_part1_religion_page{page}.json"
        path.write_bytes(body)
        n = len(json.loads(body)["features"])
        print(f"  {path.name}: {n} rows")
        if n < PAGE:
            break
        offset += n
        page += 1

    for stale in sorted(RAW.glob("sa2_part1_religion_page*.json")):
        if int(stale.stem.rsplit("page", 1)[1]) > page:
            print("  removing stale", stale.name)
            stale.unlink()

    print("2018 national level-3 table ->", NAT2018.name)
    req = urllib.request.Request(NAT2018_URL, headers={"User-Agent": "religiondots/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        NAT2018.write_bytes(r.read())


def religion_fields() -> dict:
    """{field name -> (year, category)} for every religion count column, from the aliases."""
    meta = json.loads(META.read_text(encoding="utf-8"))
    out = {}
    for f in meta["fields"]:
        m = RELIGION_RE.search(f.get("alias") or "")
        if m:
            out[f["name"]] = (int(m.group("year")), m.group("cat").strip())
    if not out:
        raise SystemExit("no religion fields found in the layer aliases — has the service changed?")
    return out


def population_fields() -> dict:
    meta = json.loads(META.read_text(encoding="utf-8"))
    return {
        f["name"]: int(m.group("year"))
        for f in meta["fields"]
        if (m := POP_RE.search(f.get("alias") or ""))
    }


def national_2018_rows() -> list:
    """The 2018 national table at level 3 of the classification — the depth SA2 does not have.

    Columns: Code, Religious_affiliation, Census_usually_resident_population_count.  Codes are
    5 characters and must stay strings: '00000' is No Religion.  The three trailing rows
    (TotalStated / TotalResponse / Total) are aggregates and carry the same warning as the SA2
    ones — TotalResponse minus Total *is* the multiple-response inflation, measured.
    """
    if not NAT2018.exists():
        print(f"  {NAT2018.name} missing — skipping the 2018 national level-3 table")
        return []

    with zipfile.ZipFile(NAT2018) as z:
        raw = z.read(NAT2018_MEMBER).decode("utf-8-sig")

    rows = []
    for rec in csv.DictReader(io.StringIO(raw)):
        code = rec["Code"].strip()
        cat = rec["Religious_affiliation"].strip()
        n = int(rec["Census_usually_resident_population_count"])
        aggregate = not code.isdigit()
        rows.append(
            {
                "geo_id": "NZ",
                "geo_level": "national",
                "geo_name": "New Zealand",
                "source_category": cat,
                "count": n,
                "basis": "self_id",
                "year": 2018,
                "source_id": NAT2018_SOURCE_ID,
                "note": (
                    "aggregate over the categories, do not sum"
                    if aggregate
                    else f"level 3 code {code}; total responses; up to 4 per person; "
                    "structure source for the 2023 SA2 level-1 totals (spec.md §3.4)"
                ),
            }
        )
    return rows


def normalize() -> None:
    if not META.exists():
        raise SystemExit(f"{META} missing — run: python {Path(__file__).name} --fetch")

    fields = religion_fields()
    this_year = {f: cat for f, (y, cat) in fields.items() if y == YEAR}
    print(f"{len(this_year)} categories for {YEAR}")

    rows = []
    seen = set()
    dropped = {}          # sentinel -> count of cells
    dropped_units = set()  # SA2s with at least one suppressed cell
    for path in sorted(RAW.glob("sa2_part1_religion_page*.json"),
                       key=lambda p: int(p.stem.rsplit("page", 1)[1])):
        for feat in json.loads(path.read_text(encoding="utf-8"))["features"]:
            a = feat["attributes"]
            geo_id = str(a[ID_FIELD])
            if geo_id in seen:
                raise SystemExit(f"duplicate SA2 {geo_id} — check the paging")
            seen.add(geo_id)
            name = a[NAME_FIELD]
            for field, cat in sorted(this_year.items(), key=lambda kv: int(kv[0].split("_")[-1])):
                v = a.get(field)
                if v is None:
                    continue
                if v in SENTINELS:
                    dropped[v] = dropped.get(v, 0) + 1
                    dropped_units.add(geo_id)
                    continue
                if v < 0:
                    raise SystemExit(f"unknown negative code {v} in {field} at SA2 {geo_id}")
                rows.append(
                    {
                        "geo_id": geo_id,
                        "geo_level": "sa2",
                        "geo_name": name,
                        "source_category": cat,
                        "count": int(v),
                        # A census question about the respondent's own religion: self_id for
                        # every row, including the two aggregates and "Object to answering",
                        # which is a stated refusal rather than a different kind of measure.
                        "basis": "self_id",
                        "year": YEAR,
                        "source_id": SOURCE_ID,
                        "note": (
                            "aggregate over the categories, do not sum"
                            if cat in AGGREGATES
                            else "total responses; up to 4 per person"
                        ),
                    }
                )

    national = national_2018_rows()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["geo_id", "geo_level", "geo_name", "source_category",
            "count", "basis", "year", "source_id", "note"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows + national)

    report(rows, len(seen), dropped, dropped_units)
    report_national(national)


def report(rows: list, units: int, dropped: dict, dropped_units: set) -> None:
    """National totals per category, and the multiple-response inflation."""
    tot = {}
    for r in rows:
        tot[r["source_category"]] = tot.get(r["source_category"], 0) + r["count"]

    print(f"\n{OUT.relative_to(ROOT)}: {len(rows):,} rows, {units:,} SA2s\n")
    width = max(len(c) for c in tot)
    for cat, n in sorted(tot.items(), key=lambda kv: -kv[1]):
        flag = "   (aggregate)" if cat in AGGREGATES else ""
        print(f"  {cat:<{width}}  {n:>11,}{flag}")

    cats = sum(n for c, n in tot.items() if c not in AGGREGATES)
    stated = tot.get("Total stated")
    people = tot.get("Total")
    print(f"\n  sum of the {len(tot) - len(AGGREGATES)} categories  {cats:>11,}")
    if stated:
        print(f"  responses per person stated       {cats / stated:>11.4f}")
    if people:
        print(f"  responses as a share of people    {cats / people:>11.4f}")

    pops = population_fields()
    field = next((f for f, y in pops.items() if y == YEAR), None)
    if field:
        pop = 0
        for path in sorted(RAW.glob("sa2_part1_religion_page*.json")):
            for feat in json.loads(path.read_text(encoding="utf-8"))["features"]:
                v = feat["attributes"].get(field)
                if v is not None and v not in SENTINELS:
                    pop += int(v)
        if pop:
            print(f"  census usually resident population {pop:>10,}")

    if dropped:
        n = sum(dropped.values())
        print()
        for code, k in sorted(dropped.items()):
            print(f"  dropped {k:,} cells coded {code} ({SENTINELS[code]})")
        print(f"  across {len(dropped_units)} of {units} SA2s; each is a count under 6,")
        print(f"  so the national sums are low by at most {n * 5:,} people")


def report_national(rows: list) -> None:
    """The 2018 level-3 table: how deep it goes, and the inflation it makes visible."""
    if not rows:
        return
    by_cat = {r["source_category"]: r["count"] for r in rows}
    detail = [r for r in rows if "aggregate" not in r["note"]]
    print(f"\n2018 national, level 3: {len(detail)} categories\n")
    for r in sorted(detail, key=lambda r: -r["count"])[:12]:
        print(f"  {r['source_category']:<42}  {r['count']:>9,}")
    print(f"  … {max(0, len(detail) - 12)} more")

    people = by_cat.get("Total")
    responses = by_cat.get("Total responses")
    if people and responses:
        print(f"\n  people                     {people:>11,}")
        print(f"  responses                  {responses:>11,}")
        print(f"  inflation from multiple responses  {responses - people:>7,} "
              f"= {100 * (responses / people - 1):.2f}%")


def main() -> None:
    # Category names carry macrons (Māori Religions…); a Windows console is cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="download data/raw/nz/ first")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    normalize()


if __name__ == "__main__":
    sys.exit(main())
