"""Singapore — Department of Statistics, Census of Population 2020, religion by planning area.

Reads (or fetches) data/raw/sg/ and writes data/normalized/sg.csv.

ONE TABLE, NINE CATEGORIES, THIRTY-ONE UNITS. SingStat's TableBuilder table CT/17592,
republished on data.gov.sg as dataset `d_a58564fbed922609a0f79af96069dd9b`, *Resident
Population Aged 15 Years and Over by Planning Area of Residence and Religion (Census of
Population 2020)*. Open Data Licence, no key, no login. The categories are:

    No Religion, Buddhism, Taoism, Islam, Hinduism, Sikhism,
    (Christianity) Catholic, (Christianity) Other Christians, Other Religions

and they partition the total exactly. Nine is a middling list for this map (the Philippines
names 129 bodies and Vietnam 28) and the shape is the interesting part: Singapore names
Sikhism separately at 12,051 people, 0.35% of the country, while leaving every non-Catholic
Christian in one undivided cell of 411,674. See `taxonomy/sg2020.py`.

**THE UNIVERSE IS RESIDENTS AGED 15 AND OVER, WHICH IS 60.8% OF THE PEOPLE IN SINGAPORE, AND
BOTH HALVES OF THAT GAP ARE LARGE ENOUGH TO SAY OUT LOUD.** Against the census's own
population table (Table 1.1 of Statistical Release 1):

    total population, June 2020                 5,685,800
      residents (citizens + permanent residents) 4,044,210
        aged 15 and over  <- the religion universe 3,459,093
        under 15                                   585,117
      non-residents                              1,641,590

Neither exclusion is scaled up, which is Chile's rule (sources/cl.md 3) applied twice. The
under-15s are excluded because Singapore's religiosity varies sharply with age -- 24.2% of
15-24s report no religion against 15.2% of the over-55s (Chapter 5, Chart 5.5) -- so a flat
scale-up would be an assertion about children the census declined to make. The non-residents
are excluded because the census does not ask them: work permit and S Pass holders, employment
pass holders, dependants and foreign students are outside the religion question entirely, and
no SingStat table anywhere gives their religion. See `sources/sg.md` 4.

THE `Others` ROW IS A REAL UNIT AND IS DRAWN, NOT DROPPED. The table names 30 planning areas
and lumps the rest into `Others`, 25,756 people. Those are the 25 URA planning areas with
almost nobody living in them -- Rochor, Newton, Singapore River, Southern Islands, Changi and
twenty more, several of which (Marina South, Straits View, Simpang, Changi Bay) have a
resident population of zero. `sources/sg_geo.py` gives that row the union of exactly those 25
polygons as its placement geometry, so its dots land in the right quarter of the island and
nobody is lost. The identification is checked rather than assumed: the census's own
population-by-planning-area table puts 34,050 residents of all ages in those same 25 areas,
against 25,756 aged 15+, and 25,756 / 34,050 = 0.76 against a national 15+ share of 0.86,
which is the right size for a set of areas skewed old (see `sg_geo.py`).

THE FILE IS RANDOMLY ROUNDED AND THE COLUMNS DO NOT QUITE ADD UP. SingStat perturbs small
cells for confidentiality, so summing the 31 unit rows gives 3,459,094 against a published
3,459,093, and the worst single column is off by 4 people in 539,251. The report says so
("Figures may not add up to the totals due to rounding"). The reconciliation below therefore
asserts a TOLERANCE OF 10 PER COLUMN rather than equality -- but it also asserts that every
column matches the printed report EXACTLY, which is a different and much stronger check,
because the printed figures came off a different production run than the TableBuilder export.

`-` IS AN IN-BAND SENTINEL MEANING NIL OR NEGLIGIBLE, not missing. It appears once, in
Downtown Core's Sikhism cell, and the metadata types that whole column as Text because of it.
Anything else non-numeric raises rather than being coerced (Sri Lanka's rule, sources/lk.md).

Usage:
    python sources/sg.py --fetch    one signed CSV from data.gov.sg, ~2 KB, seconds
    python sources/sg.py            normalise from data/raw/sg/
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sg")
OUT = os.path.join(ROOT, "data", "normalized", "sg.csv")

DATASET = "d_a58564fbed922609a0f79af96069dd9b"
RAW_NAME = "sg_religion_by_planning_area_2020.csv"

SOURCE_ID = "sg_cop_2020"
YEAR = 2020
BASIS = "self_id"          # glossary: "It is as declared by the person."
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# The nine categories, in the file's own column order. `Total` is the universe row and is
# emitted so the reconciliation can be re-run downstream, but resolves to nothing.
CATEGORIES = ["NoReligion", "Buddhism", "Taoism1", "Islam", "Hinduism", "Sikhism",
              "Christianity_Catholic", "Christianity_OtherChristians", "OtherReligions"]

# Census of Population 2020, Statistical Release 1, Table 51, `Total` row (report page 198;
# the same nine figures also appear in UNSD Demographic Yearbook table 28 for Singapore
# 2020). PRINTED, from a different production run than the TableBuilder export this script
# parses, so agreement here is two independent renderings of the same census and not a
# tautology. Exact equality is asserted.
PUBLISHED_NATIONAL = {
    "Total": 3_459_093,
    "NoReligion": 692_528,
    "Buddhism": 1_074_159,
    "Taoism1": 303_960,
    "Islam": 539_251,
    "Hinduism": 172_963,
    "Sikhism": 12_051,
    "Christianity_Catholic": 242_681,
    "Christianity_OtherChristians": 411_674,
    "OtherReligions": 9_827,
}

# Table 1.1, Statistical Release 1 (report page 4). Used only to state the gap.
TOTAL_POPULATION = 5_685_800
RESIDENT_POPULATION = 4_044_210

# Random rounding for confidentiality. The 31 unit rows may miss the published national
# figure by this much per column and no more; observed worst is 4.
ROUNDING_TOLERANCE = 10

NIL = "-"          # "nil or negligible", the report's own notation


def _num(v, where):
    v = (v or "").strip()
    if v == NIL:
        return 0
    try:
        return int(v.replace(",", ""))
    except ValueError:
        raise SystemExit(f"!! {where}: {v!r} is neither a number nor the nil sentinel "
                         f"{NIL!r}. Do NOT coerce it; find out what it means first.")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, RAW_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000:
        print("already have", dest)
        return
    # data.gov.sg hands out a SIGNED, EXPIRING S3 URL rather than a stable download path, so
    # the poll has to be re-run every time; there is no permanent URL to hard-code.
    poll = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET}/poll-download"
    print("GET", poll)
    j = requests.get(poll, timeout=120).json()
    if j.get("code") != 0 or not j.get("data", {}).get("url"):
        raise SystemExit(f"!! poll-download did not return a URL: {j}")
    r = requests.get(j["data"]["url"], timeout=300)
    r.raise_for_status()
    if b"," not in r.content[:200]:
        raise SystemExit("!! the download is not a CSV")
    with open(dest + ".part", "wb") as fh:
        fh.write(r.content)
    os.replace(dest + ".part", dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def main():
    if "--fetch" in sys.argv:
        fetch()

    src = os.path.join(RAW, RAW_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} — run with --fetch first")

    with open(src, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("!! the CSV parsed to zero rows")

    have = [c for c in rows[0] if c != "Number"]
    if have != CATEGORIES[:0] + ["Total"] + CATEGORIES:
        raise SystemExit(f"!! the column set changed: {have}\n"
                         f"   expected {['Total'] + CATEGORIES}")

    national, units = None, []
    for r in rows:
        label = (r["Number"] or "").strip()
        vals = {c: _num(r[c], f"{label}/{c}") for c in ["Total"] + CATEGORIES}
        if label == "Total":
            national = vals
        else:
            units.append((label, vals))
    if national is None:
        raise SystemExit("!! no `Total` row in the file")
    print(f"planning-area rows: {len(units)}  "
          f"(30 named + `Others`)" if len(units) == 31 else
          f"planning-area rows: {len(units)}")
    if len(units) != 31:
        raise SystemExit(f"!! expected 31 unit rows (30 named + Others), got {len(units)}")
    if not any(u == "Others" for u, _ in units):
        raise SystemExit("!! no `Others` row — the table's shape has changed")

    # ---- check 1: the printed report, exactly. Two independent renderings of one census.
    print("\nagainst Statistical Release 1 Table 51 (printed) and UNSD table 28:")
    for c in ["Total"] + CATEGORIES:
        want = PUBLISHED_NATIONAL[c]
        got = national[c]
        print(f"  {c:<30}{got:>10,}{want:>12,}   {'OK' if got == want else 'MISMATCH'}")
        if got != want:
            raise SystemExit(
                f"!! {c}: the download says {got:,}, the printed report says {want:,}. "
                "One of them is not the 2020 census; do NOT proceed on the assumption "
                "that the newer file wins.")

    # ---- check 2: the nine categories partition the total, per unit and nationally.
    worst_part = 0
    for label, v in units + [("Total", national)]:
        d = sum(v[c] for c in CATEGORIES) - v["Total"]
        if abs(d) > abs(worst_part):
            worst_part, worst_where = d, label
    print(f"\nnine categories vs each row's own Total: worst residual "
          f"{worst_part:+,} ({worst_where})")
    if abs(worst_part) > ROUNDING_TOLERANCE:
        raise SystemExit(f"!! {worst_where} is off by {worst_part}, beyond random rounding. "
                         "A category is missing or double counted.")

    # ---- check 3: the 31 units sum to the national row, within random rounding.
    print("\n31 units summed vs the national row (random rounding, tolerance "
          f"{ROUNDING_TOLERANCE}):")
    worst = 0
    for c in ["Total"] + CATEGORIES:
        s = sum(v[c] for _, v in units)
        d = s - national[c]
        worst = max(worst, abs(d))
        print(f"  {c:<30}{s:>10,}{national[c]:>12,}{d:>+6}")
        if abs(d) > ROUNDING_TOLERANCE:
            raise SystemExit(f"!! {c} is off by {d}. That is not rounding — a unit row is "
                             "missing, duplicated or attributed to the wrong area.")
    print(f"  worst column {worst:+}, which is SingStat's random rounding for confidentiality")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        note = ("residents aged 15+; Taoism includes Chinese Traditional Beliefs "
                "(report footnote 3/)")
        for c in ["Total"] + CATEGORIES:
            w.writerow(["SG", "country", "Singapore", c, national[c],
                        BASIS, YEAR, SOURCE_ID, note])
        # ALL 31 rows carry geo_level `planning_area`, INCLUDING `Others`, and the residual
        # is distinguished by its geo_id and its note instead. A separate level for it was
        # tried and reverted: tools/check_mapping.py picks one level and reports on that, so
        # a level of its own made 25,753 people vanish from the check without a word.
        for label, v in units:
            n = ("a single row covering the 25 planning areas the table does not name "
                 "separately; " + note) if label == "Others" else note
            for c in ["Total"] + CATEGORIES:
                w.writerow([label, "planning_area", label, c, v[c],
                            BASIS, YEAR, SOURCE_ID, n])
    os.replace(tmp, OUT)

    drawn = sum(v["Total"] for _, v in units)
    print(f"\nwrote {OUT}")
    print(f"  {len(units)} units, {len(CATEGORIES)} categories, {drawn:,} people")
    print(f"  universe is residents aged 15+: {drawn:,} of a resident population "
          f"{RESIDENT_POPULATION:,} and a total population {TOTAL_POPULATION:,}")
    print(f"  = {100.0 * drawn / TOTAL_POPULATION:.1f}% of everybody in Singapore; the rest "
          f"is {RESIDENT_POPULATION - national['Total']:,} residents under 15 and "
          f"{TOTAL_POPULATION - RESIDENT_POPULATION:,} non-residents,")
    print("  neither of whom the census asks (sources/sg.md 4). Not scaled up.")


if __name__ == "__main__":
    main()
