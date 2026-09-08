"""The English Church Census 2005 -> county x settlement denominational structure.

`sources/uk_ecc.py` -> `data/normalized/uk_ecc.csv`

WHY THIS EXISTS. **England and Wales publish no Christian denomination, at any geography,
for 27.5 million people** (sources/uk.md §2). Every neighbour on the map splits its
Christians at least two ways -- Scotland three, Northern Ireland twenty-six, Ireland
fourteen, France, Germany and the Netherlands all at least two -- and England is the flat
patch. This file is the structure half of the fix: spec.md §3.5a's mechanism, where a
survey supplies the totals and an institutional count supplies the shape inside them.
`sources/uk_bes.py` is the survey half and `uk_split.py` does the arithmetic.

WHAT IT IS. The fourth English Church Census, 8 May 2005, organised by Peter Brierley for
Christian Research: a postal census of every Trinitarian church in England, 37,051
approached and 18,633 returning usable data. Deposited by David Voas at the UK Data
Archive as SN 6409 and redistributed free by the ARDA, which is the copy read here.
`sources/uk_ecc.md` has the re-fetch and the licence.

THE GEOGRAPHY IS COUNTIES, AND NOT TODAY'S. The released file carries `cntycde` and
nothing finer -- no postcode, no local authority, despite the UKDA metadata claiming
Local Authority Districts as the spatial unit. Its 47 English counties are the pre-1996
set: **Avon, Cleveland and `Hereford UA and Worcester` are all present**, and they were
abolished before this census was taken. `uk_split.py` owns the lookup back to 2021
districts; this file emits the county exactly as the source names it.

THE SECOND AXIS IS WHY THIS IS WORTH DOING AT ALL. 47 counties alone would draw 47 flat
patches. `envnmcde` classifies each church's address into eight settlement types, and the
denominational difference across them is nearly as large as the difference across
counties: Anglicans are 57% of rural attendance and 24% on council estates, Catholics run
the opposite way, 14% to 54%. Collapsed to the three classes ONS's own Rural Urban
Classification can reproduce for a 2021 Output Area, the within-county spread of the
Catholic share still has a median of 17 points and reaches 54 in Staffordshire.

**The collapse is to three and not eight because eight is not reproducible.** Brierley's
coders assigned `inner city`, `council estate`, `suburb-suburban fringe` and `city centre`
by eye from the address; ONS's RUC21 separates conurbation from town from countryside and
nothing inside a conurbation. Mapping eight onto three loses the socio-economic axis and
keeps the settlement one, and it is the only part that can be joined to a census geography
without inventing a classifier. It also makes the cells sturdier: 128 of 144 cells hold at
least 20 churches and 98.4% of the weight, against 174 of 365 at the finer grain.

RESPONSE WAS 50% AND IT WAS NOT EVEN ACROSS DENOMINATIONS, so raw attendance is not a
denominational share. The user guide publishes the rate per denomination and this file
divides by it. The spread is the whole reason it matters:

    Baptist 67 · smaller denominations 57 · Anglican 55 · Catholic 54 · Independents 50
    New Churches 49 · United Reformed 44 · Methodist 37 · Pentecostal 30 · **Orthodox 7**

Uncorrected, Methodism is understated by a third and Pentecostalism by half. The
correction is national -- it assumes response did not vary by county within a
denomination, which is untestable here and is the largest assumption in this file.

**IT DOES NOT REPRODUCE THE PUBLISHED TOTAL AND MUST NOT BE MADE TO.** The user guide is
explicit: "all publications based on the census contain estimates of the total churchgoing
population and cannot be derived directly from this dataset, which includes only half of
the churches in the country... A sophisticated set of assumptions and constraints was used
to produce the published totals." Response-correcting gives 3.94M against Brierley's
published 3.17M. That gap is not a bug to close -- his grossing-up used per-county and
per-size constraints this file does not have. **Nothing downstream uses the magnitude.**
`uk_split.py` reads shares within a cell and throws the level away.

WHAT IT CANNOT DO, AND THE ONE LEG THAT HAS TO GO ELSEWHERE. The census is twenty years
old and its blind spot is exactly the denominations that arrived afterwards. **Orthodoxy
has no usable cell anywhere in England**: 49 churches, a 7% response rate because 8 May
2005 fell the Sunday after Orthodox Easter and many were shut, and zero of the 144 cells
reach three churches. The BES anchor puts Orthodoxy at 2.2% of England's Christians --
larger than the Baptists -- because of Romanian, Bulgarian, Ukrainian and Greek migration
that postdates the census entirely. `orthodox` is emitted here for completeness and
`uk_split.py` **ignores it and places Orthodoxy by census country of birth instead**
(Anita, 2026-09-07). Pentecostalism has a milder version of the same problem: 25 usable
cells holding 64% of its weight, and 2005 predates much of the African church growth in
London and Birmingham. It is kept, with the coverage stated.

EXCLUDED, DELIBERATELY:

  * **The Channel Islands and the Isle of Man**, which the census counts (`cntycde` =
    `Other`, 71 churches) because their churches sit in English dioceses. They are not in
    England, not in the 2021 census, and have no Output Area to receive a dot.
  * **Non-Trinitarian bodies**, which the census never approached: Jehovah's Witnesses,
    Latter-day Saints, Christian Scientists, Christadelphians and Unitarians. They are
    inside the census's `Christian` tick box and outside this structure, so they fall into
    `uk_split.py`'s residual rather than being placed.

Run: python sources/uk_ecc.py            # -> data/normalized/uk_ecc.csv, with checks
     python sources/uk_ecc.py --report   # print the structure table, write nothing
"""
import argparse
import csv
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "uk")
SAV = os.path.join(RAW, "ecc05.dta")
OUT = os.path.join(ROOT, "data", "normalized", "uk_ecc.csv")

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

SOURCE_ID = "uk_en_ecc_2005"
BASIS = "attendance"          # people present at services, spec §3.1
YEAR = 2005

# Denominational response rates, English Church Census 2005 user guide p.4. The key is the
# ECC's own `denom` label; anything not named here took the "smaller denominations" rate.
RESPONSE = {
    "Baptists": 67,
    "Anglicans": 55,
    "Roman Catholic": 54,
    "Independents": 50,
    "New churches": 49,
    "United Reformed": 44,
    "Methodist": 37,
    "Pentecostal": 30,
    "Orthodox": 7,
}
RESPONSE_DEFAULT = 57         # "Smaller denominations", same source

# ECC denomination -> the leg drawn on the map. The legs are the intersection of what the
# ECC names and what the British Election Study names, because a leg needs both a shape
# and a total; see sources/uk_bes.py for the other side.
#
# TWO JOINS THAT ARE JUDGEMENT AND NOT LOOKUP:
#   `West Indian` -> pentecostal.  The ECC's category is a community, not a polity, and its
#       churches are overwhelmingly the bodies BES's Pentecostal option names by name (New
#       Testament Church of God, Church of God in Christ). Filing them as "other" would put
#       156 churches and 22,819 attenders on a residual node.
#   `New churches`/`Independents`/`Emerging churches` -> newchurch. BES's option reads
#       "Evangelical - independent/non-denominational (e.g. FIEC, Pioneer, Vineyard,
#       Newfrontiers)" and those are the same movement the ECC calls New churches. This is
#       the cleanest join in the table and the only one where both sources name examples.
LEG = {
    "Anglicans": "anglican",
    "Roman Catholic": "catholic",
    "Methodist": "methodist",
    "Baptists": "baptist",
    "Pentecostal": "pentecostal",
    "West Indian": "pentecostal",
    "Orthodox": "orthodox",
    "New churches": "newchurch",
    "Independents": "newchurch",
    "Emerging churches": "newchurch",
    "United Reformed": "reformed",
    "Congregational": "reformed",
    "Presbyterian": "reformed",
}
# Everything else -> `other`: Salvation Army, Brethren, Quaker, Seventh-Day Adventist,
# Lutheran, Immigrant churches, Other Protestants, Others. This is the weakest leg in the
# join and uk_split.py says so in the panel: the ECC gives it 7.5% of attendance and BES
# has no tick box for any of these bodies, so it cannot be anchored on its own line.
LEG_DEFAULT = "other"

# Brierley's eight settlement types -> the three ONS's RUC21 can reproduce for a 2021
# Output Area. See the module docstring for why this is a collapse and not a mapping.
SETTLEMENT = {
    "City centre": "conurbation",
    "Inner city": "conurbation",
    "Council estate": "conurbation",
    "Suburb-suburban fringe": "conurbation",
    "Separate town": "town",
    "Other built up area": "town",
    "Commuter rural": "rural",
    "Remote rural": "rural",
}

# Not England. See the docstring.
NOT_ENGLAND = {"Other"}


def _read():
    """The ECC as a frame with `denom`, `cntycde` and `envnmcde` decoded to their labels.

    pandas cannot `read_stata(convert_categoricals=True)` on this file: `cmscomp` carries
    two identical value labels ("Anglo-Catholic/Broad/Radical" appears twice) and pandas
    raises rather than accepting a non-unique categorical. Only three columns need
    decoding, so the label sets are applied by hand.
    """
    if not os.path.exists(SAV):
        sys.exit(f"missing {SAV}\n  see sources/uk_ecc.md for the download")
    reader = pd.io.stata.StataReader(SAV)
    df = reader.read(convert_categoricals=False)
    labels = reader.value_labels()
    by_var = dict(zip(reader._varlist, reader._lbllist))
    for col in ("denom", "cntycde", "envnmcde"):
        table = labels.get(by_var.get(col), {})
        df[col] = df[col].map(lambda v: table.get(v, v))
    return df


def structure():
    """(county rows, settlement rows, diagnostics) -- two tables, and they are not the same.

    **`envnmcde` IS MISSING FOR 37% OF CHURCHES AND THE MISSINGNESS IS DENOMINATIONAL.**
    This is the trap in this file and it is invisible until measured. Churches carrying a
    settlement code are 67% of Anglicans, 74% of Methodists and 70% of Baptists, but only
    53% of Catholics, 38% of New churches, 36% of Pentecostals and **8% of Orthodox**. So
    the coded subset is not the census in miniature: Anglicans are 37.7% of its attendance
    against 25.5% of the uncoded, and Catholics 31.5% against 41.3%. Inner London is the
    worst-coded county in England at 47%.

    Computing a cell's denominational mix from coded churches alone would therefore make
    every conurbation in England look substantially more Anglican and less Catholic than
    the census found it. That is a bias of over ten points on the largest leg, introduced
    by a missing-data pattern rather than by anything about English religion.

    So this function emits the two things that were actually measured and refuses to
    combine them:

      `county`             leg totals over ALL churches in the county. Unbiased by the
                           coding gap, because it never looks at `envnmcde`. This is the
                           load-bearing table.
      `county_settlement`  leg totals over CODED churches only. Carries the settlement
                           signal and the coding bias together, and is only safe read
                           WITHIN a leg -- as "where are this denomination's churches",
                           never as "what is the mix in this cell".

    `uk_split.py` combines them the way `br_rescale.py` combines its two censuses: the
    county table supplies each leg's magnitude, the settlement table supplies each leg's
    own distribution across settlement classes, and the coding rate cancels because it is
    applied within a leg rather than across legs. Its fallback ladder is county, then
    national, for the same reason Brazil's is município, state, nation.
    """
    df = _read()
    n_all = len(df)

    df = df[~df["cntycde"].isin(NOT_ENGLAND)].copy()
    n_offshore = n_all - len(df)

    df["attendance"] = df["totagecl"].fillna(0.0)
    df["leg"] = df["denom"].map(lambda d: LEG.get(d, LEG_DEFAULT))
    df["settlement"] = df["envnmcde"].map(SETTLEMENT)
    df["weight"] = df["attendance"] / df["denom"].map(
        lambda d: RESPONSE.get(d, RESPONSE_DEFAULT)) * 100.0

    def agg(frame, keys):
        g = frame.groupby(keys, as_index=False).agg(
            churches=("churchno", "count"),
            attendance=("attendance", "sum"),
            weight=("weight", "sum"))
        return g[g["weight"] > 0]

    county = agg(df, ["cntycde", "leg"])
    coded = df[df["settlement"].notna()]
    settlement = agg(coded, ["cntycde", "settlement", "leg"])

    diag = {
        "churches_read": n_all,
        "churches_offshore_dropped": n_offshore,
        "churches_used": len(df),
        "churches_with_settlement": len(coded),
        "settlement_coded_pct": 100.0 * len(coded) / len(df),
        "counties": df["cntycde"].nunique(),
        "settlement_cells": settlement.groupby(["cntycde", "settlement"]).ngroups,
        "attendance_raw": float(df["attendance"].sum()),
        "attendance_corrected": float(df["weight"].sum()),
    }
    return county, settlement, diag


def _rows(county, settlement):
    for r in county.itertuples(index=False):
        yield {
            "geo_id": r.cntycde,
            "geo_level": "county",
            "geo_name": r.cntycde,
            "source_category": r.leg,
            "count": round(r.weight, 1),
            "basis": BASIS,
            "year": YEAR,
            "source_id": SOURCE_ID,
            "note": f"{r.churches} churches; {r.attendance:.0f} attenders as returned",
        }
    for r in settlement.itertuples(index=False):
        yield {
            "geo_id": f"{r.cntycde}|{r.settlement}",
            "geo_level": "county_settlement",
            "geo_name": f"{r.cntycde}, {r.settlement}",
            "source_category": r.leg,
            "count": round(r.weight, 1),
            "basis": BASIS,
            "year": YEAR,
            "source_id": SOURCE_ID,
            "note": (f"{r.churches} churches with a settlement code; "
                     f"{r.attendance:.0f} attenders as returned"),
        }


def check(county, settlement, diag):
    """Print what a reader has to know before trusting a cell. Returns a failure count."""
    fails = 0
    print("English Church Census 2005 -> county and county x settlement structure")
    for k, v in diag.items():
        print(f"  {k:28s} {v:,.1f}" if isinstance(v, float) else f"  {k:28s} {v:,}")

    if diag["churches_offshore_dropped"] != 71:
        print(f"  ! expected 71 Channel Islands / Isle of Man churches, "
              f"got {diag['churches_offshore_dropped']}")
        fails += 1
    if diag["counties"] != 47:
        print(f"  ! expected 47 English counties, got {diag['counties']}")
        fails += 1

    total = county["weight"].sum()
    print("\n  leg              churches  corrected   share | counties  settlement cells")
    print("                                                 |  >=3 ch   >=3 ch (of weight)")
    for leg in sorted(county["leg"].unique()):
        c = county[county["leg"] == leg]
        s = settlement[settlement["leg"] == leg]
        thick_c = (c["churches"] >= 3).sum()
        thick_s = s[s["churches"] >= 3]
        cover = 100.0 * thick_s["weight"].sum() / s["weight"].sum() if len(s) else 0.0
        print(f"  {leg:14s} {c['churches'].sum():9d} {c['weight'].sum():10,.0f} "
              f"{100*c['weight'].sum()/total:6.1f}% | {thick_c:7d}   "
              f"{len(thick_s):4d} ({cover:5.1f}%)")

    # The coding gap, restated per leg, because it is the reason for the two tables.
    print("\n  settlement coding rate by leg (share of that leg's churches with a code):")
    rates = (settlement.groupby("leg")["churches"].sum()
             / county.groupby("leg")["churches"].sum() * 100.0)
    print("   " + "  ".join(f"{k} {v:.0f}%" for k, v in rates.sort_values().items()))
    if rates.max() - rates.min() < 20:
        print("  ! the coding gap has closed -- uk_split.py's within-leg construction was "
              "written for a 60-point spread and could be simplified")

    # Orthodoxy is expected to fail this and uk_split.py routes around it; the check is
    # here so that the day it stops failing, somebody notices.
    orth = settlement[settlement["leg"] == "orthodox"]
    if len(orth) and (orth["churches"] >= 3).sum() >= 10:
        print("\n  ! orthodox now has 10+ usable settlement cells -- re-read uk_split.py's "
              "country-of-birth placement, it may no longer be necessary")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="print the structure table and write nothing")
    args = ap.parse_args()

    county, settlement, diag = structure()
    fails = check(county, settlement, diag)
    if args.report:
        return 1 if fails else 0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = list(_rows(county, settlement))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows):,} rows -> {OUT}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
