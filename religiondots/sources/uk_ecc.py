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
the opposite way, 14% to 54%.

**THE COLLAPSE IS TO TWO -- urban and rural -- AND THAT IS FORCED BY ONS, NOT CHOSEN.**
Brierley's coders assigned `inner city`, `council estate`, `suburb-suburban fringe` and
`city centre` by eye from the address, and nothing published for a 2021 Output Area can
reproduce that. The 2021 Rural Urban Classification, which is the only one keyed on
`OA21CD`, has six categories that are three settlement sizes (urban, larger rural, smaller
rural) crossed with proximity to a major town -- ONS calls the change from 2011 a
"streamlining of the taxonomy", and what it streamlined away is exactly the settlement
detail this file would have used. The 2011 RUC does carry major conurbation / minor
conurbation / city and town, but only for 2011 Output Areas, and joining it forward is
sources/uk.md §4's live trap: 2021 OAs reuse 2011 codes wherever the area was unchanged,
so a wrong-vintage join partly succeeds and only the split and merged areas go missing.

What two classes still buy, and what the third would have: rural attendance is 53.4%
Anglican and 14.9% Catholic against urban's 31.8% and 34.8% -- the Catholic share more
than doubles and the Anglican share falls 22 points, which is the bulk of the signal. The
tier given up is Brierley's `separate town`, whose mix (28.5 Anglican / 29.4 Catholic /
42.1 other) differs from a conurbation's mainly in the `other` leg -- the chapel towns --
and `other` is the least trustworthy leg in this join for an unrelated reason. So the
third tier would have refined the weakest number on the map. Worth revisiting only if ONS
publishes a settlement-type classification on 2021 Output Areas.

**THIS FILE EMITS MEAN CONGREGATION SIZE AND NOT ATTENDANCE TOTALS, AND THAT IS A
CORRECTION.** It used to emit totals, response-corrected by the national rate the user
guide publishes per denomination. That produced a map on which Merseyside was 72% Anglican
and 16% Catholic, which is not Liverpool.

The reason is in the user guide and it defeats any national correction. Alongside the
postal returns the census took bulk data from "ten Church of England and eight Roman
Catholic Dioceses, the Baptist Union of Great Britain, the Fellowship of Independent
Evangelical Churches, the Salvation Army and 91 Methodist Circuits". The Baptist Union,
FIEC and Salvation Army supplied nationally, so their coverage is even. **The Anglican,
Catholic and Methodist supplies were diocese by diocese, so their coverage is not.**
Measured against the Church of England's own complete register of its churches, the
Anglican response rate is:

    nationally     56.2%   (the user guide says 55%, so the measurement is sound)
    Norfolk        25.9%   Durham 35.5%   Tyne & Wear 37.3%
    Merseyside     91.4%   Gloucestershire 92.3%   Greater Manchester 92.5%

A factor of three and a half, and the high counties are the ten dioceses. So a county's
total is partly a fact about English religion and partly a fact about which bishop's office
had a spreadsheet, and nothing in this file can separate them.

**A mean survives that; a total does not.** Losing half a county's Methodist chapels to
non-response changes how many you saw, not how big they were. So the quantity emitted is
attendance per church, and `sources/uk_churches.py` supplies the church counts from a
current register — the CofE's own list for Anglicans, OpenStreetMap for the rest, validated
against it at r = 0.97. `uk_split.py` multiplies the two.

The published per-denomination response rates are kept in `RESPONSE` below, unapplied,
because they document the bias rather than fix it.

**IT NEVER REPRODUCED THE PUBLISHED TOTAL AND NOW DOES NOT TRY.** The user guide is
explicit: "all publications based on the census contain estimates of the total churchgoing
population and cannot be derived directly from this dataset, which includes only half of
the churches in the country." Brierley's grossing-up used per-county and per-size
constraints this file does not have. Nothing downstream reads a total from here.

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

# Denominational response rates, English Church Census 2005 user guide p.4. **NOT APPLIED.**
# Kept because they document why totals from this source are unusable: a national rate
# cannot describe a response that varied threefold between counties. See the docstring.
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

# Brierley's eight settlement types -> the two ONS's RUC21 can reproduce for a 2021 Output
# Area. See the module docstring for why this is a collapse and not a mapping, and why it
# is two rather than three.
SETTLEMENT = {
    "City centre": "urban",
    "Inner city": "urban",
    "Council estate": "urban",
    "Suburb-suburban fringe": "urban",
    "Separate town": "urban",
    "Other built up area": "urban",
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

    # A church that met and reported nobody is a real zero and belongs in the mean; a
    # church that did not meet on 8 May is not a congregation size and does not. The user
    # guide's §3 describes the second case -- 973 churches, mostly linked rural benefices
    # whose service was held elsewhere that Sunday and whose people were counted there.
    meeting = df[(df["attendance"] > 0) | (df["vcntsrv"].fillna(0) == 0)]
    congregations = df[df["attendance"] > 0]

    def agg(frame, keys):
        g = frame.groupby(keys, as_index=False).agg(
            churches=("churchno", "count"),
            attendance=("attendance", "sum"))
        g["mean_congregation"] = g["attendance"] / g["churches"]
        return g[g["churches"] > 0]

    county = agg(congregations, ["cntycde", "leg"])
    coded = congregations[congregations["settlement"].notna()]
    settlement = agg(coded, ["cntycde", "settlement", "leg"])

    diag = {
        "churches_read": n_all,
        "churches_offshore_dropped": n_offshore,
        "churches_used": len(df),
        "churches_with_attendance": len(congregations),
        "churches_no_service": len(df) - len(meeting),
        "churches_with_settlement": len(coded),
        "settlement_coded_pct": 100.0 * len(coded) / len(congregations),
        "counties": congregations["cntycde"].nunique(),
        "settlement_cells": settlement.groupby(["cntycde", "settlement"]).ngroups,
        "attendance_raw": float(df["attendance"].sum()),
    }
    return county, settlement, diag


def _rows(county, settlement):
    """`count` is MEAN CONGREGATION SIZE, not a number of people in the county.

    Every consumer multiplies it by a church count from sources/uk_churches.py. A reader
    who sums this column has computed nothing.
    """
    for r in county.itertuples(index=False):
        yield {
            "geo_id": r.cntycde,
            "geo_level": "county",
            "geo_name": r.cntycde,
            "source_category": r.leg,
            "count": round(r.mean_congregation, 2),
            "basis": BASIS,
            "year": YEAR,
            "source_id": SOURCE_ID,
            "note": (f"mean congregation over {r.churches} churches; "
                     f"{r.attendance:.0f} attenders as returned"),
        }
    for r in settlement.itertuples(index=False):
        yield {
            "geo_id": f"{r.cntycde}|{r.settlement}",
            "geo_level": "county_settlement",
            "geo_name": f"{r.cntycde}, {r.settlement}",
            "source_category": r.leg,
            "count": round(r.mean_congregation, 2),
            "basis": BASIS,
            "year": YEAR,
            "source_id": SOURCE_ID,
            "note": (f"mean congregation over {r.churches} churches with a settlement "
                     f"code; {r.attendance:.0f} attenders as returned"),
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

    print("\n  leg            churches   mean congregation   counties  cells")
    print("                            England  urban  rural    >=3 ch  >=3 ch")
    for leg in sorted(county["leg"].unique()):
        c = county[county["leg"] == leg]
        s = settlement[settlement["leg"] == leg]
        nat = c["attendance"].sum() / c["churches"].sum()
        by_s = (s.groupby("settlement")["attendance"].sum()
                / s.groupby("settlement")["churches"].sum())
        print(f"  {leg:13s} {c['churches'].sum():8d} {nat:9.0f} "
              f"{by_s.get('urban', float('nan')):6.0f} {by_s.get('rural', float('nan')):6.0f}"
              f"    {(c['churches'] >= 3).sum():7d} {(s['churches'] >= 3).sum():7d}")

    # The coding gap, restated per leg. It no longer biases anything -- a mean is a
    # within-leg quantity -- but it still decides how many cells have their own mean.
    print("\n  settlement coding rate by leg (share of that leg's churches with a code):")
    rates = (settlement.groupby("leg")["churches"].sum()
             / county.groupby("leg")["churches"].sum() * 100.0)
    print("   " + "  ".join(f"{k} {v:.0f}%" for k, v in rates.sort_values().items()))

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
