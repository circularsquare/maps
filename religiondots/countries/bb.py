# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _bb_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "bb_hexes.gpkg", "sources/bb_grid.py")


def _bb_counts():
    """BSS 2010 census at parish: 21 nodes on 11 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **98.77% of the TABULABLE population is drawn** — 223,419 of 226,193 — the missing part
    being `Not Stated`, 2,774 people, excluded by taxonomy/bb2010.py per §3.5.

    **BUT THE TABULABLE POPULATION IS NOT BARBADOS.** It is 226,193 of an estimated resident
    277,821: the 2010 census has an **18% undercount**, and BSS says so in its own Table A.
    So this country is drawn on 80.4% of its people, and — the part that matters for a map —
    **coverage is not uniform across parishes**, running from 74.6% of St. James to 96.1%
    of St. John. An under-covered parish therefore draws proportionally fewer dots than its
    true population warrants. Nothing here scales it (§14.4), because scaling would assume
    the people the census missed have the same religion mix as the people it found, and
    nothing establishes that.

    **`No Religious Affiliation` is 20.59%** and is spelled out rather than being the literal
    string `None`, so unlike Belize, Trinidad, the Bahamas and Cayman this file's read is not
    load-bearing on `keep_default_na=False`. It is passed anyway, for consistency and
    because `Not Stated` is the next thing that would go wrong.
    """
    from bb2010 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bb.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "parish"].copy()
    if df["geo_id"].nunique() != 11:
        raise SystemExit(f"{df['geo_id'].nunique()} parishes, expected 11 -- re-run "
                         "sources/bb.py")
    if "No Religious Affiliation" not in set(df["source_category"]):
        raise SystemExit("bb.csv has no `No Religious Affiliation` category -- that is "
                         "20.6% of Barbados and the column whose header the source leaves "
                         "BLANK (sources/bb.py). Re-run it.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"Not Stated", "Total"})
    if unmapped:
        raise SystemExit(f"bb.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "bb": dict(
        name="Barbados",
        source="2010 Population and Housing Census (Barbados Statistical Service)",
        basis="self-identification",
        view=[-59.72, 13.02, -59.37, 13.36],
        gap_share=0.18,
        gap=("the census's own 18% undercount; 49,115 people were never enumerated, unevenly "
             "between parishes"),
        note_public=(
            "**The most Anglican country on this map.** 23.9%, against 13.9% in Saint "
            "Vincent, 11.9% in the Bahamas, 5.7% in Trinidad and 2.8% in Jamaica. "
            "\"Little England\" is still legible in the answers: **St. John is 37.5% "
            "Anglican**, and the parish churches are the oldest institutions on the island. "
            "**The two big answers run opposite to each other across the island.** "
            "Anglicanism peaks in St. John (37.5%) and bottoms in St. Andrew (15.3%); the "
            "Pentecostal churches do the reverse — **29.4% of St. Andrew** against 19.5% "
            "nationally. The established church holds the south and east, the Pentecostals "
            "the rugged, poorer north-centre. "
            "**And this is the only census outside the United States that counts Nazarenes, "
            "Wesleyans and the Salvation Army as three separate answers.** 7,299 "
            "Nazarenes, 7,694 Wesleyans, 878 Salvationists, and a fourth Holiness cell — "
            "`Church of God` — on top: **9.4% of the country in one church family, split "
            "four ways**, which nothing else here does at all. They are not spread evenly. "
            "**St. Lucy is 13.4% Adventist** — more than twice the national 5.9% — and "
            "**2.3% Salvation Army against 0.4%**, six times the national share, in the "
            "northernmost parish. **St. Joseph is 7.2% Wesleyan** and **St. Thomas 5.7% "
            "Moravian**, against 1.2% nationally, which is the old Moravian mission field. "
            "**Roman Catholics are 3.8%** — very low for the Caribbean, and a reminder that "
            "Barbados was never Spanish or French. **20.6% report no religious "
            "affiliation**, the second largest answer, highest in St. Joseph (25.8%) and "
            "St. Michael (23.6%). "
            "**This is 2010, and Barbados has held a census since.** The 2021 census "
            "counted only **136,415 of an estimated 269,090 people — a 48.7% undercount** — "
            "and its own report says of the parish tables that \"most results at that level "
            "would be understated\". So the newer census is the worse map, and the older one "
            "is drawn. "
            "**Even 2010 is missing 18% of Barbados**, and unevenly: the census reached "
            "96.1% of St. John and only 74.6% of St. James. Nothing here scales the "
            "parishes back up, so an under-counted parish shows proportionally fewer dots "
            "than its true population warrants. What the composition inside each parish "
            "shows is unaffected."),
        how="census, 2010",
        grain="parishes, 20,600 people on average",
        counts=_bb_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bb" / "bb_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bb_place_weight,
        note="**THE NEWER CENSUS WAS REJECTED, ON THE PUBLISHER'S OWN WARNING.** BSS ran a "
             "2021 census and published the same Table 02.06 — *Total Population by Parish, "
             "Sex and Religion* — in `Census-2020-Tables.xlsx`, with the same 23 categories. "
             "It is not used: **2021 tabulated 136,415 of an estimated 269,090, a 48.7% "
             "undercount**, against 2010's 226,193 of 277,821 and 18%. And the 2021 report "
             "says outright that *\"in most cases, disaggregation by area is not included – "
             "as most results at that level would be understated, considering the "
             "significant size of the undercount\"* — while the workbook ships the parish "
             "cut anyway. The publisher's warning is taken over the publisher's "
             "spreadsheet. "
             "**THE `NO RELIGION` COLUMN HAS NO HEADER AND IT IS 20.6% OF THE COUNTRY.** In "
             "the 2010 sheet, column 23 sits between `Other Non-Christian` and `Not Stated` "
             "and its header cell is **blank** — 46,562 people, the second largest answer. "
             "A header-driven read names it `Unnamed: 23` or drops it. Two things identify "
             "it and `sources/bb.py` asserts both: the categories only sum to each unit's "
             "own `Total` when it is included, on all 36 rows; and **the 2021 workbook** "
             "publishes the same categories in the same relative order with that position "
             "labelled `No Religious Affiliation`. So the 2021 file, rejected as a source, "
             "is the evidence for reading the 2010 one. "
             "**KONTUR INDEPENDENTLY REPRODUCES THE CENSUS'S OWN UNDERCOUNT PATTERN, WHICH "
             "IS THE STRONGEST CHECK ON THIS COUNTRY.** BSS publishes an estimated resident "
             "population per parish, so the census's coverage can be computed: 74.6% of "
             "St. James up to 96.1% of St. John. Kontur's building-footprint grid knows "
             "nothing about any of that, and its modelled-to-tabulated ratio per parish "
             "correlates with the implied undercount factor at **r = +0.863** — a "
             "relationship 20,000 random relabellings of the parishes reproduce 0.05% of "
             "the time. Two unrelated sources agreeing on which parishes were "
             "under-enumerated. It also means the ratio table in `sources/bb_grid.py` is a "
             "COVERAGE read rather than a shape check, and is expected to sit above 1. "
             "**NOTHING IS SCALED UP** (§14.4): correcting the parishes would assume the "
             "missed 18% has the same religion mix as the counted 82%, and nothing "
             "establishes that. "
             "**THE JOIN IS 11/11 BOTH WAYS.** COD-AB's ADM1 is the parish tier exactly, "
             "and the parishes are the only sub-national geography Barbados publishes at "
             "all. Ten of the eleven names differ only as `St.` against `Saint`, handled by "
             "`fold()` rather than an alias table; BSS publishes no code, so `bb.py` carries "
             "COD's pcode by name and `bb_geo.py` asserts the pairing from the other side. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 705 hexes. Barbados is the densest "
             "country on this map and its parishes are unusually uniform in area — 23.9 to "
             "62.5 km², a 2.6-fold spread against Trinidad's 71-fold — so the grid is not "
             "here for empty land or unit size. It is here for **St. Michael**, which holds "
             "a third of the country on 40.7 km², nearly all of it in Bridgetown and the "
             "south-west coastal belt.",
    ),
}
