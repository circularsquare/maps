# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _BgGridWeighter(_KeHexWeighter):
    """Split an obshtina's dots across the census's OWN 1 km cells by cell population.

    Bulgaria is the third country here after Germany and Slovakia whose placement layer is
    measured rather than modelled. NSI's `POPGRID2021_1000M` is Census 2021 aggregated from
    the point location of every record, so no Kontur extract is read and no caveat about a
    modelled surface applies.

    IT COVERS 99.1% OF THE COUNTRY, and the missing 0.9% is NSI's own published figure for
    records with no usable point location (metadata §15.1, 58,198 people). That shortfall
    changes only the SHAPE of the weight inside an obshtina; every obshtina's dot total is
    still the workbook's.

    It is a POPULATION weight and not a religion one. Nothing measures where a municipality's
    Muslims sit inside it, so a Christian dot and a Muslim dot spread identically, and the
    map is religion by obshtina drawn where Bulgarians live. That matters more here than in
    Slovakia: 265 units over 110,000 km² is 415 km² each, sixteen times Slovakia's obce, and
    several Rhodope and Ludogorie municipalities are one town plus thirty villages.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on the census's own 1 km cell "
                f"population, {self.n_uniform:,} on equal shares where an obshtina's cells "
                f"sum to zero (sources/bg_geo.py)")


def _bg_place_weight(place):
    """countries.py hook. `place` is the 1 km grid GeoDataFrame scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! bg_grid_1km.gpkg has no `pop` column — run sources/bg_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _BgGridWeighter(place)


def _bg_counts():
    """Census 2021 at obshtina: 11 nodes on 265 units, three measured and eight derived.

    TWO FILES, ONE GEOGRAPHY. `bg.csv` is NSI's municipal workbook and supplies the three
    columns that are drawn as published (Judaism, other, no religion). `bg_split.csv` is
    bg_split.py's replacement for the two that are not: `Християнско` and `Мюсюлманско` are
    undivided at municipal level, and are split into denominations using the 2011 census's
    oblast composition raked to NSI's published 2021 national breakdown. The two source files
    partition the drawn population between them and never overlap, because taxonomy/bg2021.py
    EXCLUDES the two replaced columns.

    THE JOIN IS THE IDENTITY FUNCTION. NSI keys the workbook on its own obshtina codes and
    GISCO's LAU 2021 file carries those verbatim as `LAU_ID`, so `unit` is the workbook's
    `geo_id` unchanged and sources/bg_geo.py asserts the two 265-element sets are equal.

    EVERY SPLIT ROW IS `derived` AND CARRIES ITS OWN COLUMN (spec §7a-i-1). `parent_column`
    names the 2021 column the row came out of, which the census measured at this same
    obshtina, so `inferred dots: not shown` redraws `christianity` and `islam` rather than
    walking the tree. A derived row may not ring (§3.10).

    79.3% OF THE COUNTRY IS DRAWN and the three undrawn columns hold no religion anybody
    named: 259,235 who ticked `cannot determine`, 472,606 who ticked `do not wish to answer`,
    and 616,681 taken from administrative registers who were never asked at all. That last is
    mk2021.py's coverage-residual case. Unlike Slovakia's `ostatné` (§6.12) there is nothing
    recoverable inside them, because `Друго` is its own column beside them.
    """
    import bg2021
    from bg2021 import resolve

    base = pd.read_csv(HERE / "data" / "normalized" / "bg.csv",
                       dtype={"geo_id": str}, low_memory=False)
    base = base[base["geo_level"] == "obshtina"].copy()
    base["tier"] = "measured"

    split = pd.read_csv(HERE / "data" / "normalized" / "bg_split.csv",
                        dtype={"geo_id": str}, low_memory=False)
    if split.empty:
        raise SystemExit("bg_split.csv is empty — run `python bg_split.py`")

    df = pd.concat([base, split], ignore_index=True)
    df["unit"] = df["geo_id"].astype(str)
    if df["unit"].nunique() != 265:
        raise SystemExit(f"{df['unit'].nunique()} obshtini, expected 265")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    df = _add_roll(df, bg2021.COLUMNS)
    df["may_ring"] = df["tier"] == "measured"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier", "roll", "may_ring"]]


ENTRY = {
    "bg": dict(
        name="Bulgaria",
        source="Преброяване 2021 (National Statistical Institute)",
        basis="self-identification",
        view=[22.2, 41.1, 28.8, 44.3],
        gap_share=0.207,
        gap=("one person in five, 20.7%; two boxes for declining to answer a voluntary "
             "question, plus 9.5% taken from administrative registers and never asked"),
        note_public=(
            "**Two communities, and forty-seven municipalities where the smaller one is "
            "the larger.** Of the 5,903,108 people this census reached, **71.5%** answered "
            "Christian and **10.8%** Muslim, and the Muslim share is not spread thinly: it "
            "is **86.6%** in Ruen behind Burgas, **76.3%** in Dulovo, **74.5%** in Kirkovo, "
            "**54.3%** in the town of Kardzhali. Those places are three separate "
            "geographies rather than one, and the map shows them as such: the Rhodope "
            "mountains along the Greek and Turkish border, the Ludogorie plateau in the "
            "north east between Razgrad, Silistra and Shumen, and the villages inland from "
            "Burgas. "
            "**The denominations are not measured where they are drawn, and this is the "
            "one thing to know before reading the colours.** The 2021 census publishes a "
            "single Christian column for each of the 265 municipalities, and its "
            "denominational breakdown for the country only: **97.3%** of Bulgarian "
            "Christians Eastern Orthodox, 1.7% Protestant, 0.9% Catholic, 0.1% Armenian "
            "Apostolic. The 2011 census publishes that breakdown per oblast, 28 units. So "
            "each municipality's Christians are divided here using its oblast's 2011 "
            "composition, rescaled so every national total is the 2021 census's own "
            "published figure. Turn off inferred dots and the map returns to what was "
            "actually counted, one Christian colour and one Muslim one. The composition is "
            "assumed not to vary inside an oblast, which is where this is weakest: most of "
            "Bulgaria's **38,709** Catholics are Banat Bulgarians in Rakovski, and the map "
            "spreads them across the whole of Plovdiv oblast. Read the Orthodox share as "
            "solid and the small denominations as placed rather than found. "
            "**The same method separates the Alevi, and nothing else on this map holds "
            "them.** Bulgaria's Shia Muslims are the Kazalbash of Razgrad, Silistra, "
            "Targovishte and Sliven, **29,470** people here and the largest Shia population "
            "in Europe outside Turkey. This split is the weaker of the two: the census "
            "published no 2021 Sunni and Shia totals at all, so their proportions rest on "
            "the 2011 census alone rather than being rescaled to anything current. "
            "**A fifth of the country is blank, and where it is blank is not random.** "
            "Religion has been a voluntary question at every Bulgarian census since 1992. "
            "259,235 people said they could not determine their religion, 472,606 said "
            "they did not wish to answer, and a further 616,681 were never asked anything: "
            "the census could not reach them and their records came from administrative "
            "registers instead. Together that is 20.7% of Bulgaria and none of it is "
            "drawn. Between municipalities it runs from **3.1%** to **64.3%**, median "
            "15.9%, and the five highest are all in the Rhodopes, in municipalities with "
            "large Bulgarian-speaking Muslim populations: Nedelino 64.3%, Banite 55.8%, "
            "Zlatograd 52.5%, Laki 43.0%, Devin 37.6%. Sofia is 29.1%. Read a thin patch "
            "of dots in the Rhodopes as a question many people there left unanswered, not "
            "as an empty valley."),
        how="a census question, 2021",
        grain="municipalities, 24,600 people on average",
        fill=("from the 2011 census at oblast level, rescaled so each denomination's "
              "national total is the 2021 census's own"),
        counts=_bg_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bg" / "bg_grid_1km.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bg_place_weight,
        note="**THE QUEUE HAD BULGARIA AS 'hosts answer now; not chased further' AND BOTH "
             "HOSTS IT NAMED ARE DEAD ENDS.** `nsi.bg/en/content/6704/population-religion` "
             "returns 200 with the NSI **homepage**, which is the false positive §11o's "
             "*it answers* was reading, and `censusresults.nsi.bg` is the **2011** portal, "
             "which publishes religion by oblast and nothing finer. The route is NSI's own "
             "site search: `nsi.bg/search?q=вероизповедание` finds the press release, which "
             "leads to `statistical-data/151/1349`, which lists nine Census 2021 workbooks. "
             "**Sheet 4 of `Census2021_Ethnocultural characteristics_BG.xlsx` is religion "
             "by municipality**, 265 obshtini, two administrative tiers finer than the 2011 "
             "portal, in a 78 KB file. "
             "**THE PARTITION IS EXACT AND WITNESSED TWICE.** The eight category columns sum "
             "to `Общо` sums to 6,519,789 in every one of the 265 municipalities, the 28 "
             "oblasti and the country, with a difference of zero; and recomputing NSI's "
             "published headline shares from the municipal rows returns 71.48%, 10.82%, "
             "5.17%, 4.39% and 8.01%, which is its press release to two decimals. "
             "**THE UNSD ORACLE RETURNS VALUES AND SOURCES.MD §11R DOCUMENTS THE WRONG "
             "COLUMN SET.** §11r's `c=0,1,2,3,4,5,6` gives country, year and area with no "
             "numbers, which is why the oracle was only ever asked *does this office "
             "tabulate religion at all*. **`c=0,2,3,6,8,10,15,16` returns the counts** "
             "(column 16 is `Value`, and asking for 17 or more returns a zero-byte body "
             "with a 200, so the failure looks like a network fault). Used here it confirms "
             "the workbook to the person, including that NSI's two declining boxes are "
             "UNSD's single `Not Specified`: 259,235 + 472,606 = 731,841 exactly. "
             "**BOUNDARIES COST NOTHING**: GISCO LAU 2021 carries Bulgaria as 265 units "
             "whose `LAU_ID` is NSI's obshtina code verbatim, so the crosswalk is the "
             "identity function and §12's confident-wrong-pairing cannot arise. "
             "**AND THE PLACEMENT IS MEASURED**, the third country after Germany and "
             "Slovakia: `POPGRID2021_1000M` is Census 2021 aggregated from the point "
             "location of every record onto 1 km cells, holding 6,461,591 of the 6,519,789, "
             "with the 0.9% shortfall NSI's own metadata publishes. Its licence permits "
             "mapping products built on it but not showing individual cell values, which is "
             "how it is used here. **NOT an independent check on the counts** (§9av): the "
             "grid is the same enumeration. "
             "**THE DENOMINATIONS ARE DERIVED, AND THE FIRST BUILD REFUSED TO DERIVE THEM.** "
             "Built undivided, Bulgaria drew as one flat Christianity colour between four "
             "Orthodox neighbours; Anita, 2026-09-08: *\"bulgaria looks kinda out of place "
             "as it's the only one in the area where we dont have christianity breakdown to "
             "orthodox and not\"*, then *\"yes 2011 oblast composition is ideal.\"* The "
             "first version had cited §14.4 rule 1, and that reading was too narrow: **every "
             "magnitude in the output is NSI's**, the national denomination totals from 2021 "
             "and each obshtina's Christian and Muslim totals from 2021, with only their "
             "distribution modelled, from the same state's 2011 census. That is §14.10's "
             "amendment rather than rule 1's prohibition. `bg_split.py` rakes the 28x5 "
             "oblast table onto both sets of 2021 margins, then splits each obshtina by "
             "largest remainder so its rows sum to its own measured column exactly, and "
             "repairs the national totals with single-person moves inside one obshtina. "
             "**Three checks, one of which could have failed.** Eastern Orthodox is 97.44% "
             "of Christians in 2011 and 97.30% in 2021, so the ten-year-old shape is sound "
             "and the build fails if that drift ever exceeds half a point. Every obshtina "
             "sums to its measured column and every denomination to its published national "
             "figure, both to the person. And the Shia total, which nothing constrained, "
             "comes out at 29,470 against **27,579** for the 2021 Muslim residual UNSD's "
             "classification could not code, a 6.9% difference from a publication the model "
             "never saw. "
             "**THE 2011 REPORT SUPPRESSES SMALL CELLS AND `..` IS NOT `-`.** Nine oblasti "
             "withhold at least one religion; reading `..` as zero loses those people "
             "silently and made Blagoevgrad's eleven rows sum to 253,199 against a stated "
             "253,200. Seven are recovered exactly from the row total, two oblasti withhold "
             "two cells each and 4 people nationally stay unresolved.",
    ),
}
