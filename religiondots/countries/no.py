# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _no_counts():
    """Norway at the 11 counties of 2020-2023, from two halves of one census table.

    Norway keeps a roll of every faith and life-stance community, because the state pays each a
    grant per member, and SSB publishes it: Church of Norway membership per kommune (table
    12025) and every other community by county and five groups (08531, to 2020). That is
    `roll` (spec §3.1), and Finland and Sweden next door are drawn from self-identification, so
    it is printed by sources/no.py and not drawn here.

      * **Norwegian citizens, 4.79M.** ESS `rlgdnno`, `ctzcntr = Yes`. Rounds 5-9 are NUTS 2016
        (7 regions) and rounds 10-11 NUTS 2021 (6 regions), which do not nest; each category is
        drawn at the 7 regions, at the 4 units both vintages share, or at the national rate
        inside the county's residual, whichever is the finest level it passes at (§9cz).
      * **Foreign residents, 600k.** `cens_21ctz_r3` at NUTS 3 crossed with Pew.

    Viken straddles two NUTS 2016 regions and takes a population-weighted blend of them.
    """
    from no2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "no.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts3"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"no.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "no_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts3"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey, a nationality model and a communion split from a national list: nothing here is
    # `measured`.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


def _no_place_weight(place):
    """Norway's 356 kommuner, weighted by municipal population (GISCO's figure, 1 January 2020).

    The Greek weighter again. Norway is where it matters most of the Nordic three, because the
    counties are huge and nearly empty: Troms og Finnmark is 74,000 km² and 242,000 people, and
    without this its dots would spread evenly over the Finnmarksvidda.
    """
    if "pop" not in place.columns:
        print("  !! no_lau.gpkg has no `pop` column — run sources/no_geo.py")
        return None
    return _GrLauWeighter(place)


ENTRY = {
    "no": dict(
        name="Norway",
        name_in="Norway",
        source="ESS rounds 5-11 (citizens) + Eurostat census 2021 x Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        note_public=(
            "**Norway keeps a roll of every faith community, and this map draws what people "
            "say instead.** The state pays each church, mosque and humanist association a grant "
            "per member, and in 2020 the rolls put **67.7%** of Norway in the Church of Norway. "
            "The European Social Survey asks people whether they consider themselves as "
            "belonging to a religion, and in its two most recent rounds **32.2%** of Norwegian "
            "citizens named the Church of Norway and 60.0% named none. Sweden and Finland are "
            "drawn the same way, so the three can be compared across their borders. "
            "**The Church of Norway fell twelve points inside the survey.** It was 44.1% of "
            "citizens in 2010-2018 and 32.2% in the rounds since. Only the older interviews can "
            "be placed in Norway's seven regions, so the map takes the regional pattern from "
            "those and the national level from the newer ones. Trøndelag shows how far the two "
            "instruments part: 75.1% of it is on the church's rolls, and at **26.6%** it is one "
            "of the least Lutheran counties by what people say. "
            "**Islam is 3.7% of Norway and 6.5% of Oslo.** Muslim respondents were too few for "
            "the survey's regional pattern to pass this map's usual test, so it was checked "
            "against the rolls, which put the seven regions in nearly the same order, and it is "
            "drawn. The rolls put Oslo higher, at 9.6%, partly because they count members' "
            "children. "
            "**The free churches are the Bible belt.** Pentecostals, the Lutheran free churches "
            "and the other Protestant bodies are one answer on the survey, and it is **7.2%** of "
            "citizens in Rogaland and Agder against 1.0% in Nordland. "
            "**What this cannot do.** The regional pattern rests on 7,045 Norwegian citizens "
            "interviewed in 2010-2018, and Viken, which straddles two of the seven regions, gets "
            "a blend of both. Among citizens, Catholics, the Orthodox, Jews and the Eastern "
            "religions are drawn at the national rate inside each county, so the map says "
            "nothing about where they live. The Human-Etisk Forbund is on the same grant roll "
            "as the churches, but the survey has no humanist answer, so its members are inside "
            "no religion. Norway's 599,825 foreign citizens are drawn by nationality from the "
            "2021 census."),
        how="survey, 9,611 people; foreign residents by nationality",
        grain="counties, 490,000 people on average",
        gap_share=0.003465,
        gap=("0.35% of Norway: the 16,951 Norwegian citizens, 0.31%, who were asked about "
             "religion and declined; and the 1,730 people the 2021 census recorded as stateless "
             "or of unknown citizenship, who are in neither half"),
        counts=_no_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "no" / "no_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_no_place_weight,
        note="THE ROLL IS RICHER THAN §11k SAID AND IS STILL NOT DRAWN. SSB table 08531 has "
             "every grant-receiving community by county in five groups, 2010-2020 (the series "
             "ends because communities stopped reporting members' home municipality in 2021), "
             "beside 12025's Church of Norway share per kommune. It is `roll` (§3.1) and "
             "Finland and Sweden are self-identification, so sources/no.py prints it beside the "
             "survey instead. **THE ESS REGIONS ARE TWO NUTS VINTAGES THAT DO NOT NEST**: rounds "
             "5-9 are the 7 NUTS 2016 regions and rounds 10-11 the 6 NUTS 2021 ones, and the only "
             "geography both are unions of is four units, where a rank test has no resolution. "
             "Counted at the 11 counties of 2020-2023; Viken blends NO01, NO03 and NO02 by 2019 "
             "population. The variable is `rlgdnno` in all seven rounds, the harmonised card "
             "with Protestant split into the state church and the rest, proved by `_check_card`. "
             "**ISLAM IS AN OVERRIDE** (rank p 0.0555, chi-square 1.3e-10, and SSB's 2019 roll "
             "orders the same 7 regions at +0.929). **THE LEVEL IS RESCALED TO ROUNDS 10-11 "
             "(§3.4)** because the Church of Norway fell from 44.14% to 32.24% of citizens "
             "between the pools; one factor per category, because the roll's and the survey's "
             "own county changes are both proportional. The Orthodox answer is split 59.49 / "
             "40.51 Eastern / Oriental on the ministry's 2018 grant list, which overstates the "
             "Oriental share of a citizen cell because Norway's Eritreans are recent (the "
             "reverse of Sweden). GISCO's POP_2021 for Norway is 1 January 2020. sources/no.md, "
             "sources.md §9dd.",
    ),
}
