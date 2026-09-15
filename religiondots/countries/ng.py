# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ng_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    37 states over 924,000 km2 is 25,000 km2 a unit, four times Liberia's counties and the
    coarsest counting tier on this map. Lagos is 13.5 million people on 3,671 km2 and Niger
    State is 6.8 million on 71,934 km2; Borno, Yobe and Taraba are 44,000 to 72,000 km2 apiece
    with most of their people in a band along the roads and the Sahel end close to empty. Drawn
    flat, Kano's colour would spread evenly over a state that also holds the emptiest country
    in northern Nigeria (sources/ng_grid.py).
    """
    return _kontur_place_weight(place, "ng_hexes.gpkg", "sources/ng_grid.py")


def _ng_counts():
    """Six pooled Afrobarometer rounds on COD-PS state populations: 5 nodes, 37 units, and
    EVERY ROW IS `modelled` IN §7.

    NIGERIA HAS PUBLISHED NO RELIGION COUNT SINCE 1963 and the 1973 census that asked was
    annulled with nothing released, so unlike Liberia there is no census margin to fit to and
    no measured tier anywhere in this country. Each state is drawn at the mix its own
    respondents gave and at its COD-PS 2022 population, so the national balance is a
    consequence rather than an input; sources/ng.py's docstring has the argument, and the
    reason it is NOT fitted to the NDHS or to Pew.

    THREE STATES ARE DRAWN WITH NO MUSLIMS AT ALL -- Abia, Cross River and Ebonyi -- because
    none of their pooled respondents was one, and §3.5 drops rather than invents. On samples
    of 198 to 256 the true share could be a couple of per cent, which is what note_public
    says.
    """
    from ng2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ng.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "ng" / "ng_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ng.csv states with no polygon: {missing} -- re-run "
                         "sources/ng_geo.py, the lookup is stale")
    if df["unit"].nunique() != 37:
        raise SystemExit(f"{df['unit'].nunique()} states, expected 37")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"ng.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ng": dict(
        name="Nigeria",
        source="six pooled rounds of the Afrobarometer, 2008 to 2022, on the 2022 state "
               "population projection (UNFPA and the National Population Commission)",
        basis="self-identification, whole projected population",
        note_public=(
            "**Nigeria has not counted religion since 1963, and that is a decision rather "
            "than an oversight.** The 1973 census asked and was annulled with nothing "
            "published, amid allegations that the returns had been falsified; the 1991 and "
            "2006 censuses left the question off, and the National Population Commission has "
            "said the postponed 2023 census will leave it off as well. The reason given is "
            "always the same one, that Nigerian revenue and Nigerian offices are shared out "
            "by population, so a Christian and Muslim count is also a count of who is owed "
            "what. The country with the largest Muslim population and the largest Christian "
            "population in Africa has no official figure for either. "
            "**So the whole of this map's Nigeria comes from a survey.** Six rounds of the "
            "Afrobarometer pooled, **11,909** people interviewed between May 2008 and April "
            "2022, and every dot is desaturated because nobody counted it. Each state is "
            "drawn at the mix its own respondents gave and at its projected population, so "
            "the national balance here, **51.4%** Christian against **47.9%** Muslim, is a "
            "result rather than an input. It is one figure among several and it sits at the "
            "Christian end of them: the 2018 Demographic and Health Survey found **53.5%** "
            "Muslim among Nigerians aged 15 to 49, and Pew's 2020 estimate is 56.1% Muslim. "
            "Eight points separate the highest of these from the lowest, which is 18 million "
            "people, so read the national number as contested and the state pattern as the "
            "part this survey is good at. "
            "**The state pattern is the strongest signal in the data.** Ranked across the 37 "
            "states, the three earlier rounds and the three later ones agree at **+0.96** "
            "for Christianity and +0.95 for Islam, against a bar of +0.33. The North West is "
            "92.5% Muslim and the South East 99.0% Christian, and the line between them runs "
            "through Kaduna at **66.6%** Muslim, Adamawa, Gombe and Kwara. Oyo is the one "
            "state the survey puts within two points of even. "
            "**There is one outside thing to check that against and it is a legal record "
            "rather than a table.** Twelve northern states extended the Sharia penal code to "
            "criminal matters between 1999 and 2001 and twenty-five did not. The survey "
            "makes every one of those twelve Muslim-majority, without having been told which "
            "twelve they are, and it also makes Adamawa, Kwara and Oyo Muslim-majority, "
            "which are the three states usually described as religiously mixed. "
            "**Abia, Cross River and Ebonyi are drawn with no Muslims at all**, because none "
            "of the 198 to 256 people interviewed in each of them across six rounds was one. "
            "Every large Nigerian town has a northern trading quarter, so read those three "
            "as states where the survey found none rather than as states with none. "
            "**Christianity and Islam are each one colour here, and the survey could have "
            "split both.** Its card names about twenty Christian denominations and, on the "
            "Muslim side, Sunni, Shia, Ismaili, Izala and three Sufi brotherhoods, and "
            "Nigerians fill nearly all of them. The share who name a denomination rather "
            "than answering just Christian swings **27.9** points between rounds with no "
            "trend, so a pooled Catholic or Pentecostal share would be a measurement of how "
            "hard that round's fieldwork probed. The Muslim card is worse than that: the box "
            "for Izala is on it in three rounds and off it in the other three. "
            "**Traditional religion is 0.32% here and is a floor rather than a count.** The "
            "survey offers it as an alternative to Christianity and Islam, so it cannot see "
            "anyone who is both, and in Nigeria a great many people are. Ifa and orisha "
            "practice in the south west, Odinala in the south east and the masquerade "
            "societies of the middle belt all run through people who answer Christian or "
            "Muslim when asked for a religion. It is drawn at the national rate in every "
            "state, because at that size the survey cannot say where it is. So is irreligion "
            "at **0.17%**, and the separate atheist and agnostic boxes took two people "
            "between them in fourteen years. "
            "**The population behind the dots is a projection off a census that was itself "
            "disputed.** Nigeria last enumerated in March 2006 and gazetted **140,431,790** "
            "people; Lagos State ran a parallel count and put itself at roughly twice its "
            "gazetted 9,113,605, and the northern totals were argued over in the other "
            "direction. What is drawn here is that base projected to 2022, **216,798,930** "
            "people. The disputed part of 2006 is the north and south balance of the totals, "
            "which is the same axis this map runs along."),
        how="a pooled survey, 2008 to 2022, on projected state populations",
        grain="states, 5.9 million people on average",
        counts=_ng_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ng" / "ng_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ng_place_weight,
        note="NIGERIA PUBLISHES NOTHING. The last census to publish religion was 1963; the "
             "1973 census asked and was annulled in 1975 with nothing released; 1991 and "
             "2006 did not ask; Nigeria is ABSENT from the UNSD oracle. sources/ng.md lists "
             "what was searched. So there is no census margin to fit to, which is the whole "
             "difference from Liberia (§9cl) on the same instrument. "
             "THE NATIONAL LEVEL IS COMPUTED, NOT MEASURED AND NOT FITTED. Each state is "
             "drawn at its own measured Christian/Muslim ratio and at its COD-PS 2022 "
             "population, so the country's balance falls out of the state pattern and the "
             "state populations. That is 51.4/47.9 and NOT the Afrobarometer's own pooled "
             "56.0/43.3: the pool's state mix is not COD-PS's, because round 6 has no "
             "Adamawa, Borno or Yobe and because pooling fourteen years averages over a "
             "period in which the northern states grew fastest. An earlier version fitted "
             "the column margin to the survey's own national shares by IPF and that was "
             "wrong in a way worth remembering: with a population table as the row margin, "
             "fitting the columns back to the survey UNDOES the reweighting the row margin "
             "just did. "
             "IT IS NOT FITTED TO THE NDHS OR TO PEW EITHER, and sources/ng.py says why. "
             "Pew is a synthesis partly built on this same survey, and §3.1 does not let an "
             "`estimate` set a `self_id` magnitude; the NDHS is 15-to-49 only, and scaling "
             "that up to a whole population is what §3.4 refused for Brazil. Both are "
             "printed on every build and both are in note_public. "
             "THE DENOMINATIONS AND THE SUNNI/SHIA SPLIT ARE IN THE SURVEY AND ARE NOT "
             "DRAWN. §11ai's probing swing is 27.9 points here, and report_card() in "
             "sources/ng.py reads each round's value labels to show which answers were on "
             "that round's card at all: Izala is on three of six and the Shia box is renamed "
             "between rounds. Shia is refused under §14.4 rule 2 as well as on the "
             "arithmetic; the Islamic Movement in Nigeria has been proscribed since 2019.",
    ),
}
