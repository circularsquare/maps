# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _na_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    14 regions over 824,000 km2, //Kharas alone 161,000 km2 with 110,000 people; most regions hold
    their people in a few towns and, in the north, along the Owambo and Kavango settlement belts
    (sources/na_geo.py). No Kontur block reaches the density cap (`kontur_cap.py na`, 2026-10-03).
    """
    return _kontur_place_weight(place, "na_hexes.gpkg", "sources/na_geo.py")


def _na_counts():
    """Six pooled Afrobarometer rounds on the 2023 census's region counts: 10 nodes, 14 regions,
    and EVERY ROW IS `modelled` IN §7.

    NAMIBIA'S CENSUS DOES NOT ASK RELIGION (2001, 2011 and 2023 forms), so as in Cameroon and
    Madagascar there is no margin to fit and no measured tier. sources/na.py has the construction:
    13 survey mixes (Kavango whole), Lutheran and Anglican from rounds 5-6 inside the pool they
    trade with, the DHS 2013 as the level witness, none and traditional placed as one box.
    """
    from na2023 import resolve

    # keep_default_na=False: pandas reads the category `None` as missing otherwise (§11aq).
    df = pd.read_csv(HERE / "data" / "normalized" / "na.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "na" / "na_lookup.csv", dtype=str,
                      keep_default_na=False)
    if set(df["geo_id"]) != set(lut["geo_id"]) or len(lut) != 14:
        raise SystemExit("na.csv regions do not match na_lookup.csv's 14 -- re-run "
                         "sources/na_geo.py and sources/na.py")
    if "None" not in set(df["source_category"]):
        raise SystemExit("na.csv has no `None` rows; it was read with the default NA list")
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"na.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "na": dict(
        name="Namibia",
        source="six pooled rounds of the Afrobarometer, 2008 to 2021, on the region counts of "
               "the 2023 Population and Housing Census (Namibia Statistics Agency)",
        basis="self-identification, whole census population",
        note_public=(
            "**Namibia's census does not ask about religion.** The 2001, 2011 and 2023 census "
            "forms have no religion question, so this map's Namibia comes from a survey. Six "
            "rounds of the Afrobarometer are pooled, **7,144** adults interviewed between October "
            "2008 and November 2021, and nobody counted these dots, so they disappear when "
            "inferred dots are turned off. Each "
            "region is drawn at the mix its own respondents gave and at its 2023 census "
            "population. The 2008 and 2012 rounds sampled Kavango before it was split, so Kavango "
            "East and Kavango West are drawn with one shared mix. The survey interviews citizens; "
            "the 4.8% of residents who are not Namibian, most of them Angolan, are drawn at the "
            "mix of the region they live in. "
            "**The Lutheran figure comes from two of the six rounds.** The share of people "
            "answering Lutheran was 42.8% in 2012 and 41.9% in 2014 but 20 to 24% in the other "
            "four rounds, where the missing Lutherans were recorded as Evangelical, as Anglican "
            "or as Christian without a church, differently in each round. The 2013 Demographic "
            "and Health Survey asked women and men aged 15 to 49 and found **43.9%** in the "
            "Evangelical Lutheran Church in Namibia, the only Lutheran church its card named, "
            "which agrees with the 2012 and 2014 rounds. So Lutherans and Anglicans are drawn from those two rounds, and everything "
            "else from all six. Lutherans come out at **40.5%**, Catholics at 22.6% (the 2013 "
            "survey found 21.6%) and Anglicans at 6.6%. The survey does not separate the northern "
            "Lutheran church from the one in the centre and south. "
            "**Each church has its own part of the country.** Lutherans are 67.6% of Oshikoto and "
            "about half of Omusati, Ohangwena and Oshana; in Ohangwena 30.6% are Anglican. "
            "Catholics are 47.2% of the two Kavango regions and a third of //Kharas and Omaheke. "
            "Seventh-day Adventists are 41.0% of Zambezi, and 2.9% nationally against the 4.5% "
            "the 2013 survey found, so they are probably drawn short there. "
            "**No religion and traditional religion are placed together.** The survey offered "
            "both answers in every round, but they traded places: in Kunene, where Himba "
            "communities keep the ancestral fire, 38 to 43% gave traditional religion in 2012 and 2014 "
            "and 16 to 18% gave no religion in 2017 and 2021. The two are drawn as one share in "
            "each region, a quarter of Kunene and 1 to 9% elsewhere, and split at the ratio of "
            "all six rounds, about two in three no religion. "
            "**Muslims and other religions are not placed.** The survey found five Muslims in six "
            "rounds, and almost every answer of another religion came from 2017 on, so both are "
            "spread at their national share over every region."),
        how="a pooled survey, 2008 to 2021, on 2023 census region populations",
        grain="regions, 216,000 people on average",
        counts=_na_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "na" / "na_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_na_place_weight,
        note="NAMIBIA'S CENSUS DOES NOT ASK: the 2001, 2011 and 2023 forms have no religion item "
             "(sources.md §11aq); freed by the 2026-10-03 negatives scout. Drawn on Anita's "
             "Nigeria ruling (ask/answered/010-ng): each region at its own pooled survey mix, every "
             "row modelled, the national level computed. LUTHERAN AND ANGLICAN COME FROM ROUNDS 5 "
             "AND 6 ONLY, as a share of the pool they trade with, because those rounds reproduce "
             "the 2013 DHS's ELCIN share (43.9%) and the other four rounds sit 20 points under it. "
             "Catholic, Adventist, Pentecostal from all six rounds. NONE AND TRADITIONAL ARE PLACED "
             "AS ONE BOX and split at the pooled ratio (68.1% none). Other and Muslim flat. "
             "Kavango East and West share one mix (rounds 4-5 sample Kavango whole). "
             "sources/na.md has the record.",
    ),
}
