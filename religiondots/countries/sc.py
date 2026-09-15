# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _sc_place_weight(place):
    """countries.py hook. `place` is the Kontur 400 m layer cut to the districts.

    Mahé's districts run from the coast up granite mountains, and Other Islands is 211 km2
    of atolls of which Aldabra alone is most and holds a research station. An equal-area
    spread would put dots on Morne Seychellois and on Aldabra's lagoon. Median 20 hexes a
    district (sources/sc_grid.py), so the population surface has room to say something.
    """
    return _kontur_place_weight(place, "sc_hexes.gpkg", "sources/sc_grid.py")


def _sc_counts():
    """NBS Population and Housing Census 2010, district supplement Table 3: 34 nodes on 26 units.

    ONE level, no allocation, nothing modelled; every row is `measured` and may ring.

    **THE DISTRICT TABLE IS DEEPER THAN THE NATIONAL ONE.** The report's Table 2.9 folds 26
    small write-ins into four rows; the supplement prints all 57 labels per district, and
    sources/sc.py asserts that the folds close exactly.

    **2022 IS NEWER, FINER BY ONE DISTRICT AND NOT DRAWN.** Its Table B4.1 folds everything
    to six answers, with no religion inside `Other`, and 11.5% is `Missing` because
    institutional households were not asked. It is used in sources/sc.py as a witness on the
    district pattern (Anglican rho 0.97, Catholic 0.71 across 23 unchanged districts).
    """
    from sc2010 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sc.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "district"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 26:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 26; re-run "
                         "sources/sc.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna() & (df["count"] > 0), "source_category"]
                      .unique())
    if set(unmapped) - {"Not stated"}:
        raise SystemExit(f"sc.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df.groupby(["unit", "node"], as_index=False)[["count", "congregations"]].sum()


ENTRY = {
    "sc": dict(
        name="Seychelles",
        source="Population and Housing Census 2010, Supplement Statistical Tables "
               "(National Bureau of Statistics)",
        basis="self-identification, whole enumerated population",
        note_public=(
            "**The 2010 census wrote down whatever religion people gave.** The form had "
            "boxes for seven answers and a line for anything else, and the National Bureau "
            "of Statistics kept every write-in in its district tables, 57 of them, down to "
            "a single Presbyterian. Catholics are **76.2%** of the country, from **88.5%** "
            "in Takamaka in the south of Mahé to 54.9% in Grand Anse Praslin and 36.1% in "
            "Other Islands. "
            "**Grand Anse Praslin is 29.2% Anglican**, against 6.2% for the country and "
            "7.4% in Baie Sainte Anne, the other district on Praslin. Saint Louis and Bel "
            "Air, in Victoria, are next at 11.3% and 10.5%; in Takamaka, Port Glaud, Baie "
            "Lazare and La Digue Anglicans are under 2%. "
            "**Other Islands is where the census found hotel and construction workers.** "
            "It covers every island away from Mahé, Praslin and La Digue and their near "
            "neighbours, and in 2010 it held 1,042 people, 574 of them not Seychellois and "
            "825 of them men. Hindus are **11.7%** there against 2.4% nationally, Buddhists "
            "3.0%, and 37.1% did not state a religion. "
            "**Pentecostal churches are the largest group after the Anglicans.** The "
            "Pentecostal Assemblies counted 1,333 people and the Assembly of God 831, and "
            "the Nigerian-founded Redeemed Christian Church 394. The first two together are "
            "5.9% of Baie Sainte Anne, twice their share anywhere else but Au Cap, Port "
            "Glaud and Roche Caiman. "
            "**4,328 people, 4.8%, did not state a religion**, and are not drawn. They are "
            "not spread evenly: 21.3% of Roche Caiman, 15.1% of Cascade and 12.3% of "
            "English River did not answer, against under 1% in nine districts. "
            "The 2022 census asked again and publishes religion by district, but folds the "
            "answers to six and leaves 11.5% missing, because people in institutions got a "
            "form without the question. Counting the missing in the total, its national "
            "table puts Catholics at 61.3%, Hindus at 5.4% and Muslims at 2.4%, against "
            "2.4% and 1.6% in 2010. "
            "English River's dots are kept off Perseverance Island: the reclaimed island "
            "was part of the district in 2010, but its housing was built afterwards."),
        how="census, 2010, all ages",
        grain="districts, 3,500 people on average",
        gap="4.8% of Seychelles, who did not state a religion",
        gap_share=0.0476,
        counts=_sc_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sc" / "sc_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sc_place_weight,
        view=[55.1, -4.85, 56.05, -3.65],
        note="THE 2010 DISTRICT SUPPLEMENT CARRIES RELIGION IN THE WRITE-IN CATEGORIES. "
             "§11w priced Seychelles off the UNSD oracle's 2002 row (11 categories, national) "
             "and did not chase the office. The 2010 census post-coded the religion "
             "write-ins, and `Population and Housing Census 2010: Supplement Statistical "
             "Tables (ALL DISTRICTS)` (IHSN catalogue 4079; nbs.gov.sc no longer lists it) "
             "prints Table 3, religion by sex, for each of the 26 districts. "
             "THE PARSE CLOSES THREE WAYS: each row's sexes, each district against its own "
             "age table, and the districts against the report's national Table 2.9 with its "
             "four folds spelled out in sources/sc.py. The report's Table 2.3 misprints Other "
             "Islands as 576 (the supplement's 1,042 closes its own total). "
             "TWO BOUNDARY MOVES. COD-AB (NBS 2010 districts plus GAUL islands) draws "
             "Perseverance Island as a separate feature and puts Silhouette, North, Félicité, "
             "Marianne, the Soeurs and Cocos in Other Islands; the census counts the island "
             "with English River and the six with La Digue. Both are confirmed by Table 2.3's "
             "printed areas (English River 2.32 km2 merged against 2.3 printed, 1.38 "
             "without; La Digue 36.37 against 36.4). "
             "PLACEMENT is Kontur 2023 cut by overlap, Perseverance's cells dropped, r = 0.89 "
             "over 26 districts. See sources/sc.md.",
    ),
}
