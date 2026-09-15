# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ug_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Uganda needs this for §8.2's emptiness reason: 56 districts over about 200,500 km2 of
    land is roughly 3,600 km2 a unit, and the country is not evenly habitable at that
    scale. Karamoja is a fifth of the area and a twentieth of the people; Lake Kyoga's
    swamp runs through the middle of four districts at once; Murchison Falls sits inside
    Masindi and Gulu and Queen Elizabeth inside Kasese and Bushenyi. An equal share per
    polygon puts Uganda's dots in national parks and papyrus.

    THE GRID IS 2023 AND THE COUNTS ARE 2002, which is stated rather than hidden: it is a
    within-district weight, so the level does not matter, but a 2023 surface places a
    district's dots where its people live now (sources/ug_grid.py).
    """
    return _kontur_place_weight(place, "ug_hexes.gpkg", "sources/ug_grid.py")


def _ug_counts():
    """UBOS 2002 census Table B7 at district: 7 drawn categories on the 56 districts of 2002.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    56 UNITS AND NOT 4, AND THE GEOGRAPHY IS THE ONLY THING UGANDA HAS. The 2024 census
    has the best religion category list in Africa, ten cells separating the Adventists,
    the Orthodox and the Witnesses, and it publishes them for the nation and for
    urban/rural and nowhere else; so do 2014 and the whole NPHC 2024 portal, which reaches
    PARISH level on fifteen other tables. The 2002 analytical report reaches four regions.
    Table B7, an annex table on the retired ubos.org tree, is the only religion tabulation
    ever published beside a Ugandan geography. sources/ug.py lists what was checked.

    THE BOUNDARIES ARE REBUILT AND THE REBUILD IS PROVED. No 2002-vintage boundary file
    exists, so the 56 districts are dissolved out of COD-AB's 135 of 2020 using the 2002
    census's own district/county/sub-county tree, and the result reproduces all 56 of
    Table B1's 1991 populations exactly, on 56 distinct values, from a table published in
    a different census twelve years later. See sources/ug_geo.py.

    `None` IS THE LITERAL STRING AND PANDAS DELETES IT — §12's Philippine trap, fourth
    sighting after `ph`, `gy` and `zw`. 212,388 irreligious Ugandans, and in Uganda they
    are not where a reader would guess: half of that cell is in Karamoja.
    """
    from ug2002 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ug.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    if "None" not in set(df["source_category"]):
        raise SystemExit("ug.csv has no `None` category -- it has been read as NaN, and "
                         "212,388 people are about to disappear (§12, the Philippine trap)")
    df = df[df["geo_level"] == "district_2002"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "ug" / "ug_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ug.csv districts with no polygon: {missing} -- re-run "
                         "sources/ug_geo.py, the lookup is stale")
    if df["unit"].nunique() != 56:
        raise SystemExit(f"{df['unit'].nunique()} districts, expected 56")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "ug": dict(
        name="Uganda",
        source="2002 Population and Housing Census, Table B7 (Uganda Bureau of Statistics)",
        basis="self-identification, whole census population",
        view=[29.4, -1.6, 35.2, 4.4],
        note_public=(
            "**Uganda's best religion question and its only religious geography are "
            "twenty-two years apart, and this map is the older one.** The 2024 census "
            "separates ten answers, including the Adventists, the Orthodox and the "
            "Jehovah's Witnesses, and publishes them for the nation and for town against "
            "country and nowhere else. So does 2014. The census portal that carries the "
            "2024 results down to parish level carries fifteen tables and religion is in "
            "none of them. The one time the Bureau ever printed religion beside a place "
            "was an annex table of the 2002 census, seven categories on the 56 districts "
            "of the day, and that is what is drawn here. "
            "**Catholic and Anglican Uganda are close to each other's negative, and the "
            "line is where two missions met in the 1890s.** Catholicism is the largest "
            "answer in 29 of the 56 districts and the Church of Uganda in 25. The Catholic "
            "north runs to **82.5%** in Adjumani and 78.2% in Gulu; the Anglican south and "
            "west to **60.8%** in Nakasongola and 60.6% in Ntungamo. A map of four "
            "regions, which is as fine as the Bureau's own 2002 analysis ever got, shows "
            "almost none of it. "
            "**Yumbe is 76.2% Muslim and the districts around it are not.** Islam is 12.1% "
            "of the country and the sharpest column in the table: Yumbe, which is Aringa "
            "county in West Nile, against 0.4% in Kotido and in Pader. The other Muslim "
            "Uganda is the Busoga lakeshore and the old Buganda trading towns, Mayuge at "
            "36.2% and Iganga at 33.8%, and the two have nothing to do with each other. "
            "**Pentecostal Uganda is the thing this map is too old to show.** It is 4.6% "
            "here, strongest in Sebei and Teso rather than in Kampala, and 14.3% by the "
            "2024 census, which is the largest movement in Ugandan religion in living "
            "memory and has no published geography at all. Read every share here as 2002: "
            "Catholicism is 41.9% on this map and 36.2% in the 2024 census, and the Church "
            "of Uganda 35.9% here and 29.0% there. "
            "**In Karamoja the census offered five churches and Islam to people who "
            "practise neither, and recorded the answer twice.** `Other`, which the table's "
            "footnote says holds the Orthodox, the Baha'is, other Christians and "
            "traditional religion together, is **28.2%** of Kotido; no religion is 12.0% "
            "of Nakapiripirit, and the three Karamoja districts hold 52% of the whole "
            "country's no-religion cell between them. Neither figure should be read the "
            "way the same box is read in Europe. "
            "**And Kotido's own numbers are ones the Bureau later withdrew.** This table "
            "gives the district 591,870 people; the 2014 census report, redistributing the "
            "same census onto later boundaries, gives the same ground **377,102**, and for "
            "the other 55 districts the two publications agree to the person. Kotido holds "
            "22.5% of the national `Other` cell and 33.1% of the no-religion cell, so both "
            "are affected: without it the country reads 2.41% `Other` and 0.60% no "
            "religion instead of 3.04% and 0.87%. It is drawn as published, because the "
            "Bureau revised a population and never revised a religion table, and inventing "
            "seven numbers to fit would be worse than showing the seven it printed."),
        how="census, 2002",
        grain="districts, 436,000 people on average",
        gap_share=0.00036625,
        gap="the 8,952 people, 0.037% of the country, enumerated in hotels, whom the "
            "religion table leaves out",
        counts=_ug_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ug" / "ug_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ug_place_weight,
        note="THE QUEUE ROW SAID `regions` AND IT IS 56 DISTRICTS, FROM A TABLE NO UBOS "
             "PAGE LINKS TO. sources.md §11b established that the 2024 census publishes "
             "religion nationally only and priced Uganda as needing DHS or a microdata "
             "email for any geography. Both halves of that stand for 2024; what §11b did "
             "not reach is that the 2002 census published `Table B7: Religion by District "
             "for the Population` as a loose annex PDF in "
             "ubos.org/onlinefiles/uploads/ubos/census_tabulations/, one of fifteen such "
             "tables on the retired tree, reachable now only through the Wayback Machine. "
             "436k people a unit against the four regions the 2002 analytical report stops "
             "at. Rwanda's shape exactly: the report set everyone reads is coarse and a "
             "separate series outside it is not. "
             "WHAT WAS CHECKED FOR 2024 BEFORE FALLING BACK TWENTY-TWO YEARS, because a "
             "later table would beat this one outright: Final Report Volume 1 (434 pages, "
             "religion on fourteen of them, never with a geography); the NPHC 2024 "
             "statistics portal, which is a real query API reaching parish and serves "
             "fifteen tables, none of them religion, with `format=all` returning the same "
             "fifteen; all seventeen sub-region profile reports; the sub-county profiles "
             "workbook; the community module report, whose 200 pages of `Religious` "
             "columns are facility OWNERSHIP; 703 documents swept off ubos.org's own "
             "publications catalogue, none of which mentions religion; and the 2014 Area "
             "Specific Profiles. sources/ug.md has the list. "
             "THE BOUNDARY REBUILD IS THE RISKY PART AND IT IS PROVED ON POPULATION. No "
             "2002 boundary file exists anywhere, so the 56 districts are dissolved out of "
             "COD-AB's 135 of 2020 by matching COD-AB's counties and sub-counties against "
             "the 995 place names Table C1 prints under each 2002 district. Grouping the "
             "2014 census report's Table A3 by that concordance reproduces all 56 of Table "
             "B1's 1991 figures EXACTLY, on 56 distinct values, from a different "
             "publication of a different census. The first attempt failed it: a dropped "
             "district header sent all of Bugiri's sub-counties to Wakiso, and the 1991 "
             "test is what caught it. "
             "KOTIDO IS DRAWN AS PUBLISHED AND THE CALL WAS REVIEWED. Running the same "
             "test on the 2002 column instead leaves 55 districts exact and Kotido 214,787 "
             "people short, which is the entire national difference between the two "
             "publications. Scaling its seven cells by 0.6371 was considered and refused "
             "under §14.4 rule 1: UBOS revised a population, not a religion split, and the "
             "factor would invent seven counts in the one district where `Other` and "
             "`None` are least like the rest of the country. A reviewer agent was asked "
             "and reached the same answer, and found no drawn country where this map has "
             "ever altered a published census figure. What changes instead is that "
             "note_public carries both national shares, with and without Kotido. "
             "KONTUR IS 2023 AND THE COUNTS ARE 2002. It is a within-district weight so "
             "the level is irrelevant, but the shape has moved; sources/ug_grid.py says "
             "so, and its per-district ratios are a third witness on Kotido, which comes "
             "out at 1.07x against a national middle near 1.9x. "
             "RE-LEVELLING ONTO THE 940 SUB-COUNTIES TABLE C1 COUNTS was considered and "
             "refused: unlike the district concordance there is no second publication "
             "giving sub-county population on both vintages, so the join could not be "
             "proved and would move dots on a surface nobody could audit. "
             "THE §3.5 LEAN CHECK RUNS AND SAYS ALMOST NOTHING, which is the right answer "
             "for a 0.037% hole. The excluded hotel population correlates +0.269 with the "
             "Adventist share across the 56 districts and +0.353 with Kalangala dropped, "
             "so it leans urban as hotels would, but if every one of the 8,952 belonged to "
             "a single faith no national share would move by more than 0.04 points.",
    ),
}
