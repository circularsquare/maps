# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _tj_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Kontur TJ on the five regions, Dushanbe on OpenStreetMap's current city line
    (sources/tj_geo.py, sources/tj_grid.py).
    """
    return _kontur_place_weight(place, "tj_hexes.gpkg", "sources/tj_grid.py")


def _tj_counts():
    """The 2020 census's permanent population in five regions, as an ethnicity model.

    EVERY ROW IS `modelled` (§7b). The non-Muslim-heritage nationalities are placed on the 2010
    census's nationality by region, scaled to the Agency's 2020 share of Russians, and given a
    religion (Russians, Tatars, Ukrainians, Belarusians and Germans at Kazakhstan's 2021 census
    shares); everyone else is on Islam. sources/tj.py and sources/tj.md.
    """
    from tj2020 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "tj.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[""])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"tj.csv categories with no node: {unmapped}")
    if df["geo_id"].nunique() != 5:
        raise SystemExit(f"{df['geo_id'].nunique()} units in tj.csv, expected 5")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0].copy()
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "tj": dict(
        name="Tajikistan",
        source="The 2020 census asked religion and has not published it; nationality by region "
               "from the 2010 census and the 2020 census's regional populations (Agency on "
               "Statistics), with Kazakhstan's 2021 census shares for Russians and other "
               "minorities",
        basis="modelled from census nationality; the census's religion answers are unpublished",
        note_public=(
            "**Tajikistan's 2020 census asked everyone their religion and has published none of "
            "the answers.** Question 7 of the census form offered Islam, Christianity, "
            "non-believer, refusal and other. The Agency on Statistics has released the census "
            "volume by volume since 2022, and the volume on nationality and language is still an "
            "empty heading. The surveys that ask religion in Tajikistan do not see its minorities: "
            "in the EBRD's Life in Transition Survey in 2016, 99.5% answered Muslim and not one of "
            "**1,510** respondents answered Orthodox. So the map is built from the census's "
            "nationalities instead. Nobody counted these dots, so they disappear when inferred "
            "dots are turned off. "
            "**Every nationality of Muslim heritage is drawn as Muslim.** That covers Tajiks, "
            "Uzbeks and the Uzbek groups the census lists by tribe, such as the Lakai and Kongrat, "
            "and Kyrgyz, Turkmens, Arabs and Afghans. The 2010 census gives Russians and Tatars "
            "by region, and they are placed there, scaled down to the 0.3% Russian the 2020 census "
            "found, or about **29,000** people. Smaller groups such as Ukrainians, Germans and "
            "Armenians were counted only for the whole country, and are placed where the Russians "
            "are, more than half of them in Dushanbe. No source gives a religion for Russians, "
            "Tatars, Ukrainians, Belarusians or Germans in Tajikistan, so they take the shares "
            "Kazakhstan's 2021 census gives the same nationalities, leaving out those who refused "
            "to answer: **92.7%** of Russians Orthodox, 2.1% Muslim and 5.1% non-believers. The "
            "non-believers are the weakest part of this map, because Kazakhstan's own figures "
            "show non-belief changing from place to place inside every nationality. Koreans, "
            "Chinese, Ossetians and about forty smaller nationalities, 2,464 people, are drawn as "
            "religion not known. That makes **30,078** Christians, 1.9% of Dushanbe not drawn as "
            "Muslim, and 99.6% of the people drawn Muslim. "
            "**The published estimate is higher.** Pew Research Center, from its own survey of "
            "about 1,500 adults in 2011 and 2012, puts Tajikistan at 98.9% Muslim and 1.0% "
            "Christian, which is 97,515 Christians. One percent of a sample that size is about "
            "fifteen people, and no source says where the difference lives, so it is not drawn. "
            "Christians and non-believers among the Muslim-heritage nationalities are drawn as "
            "Muslim for the same reason. "
            "**Ismailis are not drawn as a branch.** Most of Tajikistan's Muslims are Sunni, and "
            "the Pamiri peoples of Gorno-Badakhshan are mostly Nizari Ismailis, but the census "
            "counts Pamiris as Tajiks and no survey has measured the region's faith in a way that "
            "holds up. "
            "**The dots stand where the 2020 census counted people.** Its population is the "
            "permanent one, which includes about 353,000 people who were outside the country "
            "when it was taken; they are drawn at home."),
        how="the census asked but has not published; modelled from census nationality",
        grain="5 regions, 1.9 million people on average",
        gap="Christians and non-believers of Muslim-heritage nationality, whom no source places "
            "and who are drawn as Muslim",
        counts=_tj_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tj" / "tj_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_tj_place_weight,
        note="REOPENED ON THE COMPILER AND PRIORITY RULINGS (ask/RULINGS.md 2026-09-15, "
             "2026-09-16) FROM sources.md §scout-2026-10-03-negatives. sources/tj.md is the "
             "record. The 2020 census asked religion (Form 2 Q7) and published nothing; LiTS III "
             "has zero Orthodox; CAB does not ask in Tajikistan. MODEL (sources/tj.py): 2020 "
             "census Vol I table 1 permanent population (9,657,005) in 5 regions; non-Muslim-"
             "heritage nationalities from the 2010 census Vol III (Russians and Tatars by region, "
             "the rest national and placed on the Russians), scaled by 28,971/34,838 (the Agency's "
             "2020 Russian share, 0.3%); Russians, Tatars, Ukrainians, Belarusians, Germans at "
             "Kazakhstan 2021's religion-by-nationality (kz_model.py), refusals and the small "
             "answers dropped; Armenians, Georgians, Jews religio-ethnic; other non-Muslim-heritage "
             "unknown; rest islam. NOT DRAWN: Pew's 1.0% Christian (Survey of the World's Muslims "
             "2011-12, about 15 respondents); no sect (ask 051 on Gorno-Badakhshan's Ismailis, "
             "§14). GEO: geoBoundaries ADM1 with Dushanbe redrawn on OSM relation 7328360.",
    ),
}
