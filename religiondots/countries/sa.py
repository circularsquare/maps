# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _sa_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Saudi Arabia is 1.9 million km2 and five cities hold half its people; Kontur decides where
    people are inside each region, uncalibrated, since nothing finer than the region is published
    with Saudis and non-Saudis (sources/sa_grid.py).
    """
    return _kontur_place_weight(place, "sa_hexes.gpkg", "sources/sa_grid.py")


def _sa_counts():
    """Census 2022 citizens on Islam, non-Saudis by nationality and sex: 13 regions.

    EVERY ROW IS `modelled` (§7b). No source asks religion in Saudi Arabia, so every citizen is
    drawn on Islam; each region's non-Saudi men and women take the census's national nationality mix
    for their sex through Pew 2020, with Burma's nationals (the Rohingya) on Islam and India's Hindu
    share set by Pew's Saudi estimate. Both halves are the same census's counts per region, so they
    partition each unit. sources/sa.py and sources/sa.md.
    """
    from sa2022 import resolve

    nat = pd.read_csv(HERE / "data" / "normalized" / "sa.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    nat["node"] = nat["source_category"].map(resolve)
    unmapped = sorted(nat.loc[nat["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"sa.csv categories with no node: {unmapped}")
    ext = pd.read_csv(HERE / "data" / "normalized" / "sa_foreign.csv", dtype={"geo_id": str})
    df = pd.concat([nat[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    lut = pd.read_csv(HERE / "data" / "geo" / "sa" / "sa_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"sa rows with no unit: {missing}; re-run sources/sa_geo.py")
    if df["unit"].nunique() != 13:
        raise SystemExit(f"{df['unit'].nunique()} regions, expected 13")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "sa": dict(
        name="Saudi Arabia",
        source="No census or survey asks; GASTAT's 2022 census count of Saudis and non-Saudis in each "
               "region and of non-Saudis by nationality and sex (as mirrored by the Gulf Labour "
               "Markets, Migration and Population programme), through Pew Research Center's 2020 "
               "estimates",
        basis="citizens drawn as Muslim, which nobody asked; foreign residents by nationality and sex",
        note_public=(
            "**Nobody in Saudi Arabia is asked their religion.** The 2022 census has no question on "
            "it, and the Arab Barometer's Saudi interviews leave it blank. Saudi citizens are Muslim "
            "by law, so the **18,792,262** citizens the census counted are all drawn as Muslim. The "
            "US State Department puts citizens at 85 to 90% Sunni and 10 to 12% Shia, most of the "
            "Shia in the Eastern Province, with Ismailis in Najran; the map does not split them, "
            "because Shia mosques in Qatif, Dammam and Najran were bombed in 2015. Nobody counted "
            "these dots, and they are drawn desaturated to say so. "
            "**Every non-Muslim drawn is a foreign resident.** The census counted **13,382,962** "
            "non-Saudis, 41.6% of the people living in the country, in every region, but their "
            "nationality only for the whole country, by sex. Men and women come from different "
            "places: there are 1,181 Bangladeshi men for every 100 Bangladeshi women, and 61 "
            "Filipino men for every 100 Filipino women. So each region's foreign men are drawn at "
            "the national mix of foreign men and its foreign women at the mix of foreign women, "
            "using the census's count of foreign men per 100 foreign women in each region, from "
            "264 in Makkah to 510 in Asir. Each nationality is drawn at Pew Research Center's 2020 "
            "estimate for its home country, which cannot see anyone who converted or stopped "
            "practising. "
            "**Two nationalities are not drawn at their home country's figure.** The 163,717 people "
            "from Myanmar are Rohingya, who are Muslim. Indians in the Gulf are mostly Muslim, which "
            "Pew says of its own migration estimates, so Indians are drawn at the Hindu share that "
            "gives Pew's figure for Hindus in Saudi Arabia (2.6% of everyone): 19.7% of Indians, "
            "not India's 79%. "
            "That puts **2,461,948** people on religions other than Islam: 1,337,918 Christians, "
            "843,672 Hindus, 120,949 Buddhists and 60,679 with no religion. Pew's estimate for "
            "everyone living in Saudi Arabia is 92.7% Muslim and 4.4% Christian; this map draws "
            "92.3% and 4.2%. Its Buddhists are about five times Pew's figure."),
        how="no source asks; citizens drawn as Muslim, foreign residents by nationality and sex",
        grain="regions, 2,475,000 people on average",
        gap="Saudi citizens who are not Muslim, whom no source counts; and foreign residents the 2022 "
            "census missed, whom nobody has counted",
        counts=_sa_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sa" / "sa_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sa_place_weight,
        note="BUILT ON ANITA'S MAGHREB AND MAURITANIA RULINGS (ask/RULINGS.md 2026-09-15 and 2026-09-16: "
             "a near-uniformly Muslim country on a compiler's figure, foreigners by region). "
             "sources/sa.md is the record. CITIZENS: nothing asks (census 2022 no item; AB II Q1012 "
             "empty, AB V no rows; ministry counts mosques); all 18,792,262 on islam; no Sunni/Shia "
             "split (spec §14, ask 040). NON-SAUDIS: 13,382,962 per region (GLMM's copy of the census "
             "table, checked against the census report's Figure 11), split by sex with Figure 12's "
             "ratios raked to the national sexes (Riyadh within 0.013% of RCRC's count); national "
             "nationality by sex from GLMM's four tables (continent shares reproduce the report's "
             "prose), each sex's mix applied in every region; Pew 2020 per nationality, Muslim "
             "branches folded to islam; Burma on islam (Rohingya); India's Hindu share 19.69% so the "
             "layer's Hindus equal Pew's Saudi 2.622%; remainder (42,140, the Americas) on Pew's "
             "North America and Latin America rows. WITNESS: Christians 0.95 of Pew's Saudi share; "
             "Buddhists 5.1x and unaffiliated 1.7x, not corrected. GEOGRAPHY: COD-AB ADM1 13 regions. "
             "PLACEMENT: Kontur SA uncalibrated (1.148x the census; 0.89-1.29 per region); no block at "
             "the cap.",
    ),
}
