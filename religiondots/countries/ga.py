# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ga_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Nine provinces over 265,000 km2, mostly forest; Estuaire holds 62% of the 2026 count. Kontur's
    hexes are scaled to each province's Gabonese count (sources/ga_grid.py); Kontur's Haut-Ogooué
    reads 3.2x its census share and Estuaire 0.63x, which the scaling removes.
    """
    return _kontur_place_weight(place, "ga_hexes.gpkg", "sources/ga_grid.py")


def _ga_counts():
    """Afrobarometer rounds 6-9 on the Gabonese citizens of the 2026 census: 9 provinces, 8 nodes.

    EVERY ROW IS `modelled` (§7b). No census has published religion. Nothing passes the
    split-half at the nine provinces, so every province takes one national mix, except Woleu-Ntem's
    Christian share and Nyanga's None share (the standout rule); the Christians are divided at the
    DHS 2019-21 national ratio. Foreign residents (34.1%) are not drawn. sources/ga.py, sources/ga.md.
    """
    from ga2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ga.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "ga" / "ga_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ga.csv provinces with no polygon: {missing}; re-run sources/ga_geo.py")
    if df["unit"].nunique() != 9:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 9")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"ga.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ga": dict(
        name="Gabon",
        source="Afrobarometer rounds 6 to 9 (2015 to 2021), with Christians divided at the national "
               "ratio of the Demographic and Health Survey 2019-21 (EDSG-III, Tableau 3.1), against "
               "each province's Gabonese citizens in the 2026 census (RGPL 2026, certified by the "
               "Constitutional Court; citizens per province estimated from the 2013 census)",
        basis="self-identification, citizens 18 and over",
        note_public=(
            "**Gabon's censuses do not publish religion.** The 2013 census results have a section "
            "headed for religion that prints language only, and the 2026 census has so far "
            "published its province totals and little else. This map uses the Afrobarometer "
            "survey instead, which asked **4,790** Gabonese adults their religion in four rounds "
            "between 2015 and 2021. Nobody counted these dots, so they disappear when inferred "
            "dots are turned off. Children are drawn at the shares of adults. "
            "**Only Gabonese citizens are drawn.** The survey interviews citizens, so the map "
            "shows the **2,318,365** Gabonese the 2026 census counted and leaves out its "
            "**1,200,256** foreign residents, 34.1% of the people. The census gave that split for "
            "the whole country only. Each province's citizens are estimated from the 2013 census, "
            "which counted foreigners by province, by assuming every province's share of "
            "foreigners rose in the same proportion since. "
            "**The survey is too small to say much about where religions differ.** It reached "
            "between 135 people (Nyanga) and 2,456 (Estuaire) per province over the four rounds, "
            "and no religion's ranking of the provinces held between the earlier and later "
            "rounds. Two things did: Woleu-Ntem was the most Christian province every time, "
            "**92.5%**, and Nyanga had the most people of no religion, **27.6%**. Those two "
            "shares are drawn; otherwise every province is drawn at the same mix. "
            "**The survey cannot divide the Christians.** Between 21% and 45% of respondents, "
            "depending on the round, answered only \"Christian\", and the named churches rose and "
            "fell with it. Christians are divided here at one national ratio from the 2019-21 "
            "Demographic and Health Survey: 38.1% Catholic, 11.8% Protestant, 45.1% revival "
            "churches (*églises de réveil*) and 5.0% other Christian, the same in every "
            "province. That survey asked all residents, foreigners included, so the ratio is not "
            "the citizens' alone; its Catholic share of everyone, 29.9%, is close to the "
            "Afrobarometer's 29.1%. "
            "**Traditional religion is a floor.** The survey counts only people who give it as "
            "their one religion, 1.3%. Most of Gabon's Muslims are foreign residents and so are "
            "not on this map: the health survey found 8.2% of women and 15.2% of men Muslim, "
            "against 1.7% of citizens in the Afrobarometer. "
            "**The 2026 count is far above earlier estimates.** The Constitutional Court "
            "certified 3,518,621 people on 27 August 2026, nearly twice the 1,811,079 of the 2013 "
            "census, and Gabonreview put it about a million above the UN's and the World Bank's "
            "estimates. How the difference arose has not been published."),
        how="survey, 2015 to 2021, one national mix with two provincial shares",
        grain="provinces, 258,000 citizens on average",
        gap="foreign residents, 1,200,256 in the 2026 census and 34.1% of residents; the survey "
            "interviews citizens only",
        gap_share=0.34111,                      # 1,200,256 / 3,518,621, the Minister's split
        counts=_ga_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ga" / "ga_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ga_place_weight,
        note="REOPENED 2026-10-03 on Anita's priority-holes ruling (ask/RULINGS.md 2026-09-15) and "
             "the Libya ruling (non-nationals into gap). sources/ga.md is the record. CENSUS: no "
             "religion table in RGPL 2013 or 2026 (REOPEN if a 2026 thematic volume prints one). "
             "SURVEY: Afrobarometer R6-R9, 4,790 citizens 18+, 9 provinces every round; split-half "
             "passes nothing; standouts Christian Woleu-Ntem, None Nyanga; level vs R8-R9 within "
             "2.2 points. CHRISTIAN SPLIT: DHS 2019-21 FR371 Tableau 3.1, women and men at 48.8% "
             "men, all residents; Catholic level witness 29.85 vs 29.10. POPULATION: RGPL 2026 "
             "provinces (press, sum = certified 3,518,621), Gabonese 2,318,365 split by IPF on RGPL "
             "2013 Tableau 24/5 foreign shares. GEOGRAPHY: COD-AB v01 9 provinces; area witness "
             "0.89-1.06, Ogooue-Lolo 1.15 pinned. PLACEMENT: Kontur GA scaled per province to "
             "Gabonese; Ntoum thin in Kontur. DHS microdata (would place churches): asks 047/048.",
    ),
}
