# Guam. 2020 Island Areas Census, DHC table PCT25 by tract (sources/gu_census.py); the six tracts
# whose cells the Bureau suppresses are one unit (Guam minus the published tracts), placed by each
# tract's total population. Kontur hexes cut to the 2020 tracts (sources/gu_geo.py).
# sources/gu.md is the record.
from _shared import *  # noqa: F401,F403


def _counts():
    import gu2020
    df = pd.read_csv(NORM / "gu.csv", dtype={"geo_id": str})
    df["node"] = df["source_category"].map(gu2020.resolve)
    df["unit"] = df["geo_id"]
    df["tier"] = "measured"
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Guam",
    source="2020 Island Areas Census of Guam, Demographic and Housing Characteristics, table PCT25 "
           "(U.S. Census Bureau)",
    how="census, 2020, language spoken at home (the language other than English, where there is one)",
    parts=[dict(covers="People aged 5 and over in households",
                source="2020 census, language spoken at home", rest=True)],
    grain="45 tracts, 2,800 people aged 5 and over on average, plus six withheld tracts as one",
    gap="children under 5, group quarters and military housing, 18,066 (11.7%)",
    view=[144.6, 13.22, 145.0, 13.68],
    counts=_counts,
    mappings=["gu2020"],
    place=GEO / "gu" / "gu_tracts.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked each person aged 5 or over whether they speak a language other than "
        "English at home, and if so which one. Someone who speaks Chamorro and English at home is "
        "counted under Chamorro, so English here means people who speak only English at home. "
        "The census groups the languages of the Philippines together, so a quarter of the "
        "island's people are drawn as speaking a Philippine language without saying which one. "
        "The Census Bureau withheld the figures for six tracts, most of them in Tamuning; "
        "their people are spread over those tracts by population."),
)
