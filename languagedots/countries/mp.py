# Northern Mariana Islands. 2020 Island Areas Census, DHC table PCT24 by tract (sources/mp_census.py).
# Kontur hexes cut to the 2020 tracts (sources/mp_geo.py). sources/mp.md is the record.
from _shared import *  # noqa: F401,F403


def _counts():
    import mp2020
    df = pd.read_csv(NORM / "mp.csv", dtype={"geo_id": str})
    df["node"] = df["source_category"].map(mp2020.resolve)
    df["unit"] = df["geo_id"]
    df["tier"] = "measured"
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Northern Mariana Islands",
    source="2020 Island Areas Census of the Commonwealth of the Northern Mariana Islands, "
           "Demographic and Housing Characteristics, table PCT24 (U.S. Census Bureau)",
    how="census, 2020, language spoken at home (the language other than English, where there is one)",
    parts=[dict(covers="People aged 5 and over",
                source="2020 census, language spoken at home", rest=True)],
    grain="21 tracts with people, 2,100 people aged 5 and over on average",
    gap="children under 5, 3,218 (6.8%)",
    view=[144.6, 14.0, 145.9, 15.35],
    counts=_counts,
    mappings=["mp2020"],
    place=GEO / "mp" / "mp_tracts.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked each person aged 5 or over whether they speak a language other than "
        "English at home, and if so which one. Someone who speaks Chamorro and English at home is "
        "counted under Chamorro, so English here means people who speak only English at home. "
        "The census groups the languages of the Philippines together, so 29% of the islands' "
        "people are drawn as speaking a Philippine language without saying which one. "
        "Chamorro and Carolinian are the islands' own languages: Carolinian is spoken by 4% of "
        "people, nearly all of them on Saipan."),
)
