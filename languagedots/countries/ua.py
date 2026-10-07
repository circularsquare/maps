# Ukraine. 2001 census native language by raion and city (sources/ua_c01.py), on Kontur hexes
# keyed to the census's 2001 units (sources/ua_geo.py). Crimea and Sevastopol are drawn from
# Russia's 2021 census instead (Anita's ruling, 2026-10-05; countries/ru.py, sources/ua.md).
from _shared import *  # noqa: F401,F403


def _crimean(unit):
    """The Autonomous Republic of Crimea's 25 raions and cities (UKR_01_*) and Sevastopol."""
    return unit.startswith("UKR_01_") or unit == "UKR_02_01"


def _counts():
    import ua2001
    df = pd.read_csv(NORM / "ua.csv")
    df = df[~df["geo_id"].map(_crimean)]   # drawn from Russia's 2021 census
    df["node"] = df["source_category"].map(ua2001.resolve)
    df = df[df["node"].notna()]            # "Did Not Indicate" is the gap, not drawn
    return df.rename(columns={"geo_id": "unit"}).groupby(
        ["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Ukraine",
    source=("All-Ukrainian Population Census 2001, native language by administrative unit (State "
            "Statistics Committee of Ukraine), with the U.S. Census Bureau's tabulation of its "
            "population, language and nationality tables"),
    how="census, 2001, native language",
    parts=[dict(covers="Everyone outside Crimea and Sevastopol",
                source="2001 census, native language", rest=True)],
    grain="646 raions, cities and Kyiv districts, 71,000 people on average",
    gap=("people who did not state a native language, 0.4% (188,588); Crimea and Sevastopol "
         "(2.4 million people in 2001) are drawn from Russia's 2021 census"),
    view=[22.1, 44.3, 40.3, 52.4],
    counts=_counts,
    mappings=["ua2001"],
    place=GEO / "ua" / "ua_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Ukraine has not held a census since 2001, so this map shows the country as the census "
        "found it then, 48.2 million people, before the emigration of the following decades, the "
        "occupation of Crimea and parts of the east from 2014 and the full-scale war from 2022. "
        "The census asked for native language, which leans towards identity rather than everyday "
        "use: 14.8% of ethnic Ukrainians named Russian. Tatar, Azerbaijani, Georgian and a few "
        "other languages are drawn from a second census table, the people of those "
        "nationalities who named their own language. Crimea and Sevastopol are drawn from "
        "Russia's 2021 census, which has counted them since 2014."),
)
