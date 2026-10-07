# Macau. 2021 Population Census, usual language, by the 23 statistical districts
# (sources/mo_census.py); placed on the census's own population by residential building
# (sources/mo_geo.py).
from _shared import *  # noqa: F401,F403


def _counts():
    import mo2021
    df = pd.read_csv(NORM / "mo.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "zona"]
    df["node"] = df["source_category"].map(mo2021.resolve)
    df = df.rename(columns={"geo_id": "unit"})
    return by_unit(df)


ENTRY = dict(
    name="Macau",
    source="2021 Population Census: Population Statistics Database, usual language by "
           "statistical district, and the Statistical Geographic Information System's population "
           "by building (Statistics and Census Service, DSEC)",
    how="census, 2021, usual language",
    parts=[dict(covers="Everyone aged 3 and over",
                source="2021 census, usual language, aged 3 and over", rest=True)],
    grain="23 statistical districts, 29,600 people on average, placed on residential buildings",
    gap="children under 3, 18,288 (2.7%), whom the question does not cover, and 777 people "
        "living on boats in the maritime area, which has no land to draw them on",
    view=[113.52, 22.10, 113.61, 22.22],
    counts=_counts,
    mappings=["mo2021"],
    place=GEO / "mo" / "mo_buildings.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asks everyone aged 3 and over which language they usually speak, so this "
        "map shows the language of daily use, not mother tongue. The Statistics and Census "
        "Service publishes seven groups for each of Macau's 23 statistical districts; \"Other "
        "Chinese dialects\" (36,032 people) and \"Others\" (11,626) are not broken down. The "
        "count includes the migrant workers who live in Macau: of 33,896 Filipino nationals, "
        "18,718 named Tagalog and 14,209 English, and of 12,217 Vietnamese nationals, 5,199 "
        "named Cantonese."),
)
