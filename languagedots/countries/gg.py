# Guernsey. No language question: the Electronic Census's country of birth at 31 March 2023
# (sources/gg_census.py): the Guernsey-born and British-Irish on English, the foreign-born on their
# birth country's languages through origin_mix; birthplaces not recorded shared out pro rata.
# Every row derived. Placed on Kontur hexes for the island as one unit (sources/gg_geo.py).
# Record: sources/gg.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import gg2023
    df = pd.read_csv(NORM / "gg.csv")
    if df["geo_id"].nunique() != 1 or abs(df["count"].sum() - 64_091) > 1:
        raise SystemExit("gg.csv: expected 64,091 people in one island unit -- run sources/gg_census.py")
    df["node"] = df["source_category"].map(gg2023.resolve)
    df["unit"] = "GG"   # data/geo/gg/gg_hexes.gpkg: one unit, GG
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Guernsey",
    source=("Guernsey Annual Electronic Census 2023, country of birth, population at 31 March 2023 "
            "(States of Guernsey, Data and Analysis)"),
    how=("no language question: people born in Guernsey, the Crown Dependencies, the UK or Ireland "
         "drawn as English speakers, people born elsewhere on their birth country's languages; "
         "register-based census, 2023"),
    parts=[
        dict(covers="People born outside the British Isles",
             source="2023 electronic census, country of birth, drawn on that country's languages; "
                    "people with no recorded birthplace shared out in the same proportions",
             people=8_100),
        dict(covers="Everyone else", source="born in Guernsey, the Crown Dependencies, the UK or "
                                            "Ireland, drawn as English", rest=True),
    ],
    grain="the island as one unit, 64,100 people",
    gap="none: 6,712 people (10.5%) with no recorded birthplace are shared out in proportion to the rest",
    view=[-2.70, 49.40, -2.40, 49.53],
    counts=_counts,
    mappings=["gg2023"],
    place=GEO / "gg" / "gg_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Guernsey has not held a traditional census since 2001, and neither it nor the register "
        "that replaced it asks about language. The register records where each person was born. "
        "People born in the Channel Islands, the UK or Ireland, about 86% of those with a known "
        "birthplace, are drawn as English speakers. People born elsewhere are drawn on the "
        "languages of their birth country, such as Portuguese for the 1,326 born in Portugal "
        "(mostly Madeirans) and Latvian and Russian for the 887 born in Latvia. About one in ten "
        "residents have no recorded birthplace, and they are drawn in the same proportions as "
        "everyone else. Guernsey-born children of immigrants are drawn as English speakers, so "
        "the minority languages are likely understated. The data covers Guernsey, Herm and Jethou; "
        "Alderney and Sark are not included."),
)
