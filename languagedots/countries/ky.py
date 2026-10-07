# Cayman Islands. No census language question: the 2021 census's country of birth by district
# (sources/ky_census.py): the Cayman-born on English, immigrants on their birth country's
# language. Every row derived. Placed on religiondots' Kontur hexes for its 6 districts
# (read-only). Record: sources/ky.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ky2021
    df = pd.read_csv(NORM / "ky.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 6:
        raise SystemExit(f"ky.csv: {df['geo_id'].nunique()} districts, expected 6 -- "
                         "run sources/ky_census.py")
    df["node"] = df["source_category"].map(ky2021.resolve)
    df["unit"] = df["geo_id"]   # religiondots' ky_lookup.csv: geo_id == unit
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Cayman Islands",
    source=("2021 Census of Population and Housing, Tables 4.12C-4.13I, country of birth by "
            "district (Economics and Statistics Office)"),
    how=("no language question: people born in the Cayman Islands drawn as English speakers, "
         "people born abroad on their birth country's language; census 2021"),
    parts=[
        dict(covers="People born abroad",
             source="2021 census, country of birth, drawn on that country's language",
             people=44_068),
        dict(covers="Everyone else", source="born in the Cayman Islands, drawn as English",
             rest=True),
    ],
    grain="6 districts, 11,500 people on average",
    gap="363 people whose birthplace was not stated",
    view=[-81.50, 19.20, -79.65, 19.82],
    counts=_counts,
    mappings=["ky2021"],
    place=RD_GEO / "ky" / "ky_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The Cayman Islands census does not ask about language. Only 35% of residents were born "
        "in the islands; they are drawn as English speakers. Everyone born abroad is drawn on "
        "the language of their birth country, the largest group being the 17,055 born in "
        "Jamaica, drawn as Jamaican Creole. Hondurans are drawn as Spanish speakers, though "
        "some come from the English-speaking Bay Islands."),
)
