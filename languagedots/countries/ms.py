# Montserrat. No census language question: the 2011 census's place of birth, national
# (sources/ms_census.py): the Montserrat-born on the Leeward creole, the foreign-born on their
# birth country's language. Every row derived. Placed on religiondots' Kontur hexes (one unit,
# read-only). Record: sources/ms.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ms2011
    df = pd.read_csv(NORM / "ms.csv")
    if int(df["count"].sum()) != 4_768:
        raise SystemExit("ms.csv: expected 4,768 people -- run sources/ms_census.py")
    df["node"] = df["source_category"].map(ms2011.resolve)
    df["unit"] = "MS"   # religiondots' ms_hexes.gpkg: one unit
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Montserrat",
    source=("2011 Population and Housing Census, question 47 place of birth (Statistics "
            "Department Montserrat, on ECLAC's REDATAM server)"),
    how="no language question; census, 2011, place of birth drawn as a language",
    parts=[
        dict(covers="Born on Montserrat", source="2011 census, drawn as the Leeward creole",
             people=2_910),
        dict(covers="Born abroad",
             source="2011 census, country of birth, drawn on that country's language",
             rest=True),
    ],
    grain="the island as one unit, 4,800 people",
    gap="7 people whose birthplace was not known or not stated",
    view=[-62.26, 16.66, -62.12, 16.83],
    counts=_counts,
    mappings=["ms2011"],
    place=RD_GEO / "ms" / "ms_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Montserrat's census does not ask about language. The 61% of residents in 2011 who "
        "were born on the island are drawn as speakers of the English-based creole of the "
        "Leeward Islands, which linguists treat as one language with Antiguan Creole. People "
        "born abroad are drawn on the language of where they were born. People born in "
        "Britain, the United States and Canada, many of them children of Montserratians, are "
        "drawn as English speakers."),
)
