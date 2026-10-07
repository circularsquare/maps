# Dominica. No census language question: the 2011 census's settlement counts (sources/dm_census.py)
# put Wesley and Marigot/Concord on Kokoy, the Haitian-born on Haitian Creole and everyone else on
# Kweyol. Every row derived. Placed on religiondots' Kontur hexes, re-keyed into the two villages
# and the rest (data/geo/dm/dm_hexes.gpkg). Record: sources/dm.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import dm2011
    df = pd.read_csv(NORM / "dm.csv")
    if sorted(df["geo_id"].unique()) != ["DM-MARIGOT", "DM-REST", "DM-WESLEY"]:
        raise SystemExit("dm.csv: expected units DM-WESLEY, DM-MARIGOT, DM-REST -- "
                         "run sources/dm_census.py")
    df["node"] = df["source_category"].map(dm2011.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Dominica",
    source=("2011 Population and Housing Census, preliminary results (Central Statistical "
            "Office, Dominica): population, settlement counts and the Haitian-born"),
    how=("no language question: Wesley and Marigot drawn as Kokoy, the Haitian-born as Haitian "
         "Creole, everyone else as Kweyol; census 2011"),
    parts=[
        dict(covers="Wesley and Marigot", source="2011 census, village populations, drawn as "
             "Kokoy", nodes=["creole.english_based.antiguan"]),
        dict(covers="People born in Haiti", source="2011 census, country of birth, drawn as "
             "Haitian Creole", nodes=["creole.french_based.haitian"]),
        dict(covers="Everyone else", source="2011 census, drawn as Kweyol", rest=True),
    ],
    grain="two villages and the rest of the country, 71,300 people",
    view=[-61.55, 15.14, -61.20, 15.70],
    counts=_counts,
    mappings=["dm2011"],
    place=GEO / "dm" / "dm_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Dominica's census does not ask about language. English is the official language, but "
        "most Dominicans grow up speaking Kweyol, the French-based creole also spoken in "
        "Martinique, Guadeloupe and St Lucia, and everyone is drawn as a Kweyol speaker except "
        "two groups. Wesley and Marigot, on the northeast coast, speak Kokoy, an English-based "
        "creole, and the 1,054 people born in Haiti are drawn as Haitian Creole speakers. "
        "Younger Dominicans, especially in Roseau, increasingly grow up with English first; no "
        "survey counts them, so they are drawn with Kweyol. The Kalinago speak Kweyol and "
        "English; their own language died out in the twentieth century."),
)
