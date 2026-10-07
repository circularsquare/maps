# Anguilla. No census language question: the 2001 census's citizenship, national
# (sources/ai_census.py): Anguillians on the Leeward creole, other citizens on their country's
# language. Every row derived. Placed on religiondots' Kontur hexes (one unit, read-only).
# Record: sources/ai.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ai2001
    df = pd.read_csv(NORM / "ai.csv")
    if int(df["count"].sum()) != 11_430:
        raise SystemExit("ai.csv: expected 11,430 people -- run sources/ai_census.py")
    df["node"] = df["source_category"].map(ai2001.resolve)
    df["unit"] = "AI"   # religiondots' ai_hexes.gpkg: one unit
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Anguilla",
    source=("2001 Population and Housing Census, Table 1.01.1.6, population by district, "
            "citizenship and age group (Anguilla Statistics Department)"),
    how=("no language question: Anguillian citizens drawn as the Leeward creole, other "
         "citizens on their country's language; census 2001"),
    parts=[
        dict(covers="Citizens of other countries",
             source="2001 census, citizenship, drawn on that country's language",
             people=3_130),
        dict(covers="Anguillian citizens",
             source="2001 census, drawn as the Leeward Islands creole", rest=True),
    ],
    grain="the island as one unit, 11,430 people",
    view=[-63.45, 18.15, -62.95, 18.30],
    counts=_counts,
    mappings=["ai2001"],
    place=RD_GEO / "ai" / "ai_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Anguilla's census does not ask which language people speak at home, and the only "
        "published table that says where residents come from is citizenship in 2001. "
        "Anguillian citizens, 73% of residents, and citizens of St Kitts and Nevis are drawn as "
        "speakers of the English-based creole of the Leeward Islands, which linguists treat as "
        "one language with Antiguan Creole. Other citizens are drawn on their country's "
        "language; US and British citizens, many of them Anguillians born abroad, as English "
        "speakers. Anguilla had about 13,600 people in 2011; no later breakdown has been "
        "published."),
)
