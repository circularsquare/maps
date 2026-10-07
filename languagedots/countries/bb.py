# Barbados. No census language question: the 2010 census's birthplace and ethnic-origin tables
# per parish (sources/bb_census.py) give Bajan, English and the immigrant languages. Every row
# derived. Placed on religiondots' Kontur hexes for the 11 parishes (read-only). Record:
# sources/bb.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import bb2010
    df = pd.read_csv(NORM / "bb.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 11:
        raise SystemExit(f"bb.csv: {df['geo_id'].nunique()} parishes, expected 11 -- "
                         "run sources/bb_census.py")
    df["node"] = df["source_category"].map(bb2010.resolve)
    df["unit"] = df["geo_id"]   # religiondots' bb_lookup.csv: geo_id == unit
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Barbados",
    source=("Population and Housing Census 2010, Tables 01.01, 02.04 (ethnic origin), 04.01 "
            "(country of birth) and 04.02 (Barbadian-born by parish) (Barbados Statistical "
            "Service)"),
    how=("no language question: people born in Barbados drawn as Bajan, except white "
         "Barbadians as English; people born abroad on their birth country's language; census "
         "2010"),
    parts=[
        dict(covers="People born abroad",
             source="2010 census, country of birth (national mix in every parish), drawn on "
                    "that country's language",
             people=32_825),
        dict(covers="White Barbadians", source="2010 census, ethnic origin, drawn as English",
             people=5_205),
        dict(covers="Everyone else born in Barbados", source="2010 census, drawn as Bajan",
             rest=True),
    ],
    grain="11 parishes, 20,600 people on average",
    gap=("the 18% of residents the 2010 census did not tabulate; and the 12,164 foreign-born "
         "with no country recorded, drawn in the mix of those with one"),
    view=[-59.72, 13.02, -59.37, 13.36],
    counts=_counts,
    mappings=["bb2010"],
    place=RD_GEO / "bb" / "bb_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Barbados's census does not ask about language. Most Barbadians grow up speaking "
        "Bajan, the island's English-based creole, and learn standard English at school; "
        "white Barbadians, 2.7% of the 2010 count, are drawn as English speakers. People born "
        "abroad, 15% of the count, are drawn on the language of their birth country, for the "
        "Caribbean its own creole. Birthplace is published by parish only as Barbadian or "
        "foreign, so every "
        "parish has the same mix of foreign birthplaces. The 2010 census counted 226,193 people "
        "against an estimated 277,821 residents; the dots follow the count."),
)
