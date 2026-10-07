# Turks and Caicos Islands. No census language question: the 2012 census's country of citizenship,
# national, applied to each island's population (sources/tc_census.py). Every row derived. Placed
# on Kontur hexes keyed to the seven inhabited islands (data/geo/tc/tc_hexes.gpkg). Record:
# sources/tc.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import tc2012
    df = pd.read_csv(NORM / "tc.csv")
    if df["geo_id"].nunique() != 7:
        raise SystemExit("tc.csv: expected 7 islands -- run sources/tc_census.py")
    df["node"] = df["source_category"].map(tc2012.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Turks and Caicos Islands",
    source=("2012 Population and Housing Census (Department of Statistics, Turks and Caicos "
            "Islands): population by country of citizenship, and by island"),
    how=("no language question: citizens drawn as Turks and Caicos Creole, foreign nationals on "
         "their country's language; census 2012, citizenship for the islands as a whole"),
    parts=[
        dict(covers="Citizens", source="2012 census, citizenship, drawn as Turks and Caicos "
                                       "Creole",
             nodes=["creole.english_based.turks_caicos"]),
        dict(covers="Foreign nationals",
             source="2012 census, country of citizenship, drawn on that country's language",
             rest=True),
    ],
    grain="the country as one unit, 31,500 people, placed by island",
    view=[-72.55, 21.15, -71.05, 22.0],
    counts=_counts,
    mappings=["tc2012"],
    place=GEO / "tc" / "tc_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census of the Turks and Caicos Islands does not ask about language. Citizens, 39% "
        "of the 2012 count, are drawn as speakers of Turks and Caicos Creole, the English-based "
        "creole close to Bahamian. Haitian nationals, 35% of residents, are drawn as Haitian "
        "Creole speakers. Residents of Haitian descent who hold Turks and Caicos citizenship "
        "are drawn with the creole, so Haitian Creole is undercounted to that extent. 14% of "
        "residents are from countries "
        "the published table does not name, Jamaicans among them, and are drawn as language not "
        "known. Citizenship is published for the islands as a whole, so every island is drawn "
        "with the same mix."),
)
