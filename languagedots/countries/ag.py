# Antigua and Barbuda. No census language question: the 2011 census's country of birth, national
# (sources/ag_census.py): the native-born on Antiguan and Barbudan Creole, immigrants on their
# birth country's language. Every row derived. Placed on religiondots' Kontur hexes for the
# country as one unit (read-only). Record: sources/ag.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ag2011
    df = pd.read_csv(NORM / "ag.csv")
    if df["geo_id"].nunique() != 1 or len(df) != 20:
        raise SystemExit("ag.csv: expected 20 birthplaces for one national unit -- "
                         "run sources/ag_census.py")
    df["node"] = df["source_category"].map(ag2011.resolve)
    df["unit"] = "AG"   # religiondots' ag_hexes.gpkg: one unit, AG
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Antigua and Barbuda",
    source=("2011 Population and Housing Census, Q58 country of birth (Statistics Division, "
            "Antigua and Barbuda; Redatam output)"),
    how=("no language question: people born in Antigua and Barbuda drawn as Antiguan Creole, "
         "people born abroad on their birth country's language; census 2011"),
    parts=[
        dict(covers="People born abroad",
             source="2011 census, country of birth, drawn on that country's language",
             people=25_406),
        dict(covers="People born in Antigua and Barbuda",
             source="2011 census, drawn as Antiguan Creole", rest=True),
    ],
    grain="the country as one unit, 84,800 people",
    gap="1,341 people whose birthplace was not stated",
    view=[-62.00, 16.90, -61.62, 17.78],
    counts=_counts,
    mappings=["ag2011"],
    place=RD_GEO / "ag" / "ag_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census of Antigua and Barbuda does not ask about language. People born in the "
        "country, 68% of the 2011 count, are drawn as speakers of Antiguan Creole, the "
        "English-based creole most Antiguans grow up with. People born abroad are drawn on the "
        "language of their birth country, such as Guyanese and Jamaican Creole, Kweyol for "
        "those from Dominica and Spanish for those from the Dominican Republic. White "
        "Antiguans, who speak English, are not counted separately and are drawn as Creole "
        "speakers. Birthplace is published for the country as a whole, so each language's "
        "dots follow population alone."),
)
