# Grenada. No census language question: everyone on Grenadian Creole except white Grenadians on
# English, per parish, from the 2021 census ethnicity table (sources/gd_census.py). Every row
# derived. Placed on religiondots' Kontur hexes for its 7 parishes (read-only). Record:
# sources/gd.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import gd2021
    df = pd.read_csv(NORM / "gd.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 7:
        raise SystemExit(f"gd.csv: {df['geo_id'].nunique()} parishes, expected 7 -- "
                         "run sources/gd_census.py")
    df["node"] = df["source_category"].map(gd2021.resolve)
    df["unit"] = df["geo_id"]   # religiondots' gd_lookup.csv: geo_id == unit
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Grenada",
    source=("2021 Housing and Population Census, preliminary results, Table 22 (ethnicity by "
            "parish) (Central Statistical Office, Grenada)"),
    how=("no language question: everyone drawn as Grenadian Creole except people who gave "
         "their ethnicity as white, drawn as English; census 2021"),
    parts=[
        dict(covers="English", source="2021 census, white Grenadians",
             nodes=["indoeuropean.germanic.english"]),
        dict(covers="Everyone else", source="2021 census, drawn as Grenadian Creole", rest=True),
    ],
    grain="7 parishes, 15,500 people on average",
    gap="742 people in institutions, outside the published tables",
    view=[-61.82, 11.97, -61.36, 12.55],
    counts=_counts,
    mappings=["gd2021"],
    place=RD_GEO / "gd" / "gd_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Grenada's census does not ask about language. Most Grenadians grow up speaking "
        "Grenadian Creole, an English-based creole, and learn standard English at school; the "
        "973 people who gave their ethnicity as white are drawn as English speakers. The "
        "French-based Patois once spoken across the island now has few speakers, mostly "
        "elderly, and is not drawn. The 5% born abroad are drawn as Creole speakers too, "
        "since the census does not say where they were born."),
)
