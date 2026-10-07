# Haiti. No language question in any census: everyone drawn on Haitian Creole, per department, on
# the COD-PS 2024 populations (sources/ht_pop.py). Placed on religiondots' Kontur hexes for the ten
# departments (read-only). Record: sources/ht.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ht2024
    df = pd.read_csv(NORM / "ht.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 10:
        raise SystemExit(f"ht.csv: {df['geo_id'].nunique()} departments, expected 10 -- "
                         "run sources/ht_pop.py")
    df["node"] = df["source_category"].map(ht2024.resolve)
    df["unit"] = df["geo_id"]   # religiondots' ht_lookup.csv: geo_id == unit
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Haiti",
    source=("population: COD-PS 2024 department figures (IHSI projections, HDX); no language "
            "source"),
    how=("no language question: everyone drawn as Haitian Creole, the first language of nearly "
         "all Haitians"),
    parts=[dict(covers="Everyone", source="2024 population projections, all drawn as Haitian "
                "Creole", rest=True)],
    grain="10 departments, 1.19 million people on average",
    view=[-74.6, 17.9, -71.5, 20.2],
    counts=_counts,
    mappings=["ht2024"],
    place=RD_GEO / "ht" / "ht_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Haiti's censuses have never asked about language, so every Haitian is drawn as a "
        "Haitian Creole speaker. The few families who speak French at home, and Spanish "
        "speakers near the Dominican border, have no count. The population is the 2024 "
        "projection, since the last census was in 2003."),
)
