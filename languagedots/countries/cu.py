# Cuba. No language question in any census: everyone drawn on Spanish, per municipality, on ONEI's
# 2023 municipal populations. Placed on religiondots' Kontur hexes re-keyed from provinces to the
# 168 municipalities (sources/cu_geo.py, languagedots' own copy). Record: sources/cu.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import cu2023
    df = pd.read_csv(NORM / "cu.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 168:
        raise SystemExit(f"cu.csv: {df['geo_id'].nunique()} municipalities, expected 168 -- "
                         "run sources/cu_geo.py")
    df["node"] = df["source_category"].map(cu2023.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Cuba",
    source=("population: Oficina Nacional de Estadística e Información, municipal population "
            "2023; no language source"),
    how="no language question: everyone drawn as Spanish",
    parts=[dict(covers="Everyone", source="2023 municipal population, drawn as Spanish",
                rest=True)],
    grain="168 municipalities, 60,000 people on average",
    view=[-85.1, 19.7, -74.0, 23.4],
    counts=_counts,
    mappings=["cu2023"],
    place=GEO / "cu" / "cu_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Cuba's censuses do not ask about language, and nearly every Cuban grows up speaking "
        "Spanish, so everyone is drawn as a Spanish speaker. The small Haitian Creole speaking "
        "communities of the east, descended from early 20th century cane workers, have no "
        "count and are drawn as Spanish too. The population is the statistics office's 2023 "
        "figure for each municipality, before part of the large emigration of recent years. "
        "The Guantánamo Bay Naval Base, which the United States administers, is not drawn."),
)
