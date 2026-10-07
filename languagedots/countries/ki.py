# Kiribati. No home-language question: everyone drawn as Gilbertese per island on the 2020 census
# populations (sources/ki_pop.py). Placed on religiondots' Kontur hexes for its 24 islands
# (read-only). Record: sources/ki.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ki2020
    df = pd.read_csv(NORM / "ki.csv")
    if df["geo_id"].nunique() != 24:
        raise SystemExit(f"ki.csv: {df['geo_id'].nunique()} islands, expected 24 -- "
                         "run sources/ki_pop.py")
    df["node"] = df["source_category"].map(ki2020.resolve)
    df["unit"] = df["geo_id"]   # religiondots' ki_lookup.csv `unit`
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Kiribati",
    source="Population and Housing Census 2020, island profiles (Kiribati National Statistics Office)",
    how=("no home-language question: everyone drawn as Gilbertese, the first language of "
         "nearly all I-Kiribati; census 2020"),
    parts=[
        dict(covers="Everyone", source="2020 census populations per island, drawn as Gilbertese",
             rest=True),
    ],
    grain="24 islands, 5,000 people on average",
    view=[169.0, -3.5, 203.5, 5.5],
    counts=_counts,
    mappings=["ki2020"],
    place=RD_GEO / "ki" / "ki_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Kiribati's census asks only whether a person speaks English at home, so everyone is "
        "drawn as a speaker of Gilbertese, which nearly all I-Kiribati grow up with. The few "
        "Tuvaluan speakers and English-speaking households are drawn as Gilbertese too."),
)
