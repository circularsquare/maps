# Cape Verde. No language question: everyone drawn on Kabuverdianu, per concelho, on the 2021
# census populations (sources/cv_pop.py); the Afrobarometer's home-language answers (99.5%+
# Crioulo in every round) corroborate. Placed on religiondots' Kontur hexes (read-only).
# Record: sources/cv.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import cv2021
    df = pd.read_csv(NORM / "cv.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 22:
        raise SystemExit(f"cv.csv: {df['geo_id'].nunique()} concelhos, expected 22 -- "
                         "run sources/cv_pop.py")
    df["node"] = df["source_category"].map(cv2021.resolve)
    df["unit"] = df["geo_id"]   # religiondots' cv_lookup.csv: geo_id == unit
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Cape Verde",
    source=("population: 2021 census by concelho (INE Cabo Verde); no language source in the "
            "census; Afrobarometer home language 2008-2022 as a check"),
    how=("no language question: everyone drawn as Kabuverdianu, the home language of 99.5% "
         "or more in every Afrobarometer round"),
    parts=[dict(covers="Everyone",
                source="2021 census population, drawn as Kabuverdianu (Afrobarometer 2008-2022: "
                       "99.5% or more)", rest=True)],
    grain="22 concelhos, 22,000 people on average",
    gap="foreign residents, who have no count by concelho, are drawn as Kabuverdianu",
    view=[-25.5, 14.7, -22.6, 17.3],
    counts=_counts,
    mappings=["cv2021"],
    place=RD_GEO / "cv" / "cv_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Cape Verde's census does not ask about language, so everyone is drawn as a speaker of "
        "Kabuverdianu, the creole spoken at home across the islands. In six rounds of the "
        "Afrobarometer survey from 2008 to 2022, 99.5% or more of Cape Verdeans named it as "
        "their home language in every round. Portuguese, the official language, is learned at "
        "school. Foreign residents, mostly from West Africa, have no count by concelho and are "
        "drawn as Kabuverdianu speakers too."),
)
