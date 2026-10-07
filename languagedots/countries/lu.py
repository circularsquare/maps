# Luxembourg. Census of 8 November 2021, main language: the six languages of the form by commune
# from STATEC's geoportail.lu layers, the write-ins from the national table (sources/lu_census.py),
# on religiondots' Kontur hexes for the 102 communes. The record is sources/lu.md.
from _shared import *  # noqa: F401,F403

UNITS = 102


def _counts():
    import lu2021
    df = pd.read_csv(NORM / "lu.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "commune"]
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"lu.csv: {df['geo_id'].nunique()} communes, expected {UNITS}")
    df["node"] = df["source_category"].map(lu2021.resolve)
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Luxembourg",
    source="Recensement de la population 2021, main language by commune (STATEC, geoportail.lu "
           "layers) and national tables 1 and 3 of RP2021 n°8 (STATEC and University of Luxembourg)",
    how="census, 2021, main language",
    parts=[
        dict(covers="Luxembourgish, French, German, Portuguese, English",
             source="2021 census, main language, by commune",
             people=482_489),
        dict(covers="Italian and other languages",
             source="2021 census, main language, national totals; their commune total is the "
                    "census's, the split estimated",
             rest=True),
    ],
    grain="102 communes, 6,300 people on average",
    gap=("80,849 people (12.6%) who did not answer or were marked as not yet old enough to "
         "speak."),
    view=[5.7, 49.42, 6.55, 50.2],
    counts=_counts,
    mappings=["lu2021"],
    place=RD_GEO / "lu" / "lu_grid_400m.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked for one main language, the one a person thinks in and knows best, "
        "and most people in Luxembourg speak several more every day. Italian is placed where "
        "Italian citizens live. Non-response was higher among recent immigrants, so the census "
        "itself warns that the three national languages are somewhat overstated. Cross-border "
        "workers, about half the workforce, live abroad and are not on the map."),
)
