# Aruba. Census 2010, Table P-D.2, language most spoken in the household, by 49 populated zones
# (sources/aw_census.py), on religiondots' Kontur hexes cut along CBS Aruba's zone polygons
# (sources/aw_geo.py). Record: sources/aw.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import aw2010
    df = pd.read_csv(NORM / "aw.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "zone"]
    if df["geo_id"].nunique() != 49 or abs(df["count"].sum() - 101_484) > 10:
        raise SystemExit("aw.csv: expected 49 zones summing to 101,484 within rounding")
    df["node"] = df["source_category"].map(aw2010.resolve)
    df = df[df["node"].notna()]
    df["unit"] = df["geo_id"]
    return by_unit(df)


ENTRY = dict(
    name="Aruba",
    source="Fifth Population and Housing Census 2010, Table P-D.2 (Central Bureau of Statistics "
           "Aruba); zone polygons from CBS Aruba's ArcGIS service",
    how="census, 2010, language most spoken in the household (one answer per household)",
    parts=[dict(covers="Everyone", source="2010 census, language most spoken in the household",
                rest=True)],
    grain="49 census zones, 2,100 people on average",
    gap=("1,995 people, 2.0%: 1,563 who did not speak yet (nine in ten under five) and 432 "
         "whose language was not reported"),
    view=[-70.10, 12.39, -69.84, 12.65],
    counts=_counts,
    mappings=["aw2010"],
    place=GEO / "aw" / "aw_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2010 census asked each household which language it spoke most, and gave that "
        "answer to everyone in it. Papiamento was the language of 68% of people, Spanish 14%, "
        "English 7% and Dutch 6%. English was the main language of more than four in ten people "
        "in parts of San Nicolas, and Spanish of more than half in central Oranjestad. The 2020 "
        "census published its language answers only as pairs by region, with no language "
        "named for the 19% whose household did not speak Papiamento most, so the map draws "
        "2010."),
)
