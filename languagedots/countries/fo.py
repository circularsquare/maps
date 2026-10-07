# Faroe Islands. Census 2011, first language, Hagstova MT16 (sources/fo_census.py), on religiondots'
# Kontur hexes for the same 7 districts, weighted by the November 2011 village register (read
# only; religiondots sources/fo_geo.py). The record is sources/fo.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import geopandas as gpd

    import fo2011
    df = pd.read_csv(NORM / "fo.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 7:
        raise SystemExit(f"fo.csv: {df['geo_id'].nunique()} districts, expected 7; re-run "
                         "`python sources/fo_census.py`")
    df["node"] = df["source_category"].map(fo2011.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    df["unit"] = df["geo_id"]
    # religiondots' hex layer carries the district exactly as MT16 and MT325 print it
    rd = set(gpd.read_file(RD_GEO / "fo" / "fo_hexes.gpkg", columns=["unit"],
                           ignore_geometry=True)["unit"])
    if set(df["unit"]) != rd:
        raise SystemExit(f"fo: districts differ from religiondots' fo_hexes.gpkg: "
                         f"{sorted(set(df['unit']) ^ rd)}")
    return by_unit(df)


ENTRY = dict(
    name="Faroe Islands",
    source="Census 2011, table MT16, first language by district (Hagstova Føroya)",
    how="census, 2011, first language",
    parts=[dict(covers="Everyone", source="2011 census, first language", rest=True)],
    grain="7 districts, 6,900 people on average",
    gap="41 people recorded with no language (0.1%), about half of them children under 5",
    view=[-7.75, 61.35, -6.2, 62.42],
    counts=_counts,
    mappings=["fo2011"],
    place=RD_GEO / "fo" / "fo_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2011 census recorded one first language for every resident, children included, "
        "and printed it for seven districts. Faroese is the first language of 93.8% of people "
        "and Danish of 3.2%, rising to 4.4% in southern Streymoy, the district of Tórshavn. "
        "Other languages were printed only in groups by part of the world, and the 1,348 "
        "people in groups that mix language families are drawn in grey."),
)
