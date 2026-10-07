# South Africa. Census 2011, first language spoken in the household, by small area
# (sources/za_c11.py), on Kontur hexes cut to the 84,907 small areas (sources/za_geo.py).
# The record is sources/za.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import za2011
    df = pd.read_csv(NORM / "za.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "sal"]
    if df["geo_id"].nunique() != 84_907:
        raise SystemExit(f"za.csv: {df['geo_id'].nunique()} small areas, expected 84,907")
    df["node"] = df["source_category"].map(za2011.resolve)
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(za2011.EXCLUDED))
    if unresolved:
        raise SystemExit(f"za.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna()]
    df["unit"] = df["geo_id"]
    return by_unit(df)


ENTRY = dict(
    name="South Africa",
    source="Census 2011, persons by language per small area (Statistics South Africa, "
           "Community Profiles; extracted by Adrian Frith)",
    how="census, 2011, home language (the first of the two spoken most often)",
    parts=[dict(covers="Everyone", source="2011 census, home language", rest=True)],
    grain="84,907 small areas, 600 people on average",
    gap="808,905 people, 1.6%, coded 'not applicable', concentrated in prisons, mine hostels "
        "and other collective quarters",
    view=[16.2, -35.0, 33.1, -22.0],
    counts=_counts,
    mappings=["za2011"],
    place=GEO / "za" / "za_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "These are 2011 figures. The 2022 census asked a similar question, but below the "
        "province its answers are published only to registered users, so the open 2011 "
        "small-area tables are drawn instead. The census asked for the two languages each "
        "person speaks most often at home; this map draws the first one named."),
)
