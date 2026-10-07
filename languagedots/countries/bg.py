# Bulgaria. Census 2021 mother tongue by obshtina (sources/bg_census.py), on religiondots' 1 km
# census population grid for the same 265 obshtini (read only). The record is sources/bg.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import bg2021
    df = pd.read_csv(NORM / "bg.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "obshtina"].copy()
    if df["geo_id"].nunique() != 265:
        raise SystemExit(f"bg.csv: {df['geo_id'].nunique()} obshtini, expected 265")
    # NSI's obshtina codes are the grid's `unit` (GISCO LAU_ID); sources/bg_census.py checks
    # them against religiondots' bg.csv, and religiondots' bg_geo.py against the grid
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(bg2021.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    return by_unit(df)


ENTRY = dict(
    name="Bulgaria",
    source="Преброяване 2021, population by mother tongue, by statistical region, oblast and "
           "obshtina (National Statistical Institute)",
    how="census, 2021, mother tongue",
    parts=[dict(covers="Everyone", source="2021 census, mother tongue", rest=True)],
    grain="265 obshtini (municipalities), 24,600 people on average",
    gap="676,916 people, 10.4%: 616,681 added from administrative registers and never asked, "
        "49,602 who did not wish to answer and 10,633 who could not say",
    view=[22.2, 41.1, 28.8, 44.3],
    counts=_counts,
    mappings=["bg2021"],
    place=RD_GEO / "bg" / "bg_grid_1km.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Mother tongue is the first language learned at home, and the question was voluntary. "
        "The census names three languages, Bulgarian, Turkish and Romani, and puts every other "
        "language under one other. The Muslims of the western Rhodopes mostly speak Bulgarian "
        "and are drawn as Bulgarian. About one Roma in six gave Bulgarian or Turkish as their "
        "mother tongue. 9.5% of people were added from administrative registers and never "
        "asked; they are not drawn, and they are most of Sofia's gap of 17.9%. Dots inside "
        "each municipality follow the census's own 1 km population grid."),
)
