# Slovakia. SODB 2021, mother tongue by obec (sources/sk_sodb.py), placed on religiondots' census
# 1 km grid (the census's own grid, split among the 2,927 obce by area), by population.
# The record is sources/sk.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import sk2021
    df = pd.read_csv(NORM / "sk.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "obec"]
    if df["geo_id"].nunique() != 2927:
        raise SystemExit(f"sk.csv: {df['geo_id'].nunique()} obce, expected 2,927")
    df["node"] = df["source_category"].map(sk2021.resolve)
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"])
                        - sk2021.NOT_STATED - {sk2021.TOTAL})
    if unresolved:
        raise SystemExit(f"sk.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"]
    # every row is a published count for the obec: all measured
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Slovakia",
    source="Sčítanie obyvateľov, domov a bytov 2021, mother tongue by obec "
           "(Statistical Office of the Slovak Republic)",
    how="census, 2021, mother tongue",
    parts=[dict(covers="Everyone", source="2021 census, mother tongue", rest=True)],
    grain="2,927 municipalities and city districts, 1,860 people on average",
    gap="312,364 people, 5.7%, whose mother tongue was not ascertained",
    view=[16.83, 47.73, 22.57, 49.61],
    counts=_counts,
    mappings=["sk2021"],
    place=RD_GEO / "sk" / "sk_grid_1km.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked for one mother tongue, defined as the language spoken at home in "
        "childhood. Within each municipality the dots follow the census's own 1 km population "
        "grid. Romani was named by 100,526 people, a fraction of Slovakia's Roma, many of whom "
        "named Slovak or Hungarian. Rusyn and Ukrainian are counted separately. The 5.7% whose "
        "mother tongue was not recorded are not drawn; they are most numerous in Košice's "
        "Luník IX, where they are nearly half the population."),
)
