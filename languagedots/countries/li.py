# Liechtenstein. Volkszählung 2020, main language by Gemeinde (sources/li_census.py), on
# religiondots' 400 m Kontur hexes for the 11 Gemeinden (read-only). The record is sources/li.md.
from _shared import *  # noqa: F401,F403

UNITS = 11


def _counts():
    import li2020
    df = pd.read_csv(NORM / "li.csv")
    df = df[df["geo_level"] == "gemeinde"]
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"li.csv: {df['geo_id'].nunique()} Gemeinden, expected {UNITS}")
    df["node"] = df["source_category"].map(li2020.resolve)
    if df["node"].isna().any():
        raise SystemExit(f"li.csv categories that resolve to nothing: "
                         f"{sorted(set(df.loc[df['node'].isna(), 'source_category']))}")
    df = df[df["count"] > 0]
    # religiondots' grid carries the Gemeinde name as `unit`, the same string the census uses
    df["unit"] = df["geo_id"]
    df["tier"] = "measured"
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Liechtenstein",
    source="Volkszählung 2020, eTab table 213.011d, main language by Gemeinde "
           "(Amt für Statistik)",
    how="census, 2020, main language",
    parts=[dict(covers="Everyone", source="2020 census, main language", rest=True)],
    grain="11 Gemeinden, 3,550 people on average",
    view=[9.41, 47.03, 9.68, 47.29],
    counts=_counts,
    mappings=["li2020"],
    place=RD_GEO / "li" / "li_grid_400m.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Each resident has one main language in this census. German includes the local "
        "Alemannic dialect, which the table does not count apart. The census groups some "
        "languages by region, and those groups are drawn as other: East Asian languages "
        "(179 people), West Asian (28), Indo-Aryan and Dravidian (24)."),
)
