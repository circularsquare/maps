# Turkmenistan. 2022 census mother tongue by nationality, for Ashgabat and the five velayats
# (sources/tm_census.py), on religiondots' Kontur hexes for the same six units. sources/tm.md is
# the record.
from _shared import *  # noqa: F401,F403


def _counts():
    import tm2022
    df = pd.read_csv(NORM / "tm.csv")
    lut = pd.read_csv(RD_GEO / "tm" / "tm_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any() or df["unit"].nunique() != 6:
        raise SystemExit(f"tm: velayats not in religiondots' tm_lookup.csv: "
                         f"{sorted(df.loc[df['unit'].isna(), 'geo_id'].unique())}")
    df["node"] = df["source_category"].map(tm2022.resolve)
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Turkmenistan",
    source=("Complete Population and Housing Census 2022, section 4, tables 4.12-4.27, population "
            "by nationality and mother tongue (State Committee of Turkmenistan on Statistics)"),
    how="census, 2022, mother tongue",
    parts=[dict(covers="Everyone", source="2022 census, mother tongue", rest=True)],
    grain="Ashgabat and 5 velayats, 1.2 million people on average",
    view=[52.4, 35.1, 66.7, 42.8],
    counts=_counts,
    mappings=["tm2022"],
    place=RD_GEO / "tm" / "tm_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2022 census asked each person's mother tongue, which in the countries of the former "
        "Soviet Union leans towards identity rather than everyday use. It prints the answers "
        "only for Ashgabat and the five velayats, so inside each the languages are spread over "
        "where people live, not placed where they are spoken: Dashoguz's Uzbek and Mary's "
        "Balochi cover the whole velayat. Many people named a language other than their "
        "nationality's: in Lebap 71% of Uzbeks named Turkmen. The census comes from a "
        "state that publishes little, and nobody outside the statistics committee can check "
        "its figures."),
)
