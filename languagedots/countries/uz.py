# Uzbekistan. 2026 census native language by region, from the preliminary results volume
# (sources/uz_census.py), on religiondots' Kontur hexes for the 14 regions. sources/uz.md is the record.
from _shared import *  # noqa: F401,F403


def _counts():
    import uz2026
    df = pd.read_csv(NORM / "uz.csv", dtype={"geo_id": str})
    # religiondots keys its 14 regions on the COD-AB p-codes uz_census.py already writes
    lut = pd.read_csv(RD_GEO / "uz" / "uz_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any() or df["unit"].nunique() != 14:
        raise SystemExit(f"uz: regions not in religiondots' uz_lookup.csv: "
                         f"{sorted(df.loc[df['unit'].isna(), 'geo_id'].unique())}")
    df["node"] = df["source_category"].map(uz2026.resolve)
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Uzbekistan",
    source=("Population and Agriculture Census 2026, preliminary results, population by main "
            "language of communication (mother tongue), by region (National Statistics Committee "
            "of the Republic of Uzbekistan)"),
    how="census, 2026, native language",
    parts=[dict(covers="Everyone", source="2026 census, preliminary results, native language",
                rest=True)],
    grain="14 regions, 2.8 million people on average",
    view=[55.9, 37.1, 73.2, 45.6],
    counts=_counts,
    mappings=["uz2026"],
    place=RD_GEO / "uz" / "uz_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2026 census, the first since 1989, asked each person's native language, which the "
        "census defines as the first language learned in childhood and the one mainly used day to "
        "day. In the countries of the former Soviet Union this question leans towards identity "
        "rather than everyday use. The National Statistics Committee has so far printed the "
        "answers for the 14 regions only, so within a region every language is spread by where "
        "people live. How many people in Uzbekistan speak Tajik has long been disputed: the "
        "census counts 3.6% of Samarkand region and 0.7% of Bukhara region as native Tajik "
        "speakers, and fewer people named Tajik as their language (851,000) than it counted "
        "as Tajik by nationality (1.28 million)."),
)
