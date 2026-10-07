# Lithuania. Gyventojų surašymas 2021 mother tongue by municipality (sources/lt_census.py), on
# religiondots' Kontur hexes for the same 60 savivaldybės (read only). The record is sources/lt.md.
# Withheld (confidential) cells are estimated from the margins in sources/lt_census.py, tier
# `derived` (Anita, 2026-10-05).
from _shared import *  # noqa: F401,F403


def _counts():
    import lt2021
    df = pd.read_csv(NORM / "lt.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 60:
        raise SystemExit(f"lt.csv: {df['geo_id'].nunique()} municipalities, expected 60")
    # religiondots' hexes carry the same two-digit savivaldybė code in `unit` (its lt_geo.py
    # joined GISCO LAU_ID to the census code), so the join is the identity
    df["unit"] = df["geo_id"].str.zfill(2)
    df["node"] = df["source_category"].map(lt2021.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    # `derived` rows are the office's confidential cells, estimated from the municipality and
    # county margins (sources/lt_census.py impute)
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Lithuania",
    source="Gyventojų ir būstų surašymas 2021, population by mother tongue, by municipality "
           "(Statistics Lithuania, dataflow S3R778_GBS010509)",
    how="census, 2021, mother tongue, from the census's sample survey and the office's estimates",
    parts=[dict(covers="Everyone",
                source="2021 census, mother tongue (sample survey of about 171,000 people)",
                rest=True)],
    grain="60 municipalities, 47,000 people on average",
    gap="No one is recorded as not stated.",
    view=[20.8, 53.8, 26.9, 56.5],
    counts=_counts,
    mappings=["lt2021"],
    place=RD_GEO / "lt" / "lt_grid_400m.gpkg",
    place_unit=lambda g: g["unit"].astype(str).str.zfill(2),
    place_weight=pop_weight,
    note_public=(
        "Mother tongue (gimtoji kalba) is the language a person names as their own, which in "
        "former Soviet countries leans towards identity rather than everyday use. In Lithuania "
        "it does not simply follow ethnicity: 19,260 people who gave Polish ethnicity named "
        "Russian as their mother tongue. Registers do not hold language, so Statistics "
        "Lithuania surveyed about 171,000 people and estimated the rest. 49,066 people gave two "
        "mother tongues without the census saying which, and are drawn in grey on a category "
        "of their own. Cells the office withheld as confidential (864 people) are estimated "
        "from the totals; they hold 56% of German and 18% of Latvian speakers."),
)
