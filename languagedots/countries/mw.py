# Malawi. 2018 PHC tribe by district (Main Report Table E4), read as home language with
# Afrobarometer R5, R7-R9's shares per tribe (sources/mw_census.py); every row `modelled`. On
# religiondots' Kontur hexes for its 32 district units. The record is sources/mw.md.
from _shared import *  # noqa: F401,F403

UNITS = 32
MALAWIANS = 17_506_538


def _counts():
    import mw2018
    df = pd.read_csv(NORM / "mw.csv", dtype={"geo_id": str})
    lut = pd.read_csv(RD_GEO / "mw" / "mw_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any() or df["unit"].nunique() != UNITS:
        raise SystemExit(f"mw.csv: {df['unit'].nunique()} units joined, expected {UNITS}")
    if abs(df["count"].sum() - MALAWIANS) > 100:
        raise SystemExit(f"mw.csv sums to {df['count'].sum():,.0f}, not {MALAWIANS:,}")
    df["node"] = df["source_category"].map(mw2018.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"mw.csv answers with no node: {missing}")
    out = by_unit(df[df["count"] > 0])
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Malawi",
    source=("2018 Malawi Population and Housing Census, Main Report Table E4, tribe by "
            "district (National Statistical Office); home language by tribe from "
            "Afrobarometer rounds 5, 7, 8 and 9 (2012-2022)"),
    how=("census, 2018, tribe, shared across languages by a survey's home language per tribe "
         "(Afrobarometer, 2012-2022); Chichewa from its round 7 mother-tongue question"),
    parts=[dict(covers="Malawians",
                source="2018 census tribe, read as language through Afrobarometer 2012-2022, "
                       "about 6,000 adults",
                rest=True)],
    grain="28 districts and 4 cities, 547,000 people on average",
    gap="non-Malawians, about 57,000 (0.3%), whom the tribe table leaves out",
    view=[32.5, -17.3, 36.2, -9.2],
    counts=_counts,
    mappings=["mw2018"],
    place=RD_GEO / "mw" / "mw_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Malawi's census does not ask about language. It asks tribe, published by district "
        "for 12 named tribes, and not everyone speaks their tribe's language. Each "
        "district's tribes are shared across languages using the Afrobarometer survey of "
        "about 6,000 adults. Many Lomwe, Ngoni, Yao and Sena use Chichewa at home, so Chichewa "
        "is taken from the survey's mother-tongue question: half of Malawians, against about "
        "70% who speak it at home. Chichewa, Chinyanja and Chimang'anja are one language under "
        "three names, drawn as people named them."),
)
