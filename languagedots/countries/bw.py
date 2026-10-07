# Botswana. Census 2011, language spoken at home, persons aged 2+, by the 28 census districts
# (sources/bw_census.py). Placed on religiondots' Kontur hexes, whose ADM3 unit ids begin with
# their census district's ADM2 code (BW120238 is in BW1202, Central Mahalapye). Read-only.
# The record is sources/bw.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import bw2011
    df = pd.read_csv(NORM / "bw.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 28:
        raise SystemExit(f"bw.csv: {df['geo_id'].nunique()} census districts, expected 28")
    df["node"] = df["source_category"].map(bw2011.NAMES)
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(bw2011.EXCLUDED))
    if unresolved:
        raise SystemExit(f"bw.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna()].copy()
    df["unit"] = df["geo_id"]
    return by_unit(df)


def _district(g):
    # ADM3 pcode BWddssnn -> census district BWddss (COD-AB 2011 ADM2_PCODE)
    u = g["unit"].astype(str)
    if not u.str.fullmatch(r"BW\d{6}").all():
        raise SystemExit("bw_hexes.gpkg: a unit id is not an 8-character BW ADM3 pcode")
    return u.str[:6]


ENTRY = dict(
    name="Botswana",
    source="Population and Housing Census 2011, National Statistical Tables Report, "
           "Category B Table 2 (Statistics Botswana, 2015)",
    how="census, 2011, language spoken at home",
    parts=[dict(covers="Everyone aged 2 and over", source="2011 census, language spoken at "
                "home", rest=True)],
    grain="28 census districts, 69,000 people on average",
    gap="104,666 children under two, who were not asked, and 888 people who gave no answer",
    view=[19.6, -27.2, 29.6, -17.6],
    counts=_counts,
    mappings=["bw2011"],
    place=RD_GEO / "bw" / "bw_hexes.gpkg",
    place_unit=_district,
    place_weight=pop_weight,
    note_public=(
        "These are the 2011 census figures. The 2022 census asked the same question but has "
        "published it by district only where each language is largest. The census counts the "
        "San languages together as Sesarwa, 1.7% of people, and they are drawn as one "
        "remainder. In the west a single census district runs across hundreds of kilometres "
        "of the Kalahari, so a district's languages are spread over it by where people live "
        "and not by where each language is spoken."),
)
