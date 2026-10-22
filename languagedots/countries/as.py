# American Samoa. 2020 Island Areas Census, DHC table PCT24 by county (sources/as_census.py), on
# religiondots' 10 units and its Kontur hexes (read-only): Manu'a's five counties are one unit.
# sources/as.md is the record.
from _shared import *  # noqa: F401,F403


def _counts():
    import as2020
    df = pd.read_csv(NORM / "as.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 10:
        raise SystemExit(f"as.csv: {df['geo_id'].nunique()} units, expected 10 -- run sources/as_census.py")
    df["node"] = df["source_category"].map(as2020.resolve)
    df["unit"] = df["geo_id"]      # religiondots' as_lookup.csv: unit names
    df["tier"] = "measured"
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="American Samoa",
    source="2020 Island Areas Census of American Samoa, Demographic and Housing Characteristics, "
           "table PCT24 (U.S. Census Bureau)",
    how="census, 2020, language spoken at home (the language other than English, where there is one)",
    parts=[dict(covers="People aged 5 and over in households",
                source="2020 census, language spoken at home", rest=True)],
    grain="10 counties (the five counties of Manu'a as one), 4,400 people aged 5 and over on average",
    gap="children under 5 and people in group quarters, 5,536 (11.1%)",
    view=[-171.0, -14.45, -169.4, -14.2],
    counts=_counts,
    mappings=["as2020"],
    place=RD_GEO / "as" / "as_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked each person aged 5 or over whether they speak a language other than "
        "English at home, and if so which one. Someone who speaks Samoan and English at home is "
        "counted under Samoan, so English here means people who speak only English at home. "
        "Samoan is spoken at home by 88% of people. Tongan is 2% of the territory and 4% of "
        "Tualauta county, the Tafuna plain, which holds 46% of the people. "
        "The census groups the other Pacific Island languages together, so they are drawn as one "
        "group without saying which, and the same for Asian languages."),
)
