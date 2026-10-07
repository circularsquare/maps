# US Virgin Islands. 2020 Island Areas Census, DHC table PBG5 (language spoken at home, four
# groups) by block group (sources/vi_census.py), placed on Kontur hexes cut to the 2020 block
# groups (sources/vi_geo.py). sources/vi.md is the record.
from _shared import *  # noqa: F401,F403


def _counts():
    import vi2020
    df = pd.read_csv(NORM / "vi.csv", dtype={"geo_id": str})
    if df["count"].sum() != 80_430 or df["geo_id"].nunique() != 89:
        raise SystemExit("vi.csv: expected 80,430 people over 89 block groups; rerun sources/vi_census.py")
    df["node"] = df["source_category"].map(vi2020.resolve)
    df = df.rename(columns={"geo_id": "unit"})
    out = by_unit(df)
    out["tier"] = "measured"
    return out


ENTRY = dict(
    name="US Virgin Islands",
    source="2020 Island Areas Census, Demographic and Housing Characteristics File, table PBG5 "
           "(U.S. Census Bureau)",
    how="census, 2020, language spoken at home (the language other than English, where there is one)",
    parts=[dict(covers="Everyone aged 5 and over",
                source="2020 census, language spoken at home, four groups", rest=True)],
    grain="89 block groups, 900 people aged 5 and over on average",
    gap="children under 5 and people in group quarters, 6,716 together (7.7%), whom the table "
        "does not cover",
    view=[-65.1, 17.65, -64.55, 18.42],
    counts=_counts,
    mappings=["vi2020"],
    place=GEO / "vi" / "vi_bg.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked each person aged 5 or over whether they speak a language other than "
        "English at home, and if so which one. Someone who speaks Spanish and English at home is "
        "counted under Spanish, so English here means people who speak only English at home, "
        "including Virgin Islands Creole English, which the census does not count apart. The "
        "census publishes four groups only. Its group \"French, Haitian, or Cajun\" (9% of "
        "people) is drawn as French-based creoles with the language not named: in the islands it "
        "is mostly the Kweyol of people from Dominica and St Lucia and the Haitian Creole of "
        "people from Haiti, with some French. \"Other languages\" (4%) are not broken down."),
)
