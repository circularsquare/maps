# United States. ACS 2020-2024 5-year (sources/us_acs.py): C16001 by tract, each group shared out
# by its PUMA's mix from B16001 and PUMS; placed on 2024 tracts (sources/us_geo.py).
from _shared import *  # noqa: F401,F403


def _counts():
    import us2024
    t = pd.read_csv(NORM / "us.csv", dtype={"geo_id": str, "puma": str})
    sp = pd.read_csv(NORM / "us_split.csv", dtype={"puma": str})
    en = t[t["source_category"] == "Speak only English"].copy()
    grp = t[t["source_category"] != "Speak only English"].rename(columns={"source_category": "c_group"})
    df = grp.merge(sp[["puma", "c_group", "source_category", "share"]], on=["puma", "c_group"],
                   how="left", validate="many_to_many")
    if df["share"].isna().any():
        bad = df[df["share"].isna()]
        raise SystemExit(f"us: {len(bad)} tract groups with no PUMA mix: {bad.head(3).to_dict('records')}")
    df["count"] = df["count"] * df["share"]
    # A group of one language (Spanish, Korean, Vietnamese, Arabic) is the tract count as
    # published; any group shared out by a PUMA's mix is `derived`.
    codes = sp.groupby("c_group")["source_category"].nunique()
    single = set(codes[codes == 1].index)
    df["tier"] = df["c_group"].map(lambda g: "measured" if g in single else "derived")
    en["tier"] = "measured"
    df = pd.concat([en[["geo_id", "source_category", "count", "tier"]],
                    df[["geo_id", "source_category", "count", "tier"]]], ignore_index=True)
    df["node"] = df["source_category"].map(us2024.resolve)
    df = df.rename(columns={"geo_id": "unit"})
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="United States",
    source="American Community Survey 2020-2024 5-year: tables C16001 (tract) and B16001 (PUMA), "
           "and the public use microdata (U.S. Census Bureau)",
    how="survey, 2020-2024, language spoken at home (the language other than English, where there is one)",
    parts=[
        dict(covers="English only, Spanish, Korean, Vietnamese and Arabic",
             source="ACS 2020-2024, language spoken at home, per tract (C16001)",
             people=292_724_320),
        dict(covers="Chinese, Tagalog and the survey's wider groups",
             source="each tract's group shared out by the ACS microdata for its area of about "
                    "128,000 people (PUMA)",
             rest=True),
    ],
    grain="83,600 tracts, 3,800 people aged 5 and over on average; languages inside the "
          "survey's groups shared out by 2,462 areas of 128,000 people",
    gap="children under 5, 18.8 million (5.6%), whom the survey does not ask",
    view=[-125.0, 24.0, -66.5, 49.8],
    counts=_counts,
    mappings=["us2024"],
    # One polygon per unit, except two Suffolk County units of 12 and 2 tracts that share equally
    # (sources/us_geo.py). The tracts are cut by Kontur hexes, each tract keeping its equal share,
    # so a rural tract's dots follow where its people live instead of spreading over all its land
    # (sources/kontur_cut.py; 2026-10-08)
    place=GEO / "us" / "us_konturcut.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The American Community Survey asks whether each person aged 5 or over speaks a language "
        "other than English at home, and if so which one. Someone who speaks Spanish and English "
        "at home is counted under Spanish, so English here means people who speak only English "
        "at home. Children under 5 are not asked and are not drawn. For each census tract the "
        "survey publishes a few single languages and wider groups such as \"other "
        "Indo-European languages\". The languages inside a group come from the survey's "
        "microdata for the surrounding area of about 128,000 people, so a tract's Hmong or "
        "Navajo dots are its group's speakers shared out in that area's proportions."),
)
