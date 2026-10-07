# Puerto Rico. Puerto Rico Community Survey 2020-2024 5-year (sources/pr_acs.py): C16001 by tract,
# each group shared out by its PUMA's mix from B16001 and PUMS, as countries/us.py does; placed on
# Kontur hexes cut to the 2024 tracts (sources/pr_geo.py). sources/pr.md is the record.
from _shared import *  # noqa: F401,F403


def _counts():
    import pr2024
    t = pd.read_csv(NORM / "pr.csv", dtype={"geo_id": str, "puma": str})
    sp = pd.read_csv(NORM / "pr_split.csv", dtype={"puma": str})
    en = t[t["source_category"] == "Speak only English"].copy()
    grp = t[t["source_category"] != "Speak only English"].rename(columns={"source_category": "c_group"})
    df = grp.merge(sp[["puma", "c_group", "source_category", "share"]], on=["puma", "c_group"],
                   how="left", validate="many_to_many")
    if df["share"].isna().any():
        bad = df[df["share"].isna()]
        raise SystemExit(f"pr: {len(bad)} tract groups with no PUMA mix: {bad.head(3).to_dict('records')}")
    df["count"] = df["count"] * df["share"]
    # Spanish (95%) and the other one-language groups are tract counts as published; a group
    # shared out by a PUMA's mix is `derived` (about 4,400 people).
    codes = sp.groupby("c_group")["source_category"].nunique()
    single = set(codes[codes == 1].index)
    df["tier"] = df["c_group"].map(lambda g: "measured" if g in single else "derived")
    en["tier"] = "measured"
    df = pd.concat([en[["geo_id", "source_category", "count", "tier"]],
                    df[["geo_id", "source_category", "count", "tier"]]], ignore_index=True)
    df["node"] = df["source_category"].map(pr2024.resolve)
    df = df.rename(columns={"geo_id": "unit"})
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Puerto Rico",
    source="Puerto Rico Community Survey 2020-2024 5-year: tables C16001 (tract) and B16001 (PUMA), "
           "and the public use microdata (U.S. Census Bureau)",
    how="survey, 2020-2024, language spoken at home (the language other than English, where there is one)",
    parts=[dict(covers="Everyone aged 5 and over",
                source="Puerto Rico Community Survey 2020-2024, language spoken at home",
                rest=True)],
    grain="921 tracts, 3,400 people aged 5 and over on average",
    gap="children under 5, 101,500 (3.1%), whom the survey does not ask",
    view=[-67.95, 17.85, -65.2, 18.6],
    counts=_counts,
    mappings=["pr2024"],
    place=GEO / "pr" / "pr_tracts.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The Puerto Rico Community Survey is the American Community Survey as run in Puerto Rico, "
        "and asks the same question: whether each person aged 5 or over speaks a language other "
        "than English at home, and if so which one. Someone who speaks Spanish and English at "
        "home is counted under Spanish, so English here means people who speak only English at "
        "home. Languages other than Spanish and English come to about 4,700 people; inside the "
        "survey's wider groups, such as \"other Indo-European languages\", the language is "
        "taken from its microdata for areas of about 130,000 people."),
)
