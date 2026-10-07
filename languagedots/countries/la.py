# Laos. 2015 census ethnicity at village level (K4D's ten ethno-linguistic categories), split into
# the 49 official groups on the 2011 agricultural census's village groups and raked to the
# census's national table (sources/la_census.py), read as languages through taxonomy/la2015.py:
# a proxy Anita allowed on 2026-10-05. Placed on religiondots' 400m Kontur hexes for the same
# 8,499 villages (read-only). The record is sources/la.md.
from _shared import *  # noqa: F401,F403

# Groups whose census category holds only them, so their village counts are the census's own
# (Lao also takes a modelled share of the Tai-Thai category; most of it is its own category).
EXACT = {"Lao", "Hmong", "Ewmien", "Other, not stated and foreigners"}


def _counts():
    import la2015
    df = pd.read_csv(NORM / "la.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 8499:
        raise SystemExit("la.csv: expected 8,499 villages; re-run sources/la_census.py")
    df["node"] = df["source_category"].map(la2015.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"la.csv categories that resolve to nothing: {missing}")
    df["tier"] = df["source_category"].map(lambda s: "derived" if s in EXACT else "modelled")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Laos",
    source=("Population and Housing Census 2015, ethno-linguistic category by village (Lao "
            "Statistics Bureau, through the K4D atlas platform) and Table P2.7, population by "
            "ethnic group; 2011 Lao Census of Agriculture, the three largest ethnic groups of "
            "each village (same platform)"),
    how="census, 2015, ethnic group (no language question), each group drawn as its language",
    parts=[
        dict(covers="Lao, Hmong and Iu Mien",
             source="2015 census, ethnic category by village",
             nodes=["kradai.lao", "hmongmien.hmong", "hmongmien.dao"]),
        dict(covers="Other named groups",
             source="2015 census, national totals by group, placed on villages by the 2011 "
                    "agricultural census",
             rest=True),
        dict(covers="Other, not stated and foreigners",
             source="2015 census, drawn as other",
             nodes=["other"]),
    ],
    grain="8,499 villages, 760 people on average",
    gap="10,746 people (0.2%) in villages the village file does not carry.",
    view=[100.0, 13.8, 108.0, 22.6],
    counts=_counts,
    mappings=["la2015"],
    place=RD_GEO / "la" / "la_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The Lao census asks ethnic group, not language. Each of the 49 official groups is drawn "
        "as its language. No survey publishes how many in each group speak Lao at home instead, "
        "so nobody is moved onto Lao, though some Tai groups and town dwellers will have "
        "shifted. Per village the census counts only ten broad groupings, so Lao, Hmong and Iu "
        "Mien are exact by village, while which Katuic village is Katang and which is Bru "
        "follows the 2011 agricultural census. About 590,000 Tai subgroups the official list "
        "counts as Lao, such as the Phuan, are drawn as Lao."),
)
