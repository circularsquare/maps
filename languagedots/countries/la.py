# Laos. 2015 census ethnicity at village level (K4D's ten ethno-linguistic categories), split into
# the 49 official groups on the 2011 agricultural census's village groups and raked to the
# census's national table (sources/la_census.py), read as languages through taxonomy/la2015.py:
# a proxy Anita allowed on 2026-10-05. Since 2026-10-09 part of each non-Tai group is moved onto
# Lao by the share of its children who speak Lao at home in LSIS III (MICS6) 2023
# (sources/la_mics.py -> la_mics.csv). Placed on religiondots' 400m Kontur hexes for the same
# 8,499 villages (read-only). The record is sources/la.md.
from _shared import *  # noqa: F401,F403

# False draws the census groups with no move onto Lao (la.csv, the build before 2026-10-09).
RETENTION = True
TOTAL = 6_481_482
# People per part in la_mics.csv, checked in _counts
LAO_TAI, SHIFTED, OTHER = 4_021_109, 812_550, 115_924
TAI = {"Lao", "Tai", "Phouthay", "Lue", "Nhoaun", "Yang", "Xaek", "Thaineau"}
# Rows whose village counts are the census's own: Lao (its own category) and the residual.
EXACT = {"Lao", "Other, not stated and foreigners"}


def _counts():
    import la2015
    name = "la_mics.csv" if RETENTION else "la.csv"
    df = pd.read_csv(NORM / name, dtype={"geo_id": str})
    if df["geo_id"].nunique() != 8499 or int(df["count"].sum()) != TOTAL:
        raise SystemExit(f"{name}: expected 8,499 villages and {TOTAL:,} people; re-run "
                         "sources/la_census.py, then sources/la_mics.py")
    if RETENTION:
        if set(df["source_id"]) != {"lsis3_2023_fl7"}:
            raise SystemExit("la_mics.csv: unexpected source_id; re-run sources/la_mics.py")
        sh = df["source_category"].str.startswith(la2015.SHIFT)
        got = (int(df.loc[df["source_category"].isin(TAI), "count"].sum()),
               int(df.loc[sh, "count"].sum()),
               int(df.loc[df["source_category"].str.startswith("Other"), "count"].sum()))
        if got != (LAO_TAI, SHIFTED, OTHER):
            raise SystemExit(f"la_mics.csv parts {got}; update LAO_TAI, SHIFTED, OTHER and parts")
    df["node"] = df["source_category"].map(la2015.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"{name} categories that resolve to nothing: {missing}")
    df["tier"] = df["source_category"].map(lambda s: "derived" if s in EXACT else "modelled")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Laos",
    source=("Population and Housing Census 2015, ethno-linguistic category by village (Lao "
            "Statistics Bureau, through the K4D atlas platform) and Table P2.7, population by "
            "ethnic group; 2011 Lao Census of Agriculture, the three largest ethnic groups of "
            "each village (same platform); Lao Social Indicator Survey III 2023 (MICS6; Lao "
            "Statistics Bureau, UNICEF), microdata, children's home language"),
    how=("census, 2015, ethnic group (no language question), each group drawn as its language, "
         "less the share of each non-Tai group whose children speak Lao at home in a 2023 "
         "household survey, drawn as Lao"),
    parts=[
        dict(covers="Lao and the Tai groups",
             source="2015 census, ethnic group by village, drawn as their languages",
             people=LAO_TAI),
        dict(covers="Other ethnic groups speaking Lao",
             source="LSIS III 2023, about 3,200 children of these groups, language spoken at "
                    "home",
             people=SHIFTED),
        dict(covers="Other, not stated and foreigners",
             source="2015 census, drawn as other",
             nodes=["other"]),
        dict(covers="Other ethnic groups on their own languages",
             source="2015 census, national totals by group, placed on villages by the 2011 "
                    "agricultural census",
             rest=True),
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
        "as its language, except that part of each group outside the Tai family is drawn as "
        "Lao: the share whose children aged 7 to 14 speak Lao most of the time at home, from "
        "UNICEF's 2023 household survey (37% of Khmu, 17% of Hmong, 11% of Akha). Children "
        "have shifted further than their parents, so this probably draws too many as Lao. "
        "Within a group the Lao speakers are put where the group is a smaller share of the "
        "village, as in the survey. The Tai groups are not moved, because the survey could "
        "only answer Lao or other. Per village the census counts only ten broad groupings, so "
        "which Katuic village is Katang and which is Bru follows the 2011 agricultural census. "
        "About 590,000 Tai subgroups the official list counts as Lao, such as the Phuan, are "
        "drawn as Lao."),
)
