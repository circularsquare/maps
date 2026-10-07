# Mongolia. 2020 census ethnic group by aimag (sources/mn_census.py), read as language through
# taxonomy/mn2020.py: Anita's 2026-10-05 ethnicity ruling. Placed on religiondots' 400m Kontur hexes
# (read-only); its Ulaanbaatar hexes are keyed by düüreg and are folded back into the city here,
# since no table gives ethnicity by düüreg. The record is sources/mn.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import mn2020
    df = pd.read_csv(NORM / "mn.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 22:
        raise SystemExit("mn.csv: expected 21 aimags and Ulaanbaatar; re-run sources/mn_census.py")
    df["node"] = df["source_category"].map(mn2020.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"mn.csv categories that resolve to nothing: {missing}")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _unit(g):
    # religiondots keys the capital's hexes by düüreg (MN11xx); the census table is city-wide
    u = g["unit"].astype(str)
    return u.where(~u.str.startswith("MN11"), "MN11")


ENTRY = dict(
    name="Mongolia",
    source=("2020 Population and Housing Census, national report appendix tables 1.1, 3.6, 3.6a "
            "and 4.1 (ethnic group), 3.6 and 3.7 (foreign citizens) (National Statistics Office "
            "of Mongolia)"),
    how=("census, 2020, ethnic group, each group drawn as its language; foreign citizens by "
         "citizenship"),
    parts=[
        dict(covers="Mongolian citizens",
             source="2020 census, ethnic group, each group drawn as its language",
             people=3_174_557),
        dict(covers="Foreign citizens",
             source="2020 census, citizenship, national split applied in every province",
             rest=True),
    ],
    grain="21 aimags and Ulaanbaatar, 145,000 people on average",
    gap="37 stateless people; the census asked no language question",
    view=[87.5, 41.4, 120.0, 52.3],
    counts=_counts,
    mappings=["mn2020"],
    place=RD_GEO / "mn" / "mn_hexes.gpkg",
    place_unit=_unit,
    place_weight=pop_weight,
    note_public=(
        "Mongolia's census asks ethnic group, not language, so each group is drawn as the "
        "language it is known for. The many Mongol groups are drawn by the dialect group they "
        "speak: Oirat for the Durvud, Bayad, Zakhchin, Torguud, Uriankhai, Darkhad and others "
        "of the west and north, Buryat for the Buryat and Barga, and Mongolian (Khalkha) for "
        "the Khalkh, Dariganga and the rest. Oirat and Buryat are usually counted as dialects "
        "of Mongolian inside Mongolia, schooling is in Khalkha, and no survey says how many "
        "of these groups still speak their own variety, especially in Ulaanbaatar, so the "
        "Oirat and Buryat dots are an upper bound. "
        "The Kazakhs of Bayan-Olgii are 91% of that province and keep Kazakh as their home "
        "and school language; the Tuvans live mostly in Tsengel in the same province. "
        "Foreign citizens, 0.7% of residents, are drawn by the national split of their "
        "citizenships (China, Russia, Korea, the United States, other), since no table gives it "
        "by province."),
)
