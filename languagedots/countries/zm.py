# Zambia. Census 2022, predominant language of communication at home (sources/zm_census.py):
# nine language groups measured per constituency, rural and urban, split into ~85 languages by
# the province's rural or urban mix. Placed on religiondots' Kontur hexes, each constituency cut
# into a town and a countryside part (sources/zm_geo.py). The record is sources/zm.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import zm2022
    df = pd.read_csv(NORM / "zm.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 156:
        raise SystemExit(f"zm.csv: {df['geo_id'].nunique()} constituencies, expected 156")
    df["node"] = df["source_category"].map(zm2022.NAMES)
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(zm2022.EXCLUDED))
    if unresolved:
        raise SystemExit(f"zm.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna()].copy()
    df["unit"] = df["geo_id"] + "-" + df["residence"]
    # every row is the constituency's measured group count x its province's language mix
    df["tier"] = "derived"
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Zambia",
    source="2022 Census of Population and Housing, Series C1 Language Descriptive Tables "
           "(Zambia Statistics Agency, 2026)",
    how="census, 2022, predominant language of communication at home; languages by province, "
        "language groups by constituency",
    parts=[dict(covers="Everyone able to speak",
                source="2022 census, main language at home; constituency language groups split "
                       "by the province's languages",
                rest=True)],
    grain="156 constituencies, 117,000 people on average, each split into town and countryside",
    gap="1,216,763 people, 6.7%, mostly babies not yet able to speak; and the 47,941 people in "
        "institutions",
    view=[21.9, -18.1, 33.8, -8.2],
    counts=_counts,
    mappings=["zm2022"],
    place=GEO / "zm" / "zm_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked each person's main language at home and names about 80 Zambian "
        "languages, but it prints them only by province. For each constituency it gives nine "
        "language groups, for town and countryside separately. So inside a constituency, each "
        "group is shared among "
        "its languages in the proportions of the same group in the province's towns or "
        "countryside. A constituency's split between Chewa and Nyanja, or between Lala and "
        "Bemba, follows its province rather than its own count. The question was about the "
        "language used most at home, which in the towns is often Bemba, Nyanja or English "
        "rather than the language of a person's ethnic group."),
)
