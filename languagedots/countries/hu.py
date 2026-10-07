# Hungary. Népszámlálás 2022, mother tongue by settlement (sources/hu_census.py), on Kontur hexes
# keyed to religiondots' 3,177 settlements, Budapest as its 23 districts (sources/hu_geo.py).
# The record is sources/hu.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import hu2022
    df = pd.read_csv(NORM / "hu.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "settlement"]
    if df["geo_id"].nunique() != 3177:
        raise SystemExit(f"hu.csv: {df['geo_id'].nunique()} settlements, expected 3,177")
    df["node"] = df["source_category"].map(hu2022.resolve)
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"]) - hu2022.NOT_STATED)
    if unresolved:
        raise SystemExit(f"hu.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"]
    # `measured`: people who named one mother tongue; `derived`: half of each two-answer person,
    # and KSH's suppressed 1-or-2 cells estimated from the járás and vármegye
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Hungary",
    source="Népszámlálás 2022, table WBS003, mother tongue by settlement (Hungarian Central "
           "Statistical Office, KSH)",
    how="census, 2022, mother tongue, one or two allowed; a person who named two is shared "
        "between them",
    parts=[dict(covers="Everyone who answered", source="2022 census, mother tongue, one or "
                "two allowed", rest=True)],
    grain="3,177 settlements and Budapest districts, 3,000 people on average",
    gap="1,175,656 people (12.2%) who did not answer the optional question",
    view=[16.0, 45.6, 23.0, 48.7],
    counts=_counts,
    mappings=["hu2022"],
    place=GEO / "hu" / "hu_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "58,000 people named two mother tongues; each is drawn half to Hungarian and half to "
        "the other language. By settlement the census prints only Hungarian and the 13 "
        "recognised minority languages, so Russian, Chinese, Vietnamese and every other "
        "language are drawn as one other group. Boyash, the Romanian dialect of Roma in the "
        "south-west, is drawn apart from Romani."),
)
