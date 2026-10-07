# Czechia. SLDB 2021, mother tongue, by obec and city district (sources/cz_sldb.py), on Kontur hexes
# keyed to religiondots' finest cover of 6,250 obce + 142 city districts (sources/cz_geo.py).
# The record is sources/cz.md.
from _shared import *  # noqa: F401,F403

REPLACED = RD_GEO / "cz" / "cz_replaced.csv"     # the 8 obce that city districts subdivide


def _counts():
    import cz2021
    df = pd.read_csv(NORM / "cz.csv", dtype={"geo_id": str})
    # one level per place: city districts REPLACE the 8 statutory cities they subdivide, so the
    # two levels are alternatives and summing both would draw Prague, Brno, Ostrava... twice
    rep = set(pd.read_csv(REPLACED, dtype=str)["kod"])
    assert len(rep) == 8
    df = df[((df["geo_level"] == "municipality") & ~df["geo_id"].isin(rep))
            | (df["geo_level"] == "city_district")]
    if df["geo_id"].nunique() != 6388:
        raise SystemExit(f"cz.csv: {df['geo_id'].nunique()} units, expected 6,388")
    df["node"] = df["source_category"].map(cz2021.resolve)
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"]) - cz2021.NOT_STATED)
    if unresolved:
        raise SystemExit(f"cz.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"]
    # people who named one mother tongue are `measured`; half of each two-answer person, and the
    # kraj's split of the 43 labels ČSÚ does not print below kraj, are `derived`
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Czechia",
    source="Sčítání lidu, domů a bytů 2021, mother tongue by obec and city district "
           "(Czech Statistical Office)",
    how="census, 2021, mother tongue, one or two allowed; a person who named two is shared "
        "between them",
    parts=[dict(covers="Everyone", source="2021 census, mother tongue, one or two allowed",
                rest=True)],
    grain="6,388 municipalities and city districts, 1,650 people on average",
    gap="759,394 people, 7.2%, who did not answer the question, which was optional",
    view=[12.0, 48.5, 18.9, 51.1],
    counts=_counts,
    mappings=["cz2021"],
    place=GEO / "cz" / "cz_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census let each person name one or two mother tongues. 260,000 named two, 218,000 "
        "of them Czech and another language, and each of them is drawn half to each. For "
        "municipalities the Czech Statistical Office prints 13 languages; how many people in a "
        "municipality have some other mother tongue is known, but which one only for its "
        "region, so they are drawn with the region's mix. Moravian is drawn as the census "
        "names it, apart from Czech."),
)
