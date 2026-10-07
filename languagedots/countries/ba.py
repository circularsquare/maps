# Bosnia and Herzegovina. Popis 2013 mother tongue by municipality (sources/ba_census.py), on
# religiondots' Kontur hexes for the same 142 units (read only). The record is sources/ba.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ba2013
    df = pd.read_csv(NORM / "ba.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 142:
        raise SystemExit(f"ba.csv: {df['geo_id'].nunique()} municipalities, expected 142 "
                         "(141 and Brcko District)")
    # geo_id is religiondots' folded name, which its hex layer carries as `unit`;
    # sources/ba_census.py asserts the 142 match religiondots' units both ways.
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(ba2013.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    return by_unit(df)


ENTRY = dict(
    name="Bosnia and Herzegovina",
    source="Popis stanovništva 2013, Knjiga 2, table 6.1, population by mother tongue, by "
           "municipalities (Agencija za statistiku Bosne i Hercegovine)",
    how="census, 2013, mother tongue",
    parts=[dict(covers="Everyone", source="2013 census, mother tongue", rest=True)],
    grain="142 municipalities, 25,000 people on average",
    gap="7,487 people, 0.2%, whose mother tongue is recorded as unknown",
    view=[15.6, 42.5, 19.7, 45.35],
    counts=_counts,
    mappings=["ba2013"],
    place=RD_GEO / "ba" / "ba_grid_400m.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Bosnian, Croatian and Serbian are one language by linguists' measure, and the answer "
        "follows nationality: in the same census 99.6% of Bosniaks gave Bosnian, 98.1% of Serbs "
        "Serbian and 93.8% of Croats Croatian. The other names some people chose for the common "
        "language, such as Serbo-Croatian, are each drawn as given. Of the minority languages "
        "the census names only Romani, Albanian, Turkish, Ukrainian and German; any other "
        "answer is in Other. 2013 is the most recent census. Republika Srpska's statistics institute "
        "disputes which emigrants were counted as residents and publishes lower figures for "
        "its entity; these are the state agency's."),
)
