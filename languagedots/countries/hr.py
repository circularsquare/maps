# Croatia. Popis 2021 mother tongue by town/municipality (sources/hr_census.py), on Kontur hexes
# keyed to religiondots' 556 GISCO LAU polygons (sources/hr_geo.py). The record is sources/hr.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import hr2021
    df = pd.read_csv(NORM / "hr.csv")
    df = df[df["geo_level"].isin(["municipality", "city_district"])].copy()
    if df["geo_id"].nunique() != 572:
        raise SystemExit(f"hr.csv: {df['geo_id'].nunique()} units, expected 572 "
                         "(555 municipalities and Zagreb's 17 districts)")
    # The census has no codes; religiondots' lookup (read only) routes each "ZUPANIJA|NAME" to
    # its LAU code, and all 17 Zagreb districts to 01333: no district boundaries were found.
    lut = pd.read_csv(RD_GEO / "hr" / "hr_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"])))
    if df["unit"].isna().any():
        raise SystemExit(f"hr: {df.loc[df['unit'].isna(), 'geo_id'].nunique()} units missing "
                         "from religiondots' hr_lookup.csv")
    df["node"] = df["source_category"].map(hr2021.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    return by_unit(df)


ENTRY = dict(
    name="Croatia",
    source="Popis stanovništva 2021, population by mother tongue, by towns and municipalities "
           "(Državni zavod za statistiku)",
    how="census, 2021, mother tongue",
    parts=[dict(covers="Everyone", source="2021 census, mother tongue", rest=True)],
    grain="556 towns and municipalities, 7,000 people on average; Zagreb, a fifth of the "
          "country, is one",
    gap="20,840 people, 0.5%, whose mother tongue is recorded as unknown",
    view=[13.3, 42.3, 19.5, 46.6],
    counts=_counts,
    mappings=["hr2021"],
    place=GEO / "hr" / "hr_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Mother tongue is the language first learned in childhood. Croatian, Serbian, Bosnian "
        "and Montenegrin are one language by linguists' measure; the census prints each apart, "
        "and also Serbo-Croatian and Croato-Serbian, two orders of one Yugoslav-era name, and "
        "so does this map. Many of Croatia's Roma speak Boyash, a form of Romanian, which the "
        "census has no answer for, so its speakers are counted under Romani, Romanian or "
        "Croatian. The census counts Zagreb's 17 districts, but no boundaries for them were "
        "found, so the city is drawn as one unit."),
)
