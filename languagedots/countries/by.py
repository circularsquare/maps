# Belarus. 2019 census language usually spoken at home, by raion and city (sources/by_census.py,
# Belstat's census database, cube tb503), on Kontur hexes keyed to OSM's raions and cities
# (sources/by_geo.py).
# sources/by.md is the record.
from _shared import *  # noqa: F401,F403

_LUT = GEO / "by" / "by_lookup.csv"


def _counts():
    import by2019
    df = pd.read_csv(NORM / "by.csv")
    # home language, not native (Anita, 2026-10-05: "it does seem like Russian majority is
    # reality"); the native-language rows stay in by.csv and in sources/by.md
    df = df[df["question"] == "home"].copy()
    lut = pd.read_csv(_LUT)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"by: {sorted(df.loc[df['unit'].isna(), 'geo_name'].unique())} not in "
                         "by_lookup.csv; re-run sources/by_geo.py")
    if df["unit"].nunique() != 129:
        raise SystemExit(f"by: {df['unit'].nunique()} units, expected 129")
    df["node"] = df["source_category"].map(by2019.resolve)
    df = df[df["node"].notna()]
    return by_unit(df)


ENTRY = dict(
    name="Belarus",
    source=("Population Census 2019, population by language usually spoken at home, by raion "
            "and city "
            "(National Statistical Committee of the Republic of Belarus, census database cube "
            "F503)"),
    how="census, 2019, language usually spoken at home",
    parts=[dict(covers="Everyone", source="2019 census, language usually spoken at home",
                rest=True)],
    grain="118 raions, 10 cities and Minsk, 73,000 people on average",
    gap="216,519 people, 2.30%, whose home language the census form left blank",
    view=[23.1, 51.2, 32.8, 56.2],
    counts=_counts,
    mappings=["by2019"],
    place=GEO / "by" / "by_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2019 census asked two questions: each person's native language and the language "
        "they usually speak at home. This map draws the home language, where 71.4% named Russian "
        "and 26.0% Belarusian. Native language, which in the countries of the former Soviet "
        "Union leans towards identity rather than everyday use, gives nearly the reverse: 54.1% "
        "Belarusian and 42.3% Russian. The gap is widest in the cities: in Brest 71% named "
        "Belarusian as their native language and 6% speak it at home."),
)
