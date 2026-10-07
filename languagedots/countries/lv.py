# Latvia. Tautas skaitīšana 2011, language mostly spoken at home, by municipality
# (sources/lv_census.py), on Kontur hexes keyed to religiondots' 119 LAUs (sources/lv_geo.py).
# The record is sources/lv.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import lv2011
    df = pd.read_csv(NORM / "lv.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 119:
        raise SystemExit(f"lv.csv: {df['geo_id'].nunique()} municipalities, expected 119")
    # census code "LV" + ATVK; the hexes carry the bare seven-digit ATVK (GISCO's LAU_ID)
    df["unit"] = df["geo_id"].str.removeprefix("LV")
    df["node"] = df["source_category"].map(lv2011.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    return by_unit(df)


ENTRY = dict(
    name="Latvia",
    source="Tautas skaitīšana 2011, resident population by language mostly spoken at home, by "
           "municipality (Central Statistical Bureau, table TSG11-07)",
    how="census, 2011, language mostly spoken at home",
    parts=[dict(covers="Everyone", source="2011 census, language mostly spoken at home",
                rest=True)],
    grain="119 municipalities, 17,400 people on average",
    gap="193,559 people (9.3%) whose home language the census did not record.",
    view=[20.8, 55.6, 28.3, 58.1],
    counts=_counts,
    mappings=["lv2011"],
    place=GEO / "lv" / "lv_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "These figures are from 2011, the last census to ask about language. The 2021 census "
        "was compiled from registers, which do not record language. Latvia has lost about a "
        "tenth of its people since 2011, and the Ukrainians who arrived after 2022 are not "
        "counted here. "
        "Latgalian, which the law treats as a variety of Latvian, was not an answer to the home "
        "language question, so its speakers are drawn as Latvian. 35.5% of people in Latgale "
        "said they use it every day; in a 2022 survey 8.8% there named it as their language at "
        "home."),
)
