# Estonia. Rahvaloendus 2021 mother tongue by municipality, town and city district
# (sources/ee_census.py), on Kontur hexes keyed to those 127 parts (sources/ee_geo.py). The
# record is sources/ee.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ee2021
    df = pd.read_csv(NORM / "ee.csv", dtype={"geo_id": str}, keep_default_na=False)
    df = df[df["geo_level"] == "part"].copy()
    if df["geo_id"].nunique() != 127:
        raise SystemExit(f"ee.csv: {df['geo_id'].nunique()} parts, expected 127")
    # the hex layer's `unit` is the census place code itself (sources/ee_geo.py)
    df["unit"] = df["geo_id"]
    df["count"] = df["count"].astype(int)
    df["node"] = df["source_category"].map(ee2021.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    return by_unit(df)


ENTRY = dict(
    name="Estonia",
    source="Rahvaloendus 2021, table RL21434, population by mother tongue and place of "
           "residence (Statistics Estonia)",
    how="census, 2021, mother tongue",
    parts=[dict(covers="Everyone", source="2021 census, mother tongue", rest=True)],
    grain="127 municipalities, towns and city districts, 10,500 people on average",
    gap="Mother tongue unknown: 8,176 people, 0.6%.",
    view=[21.5, 57.4, 28.3, 59.8],
    counts=_counts,
    mappings=["ee2021"],
    place=GEO / "ee" / "ee_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Mother tongue (emakeel) is the language a person names as their own, which in former "
        "Soviet countries leans towards identity rather than everyday use. Statistics Estonia "
        "publishes 15 languages by place; the other 229 languages it counts nationally, 15,848 "
        "people, are drawn in grey as other languages. Võro and Seto speakers are counted as "
        "Estonian. About 30,700 people gave two mother tongues, most of them Estonian and "
        "Russian; each is drawn once, on the language they gave first."),
)
