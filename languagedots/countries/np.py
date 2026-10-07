# Nepal. NPHC 2021 Table 3 (sources/np_nphc.py), on religiondots' Kontur hexes for 753 local levels.
from _shared import *  # noqa: F401,F403


def _counts():
    import np2021
    df = pd.read_csv(NORM / "np.csv")
    df = df[df["geo_level"] == "local"]
    df["node"] = df["source_category"].map(np2021.resolve)
    df = df[df["node"].notna()]
    # religiondots joins its NP-<p>-<d>-<l> ids to COD-AB p-codes with this lookup
    lut = pd.read_csv(RD_GEO / "np" / "np_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"np: {df.loc[df['unit'].isna(), 'geo_id'].nunique()} local levels "
                         "missing from religiondots' np_lookup.csv")
    return by_unit(df)


ENTRY = dict(
    name="Nepal",
    source="National Population and Housing Census 2021, Table 3 (National Statistics Office)",
    how="census, 2021, mother tongue",
    parts=[dict(covers="Everyone", source="2021 census, mother tongue", rest=True)],
    grain="753 local levels, 38,000 people on average",
    gap="the institutional population, 0.8%, which is published by district only",
    view=[80.0, 26.3, 88.3, 30.5],
    counts=_counts,
    mappings=["np2021"],
    place=RD_GEO / "np" / "np_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public="",
)
