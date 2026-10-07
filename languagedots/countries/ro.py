# Romania. RPL 2021 mother tongue by UAT (sources/ro_census.py), on Kontur hexes keyed to
# religiondots' 3,181 UATs (sources/ro_geo.py). The record is sources/ro.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ro2021
    df = pd.read_csv(NORM / "ro.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "uat"]
    if df["geo_id"].nunique() != 3181:
        raise SystemExit(f"ro.csv: {df['geo_id'].nunique()} UATs, expected 3,181")
    # the census has no codes; religiondots' lookup resolves "JUDEŢ|NAME" to SIRUTA (read only)
    lut = pd.read_csv(RD_GEO / "ro" / "ro_uat_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"])))
    if df["unit"].isna().any():
        raise SystemExit(f"ro: {df.loc[df['unit'].isna(), 'geo_id'].nunique()} UATs missing "
                         "from religiondots' ro_uat_lookup.csv")
    df["node"] = df["source_category"].map(ro2021.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    # `derived` rows are INS's `*` cells, estimated from the unit and judeţ margins
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Romania",
    source="Recensământul Populaţiei şi Locuinţelor 2021, table 2.3, mother tongue by "
           "commune and town (INS)",
    how="census, 2021, mother tongue; missing for 13%",
    parts=[dict(covers="Everyone", source="2021 census, mother tongue", rest=True)],
    grain="3,181 communes and towns, 6,000 people on average",
    gap="2,502,378 people (13.1%) whose mother tongue the census could not establish",
    view=[20.2, 43.5, 30.0, 48.4],
    counts=_counts,
    mappings=["ro2021"],
    place=GEO / "ro" / "ro_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Mother tongue could not be established for 13% of Romania. The 2021 census was "
        "built largely from administrative registers, which do not record language, so those "
        "2.5 million people are not drawn. The census names 22 languages, those of Romania's "
        "recognised minorities with Italian, Greek and Yiddish, and puts everything else under "
        "one other. Statistics Romania hides a commune's figure for a language when it is very "
        "small; those are estimated from the commune's total and its county's figure for the "
        "language, about 24,000 people in all."),
)
